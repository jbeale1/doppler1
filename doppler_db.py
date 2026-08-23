#!/usr/bin/env python3
"""
doppler_db.py -- SQLite persistence for computed DOPPLER1 events.

Stores one row per computed track (the output of
doppler_common.segment_tracks()), NOT raw samples -- raw samples stay in
the daily CSV archives, which remain the source of truth. This is a derived
cache for fast per-day/date-range lookup and stats, avoiding re-parsing and
re-segmenting entire CSV files on every request.

Schema:
    days   -- bookkeeping: which source file (by mtime/size) has been
              processed into the events table, and when
    events -- one row per track, as computed by segment_tracks()

Build/update with doppler_db_build.py.
"""

import os
import sqlite3

import doppler_common as dc

DB_PATH = os.path.join(dc.DATA_DIR, "doppler1_events.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS days (
    date TEXT PRIMARY KEY,
    source_mtime REAL NOT NULL,
    source_size INTEGER NOT NULL,
    processed_at REAL NOT NULL,
    n_events INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    idx_in_day INTEGER NOT NULL,
    start_epoch REAL NOT NULL,
    end_epoch REAL NOT NULL,
    duration_s REAL NOT NULL,
    n_samples INTEGER NOT NULL,
    direction TEXT NOT NULL,
    type TEXT NOT NULL,
    max_kmh REAL NOT NULL,
    max_mph REAL NOT NULL,
    median_mph REAL,
    track_distance_ft REAL,
    artifact_flag INTEGER NOT NULL,
    overlap_split INTEGER NOT NULL DEFAULT 0,
    UNIQUE(date, idx_in_day)
);
CREATE INDEX IF NOT EXISTS idx_events_date ON events(date);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_maxmph ON events(max_mph);

-- Written by batch_match_doppler.py: links a track (events.id) to the
-- CAMA image(s) that reliably correspond to it, by acquisition-time
-- matching against cama_images.db. A track can have more than one row
-- if multiple CAMA frames matched; delta_s is the timing offset used to
-- judge the match, smallest = best.
CREATE TABLE IF NOT EXISTS cama_labels (
    id INTEGER PRIMARY KEY,
    doppler_event_id INTEGER NOT NULL,
    cama_id INTEGER NOT NULL,
    cama_filename TEXT NOT NULL,
    delta_s REAL NOT NULL,
    match_field TEXT NOT NULL,
    UNIQUE(doppler_event_id, cama_id)
);
CREATE INDEX IF NOT EXISTS idx_cama_labels_event ON cama_labels(doppler_event_id);

-- Written by batch_match_lidar_doppler.py: links a track (events.id) to
-- the LIDAR3 gate event that reliably corresponds to it, by acquisition-
-- time matching against lidar3_events.db. LIDAR-side fields are
-- denormalized in directly (unlike cama_labels' filename-only pointer)
-- since lidar3_events.db is a separate database file.
CREATE TABLE IF NOT EXISTS lidar_labels (
    id INTEGER PRIMARY KEY,
    doppler_event_id INTEGER NOT NULL,
    lidar_event_id INTEGER NOT NULL,
    lidar_l1_epoch REAL NOT NULL,
    lidar_l1_local_time TEXT NOT NULL,
    lidar_direction TEXT NOT NULL,
    lidar_type TEXT NOT NULL,
    lidar_match_type TEXT NOT NULL,
    lidar_speed_avg_mph REAL NOT NULL,
    lidar_speed_consistency_pct REAL,
    delta_s REAL NOT NULL,
    match_field TEXT NOT NULL,
    dop_compare_field TEXT NOT NULL,
    speed_diff_mph REAL,
    UNIQUE(doppler_event_id, lidar_event_id)
);
CREATE INDEX IF NOT EXISTS idx_lidar_labels_event ON lidar_labels(doppler_event_id);
"""


def _column_exists(conn, table, column):
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    return column in cols


def _migrate_schema(conn):
    """Add columns to an existing DB created before they existed. Existing
    rows get NULL until reprocessed (doppler_db_build.py --full backfills
    them); new rows always populate them via upsert_day()."""
    changed = False
    if not _column_exists(conn, "events", "median_mph"):
        conn.execute("ALTER TABLE events ADD COLUMN median_mph REAL")
        changed = True
    if not _column_exists(conn, "events", "track_distance_ft"):
        conn.execute("ALTER TABLE events ADD COLUMN track_distance_ft REAL")
        changed = True
    if not _column_exists(conn, "events", "overlap_split"):
        conn.execute("ALTER TABLE events ADD COLUMN overlap_split INTEGER NOT NULL DEFAULT 0")
        changed = True
    if changed:
        conn.commit()


def get_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate_schema(conn)
    return conn


def get_day_record(conn, day_str):
    row = conn.execute("SELECT * FROM days WHERE date = ?", (day_str,)).fetchone()
    return dict(row) if row else None


def upsert_day(conn, day_str, source_mtime, source_size, tracks, processed_at):
    """Replace all events for day_str with the given list of track dicts
    (as returned by doppler_common.segment_tracks()), and update the days
    bookkeeping row. Runs as a single transaction."""
    with conn:
        conn.execute("DELETE FROM events WHERE date = ?", (day_str,))
        conn.executemany(
            """INSERT INTO events
               (date, idx_in_day, start_epoch, end_epoch, duration_s, n_samples,
                direction, type, max_kmh, max_mph, median_mph, track_distance_ft,
                artifact_flag, overlap_split)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (day_str, t["idx"], t["start_epoch"], t["end_epoch"], t["duration_s"],
                 t["n_samples"], t["direction"], t["type"], t["max_kmh"], t["max_mph"],
                 t["median_mph"], t["track_distance_ft"], int(t["artifact_flag"]),
                 int(t["overlap_split"]))
                for t in tracks
            ],
        )
        conn.execute(
            """INSERT INTO days (date, source_mtime, source_size, processed_at, n_events)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(date) DO UPDATE SET
                 source_mtime=excluded.source_mtime,
                 source_size=excluded.source_size,
                 processed_at=excluded.processed_at,
                 n_events=excluded.n_events""",
            (day_str, source_mtime, source_size, processed_at, len(tracks)),
        )


def _row_to_event(row):
    d = dict(row)
    d["idx"] = d.pop("idx_in_day")
    d["artifact_flag"] = bool(d["artifact_flag"])
    return d


def query_events(conn, date_from=None, date_to=None, min_duration=dc.MIN_TRACK_DURATION_S,
                  time_from=None, time_to=None, max_speed_mph=dc.MAX_PLAUSIBLE_SPEED_MPH):
    """date_from/date_to are inclusive YYYYMMDD strings; pass the same value
    for both (or just date_from) for a single day. time_from/time_to are
    optional 'HH:MM' or 'HH:MM:SS' strings restricting each day's events to
    a local clock-time window (e.g. only 07:00-09:00 on every day in the
    range) -- applied in Python after the date-range SQL query, since "the
    same time window on every day" isn't an epoch range. A window where
    time_from > time_to is treated as wrapping past midnight (e.g.
    22:00-06:00). max_speed_mph excludes known-bad readings (e.g. RF
    interference); pass None to see everything, including those. Returns a
    list of dicts shaped like doppler_common.segment_tracks() output (minus
    'samples', which isn't stored in the DB -- read the source CSV via
    /plot/ for that)."""
    clauses = []
    params = []
    if date_from:
        clauses.append("date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("date <= ?")
        params.append(date_to)
    if min_duration is not None:
        clauses.append("duration_s >= ?")
        params.append(min_duration)
    if max_speed_mph is not None:
        clauses.append("max_mph <= ?")
        params.append(max_speed_mph)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(f"SELECT * FROM events {where} ORDER BY start_epoch", params).fetchall()
    events = [_row_to_event(r) for r in rows]

    if time_from or time_to:
        tf = dc.parse_time_of_day(time_from) if time_from else 0.0
        tt = dc.parse_time_of_day(time_to) if time_to else (24 * 3600 - 1e-6)

        def _in_window(e):
            tod = dc.time_of_day_seconds(e["start_epoch"])
            if tf <= tt:
                return tf <= tod <= tt
            return tod >= tf or tod <= tt  # wraps past midnight

        events = [e for e in events if _in_window(e)]

    return events


def get_cama_labels(conn, event_ids):
    """Best (smallest delta_s) CAMA image match for each of the given
    events.id values. Returns {event_id: {"filename": ..., "delta_s": ...}}
    for events that have at least one label; events with none are simply
    absent from the returned dict."""
    event_ids = [e for e in event_ids if e is not None]
    if not event_ids:
        return {}
    placeholders = ",".join("?" * len(event_ids))
    rows = conn.execute(
        f"""SELECT doppler_event_id, cama_filename, delta_s
            FROM cama_labels
            WHERE doppler_event_id IN ({placeholders})
            ORDER BY delta_s ASC""",
        event_ids,
    ).fetchall()
    out = {}
    for r in rows:
        eid = r["doppler_event_id"]
        if eid not in out:  # first row per event_id is the smallest delta_s
            out[eid] = {"filename": r["cama_filename"], "delta_s": r["delta_s"]}
    return out


def get_lidar_labels(conn, event_ids):
    """Best (smallest delta_s) LIDAR3 match for each of the given
    events.id values. Returns {event_id: {...}} for events that have at
    least one label; events with none are simply absent from the
    returned dict. lidar_labels lives in this same database (unlike
    cama_labels' source images, LIDAR match data is fully denormalized
    in at write time), so no cross-database lookup is needed."""
    event_ids = [e for e in event_ids if e is not None]
    if not event_ids:
        return {}
    placeholders = ",".join("?" * len(event_ids))
    rows = conn.execute(
        f"""SELECT doppler_event_id, lidar_speed_avg_mph, lidar_type,
                   dop_compare_field, speed_diff_mph, delta_s
            FROM lidar_labels
            WHERE doppler_event_id IN ({placeholders})
            ORDER BY delta_s ASC""",
        event_ids,
    ).fetchall()
    out = {}
    for r in rows:
        eid = r["doppler_event_id"]
        if eid not in out:  # first row per event_id is the smallest delta_s
            out[eid] = {
                "speed_avg_mph": r["lidar_speed_avg_mph"],
                "type": r["lidar_type"],
                "dop_compare_field": r["dop_compare_field"],
                "speed_diff_mph": r["speed_diff_mph"],
                "delta_s": r["delta_s"],
            }
    return out


def has_day(conn, day_str):
    return get_day_record(conn, day_str) is not None


def known_days(conn):
    rows = conn.execute("SELECT date FROM days ORDER BY date DESC").fetchall()
    return [r["date"] for r in rows]
