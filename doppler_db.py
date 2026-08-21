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
    artifact_flag INTEGER NOT NULL,
    UNIQUE(date, idx_in_day)
);
CREATE INDEX IF NOT EXISTS idx_events_date ON events(date);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_maxmph ON events(max_mph);
"""


def get_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
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
                direction, type, max_kmh, max_mph, artifact_flag)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (day_str, t["idx"], t["start_epoch"], t["end_epoch"], t["duration_s"],
                 t["n_samples"], t["direction"], t["type"], t["max_kmh"], t["max_mph"],
                 int(t["artifact_flag"]))
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


def has_day(conn, day_str):
    return get_day_record(conn, day_str) is not None


def known_days(conn):
    rows = conn.execute("SELECT date FROM days ORDER BY date DESC").fetchall()
    return [r["date"] for r in rows]
