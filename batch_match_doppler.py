#!/usr/bin/env python3
"""
batch_match_doppler.py - Match CAMA images to DOPPLER1 events using a
per-direction, speed-scaled validity threshold, and write reliable
matches back into doppler1_events.db as image labels.

For each CAMA image, the nearest doppler event is found:
  - Eastbound (velocity_mph > 0): compare to nearest doppler start_epoch.
  - Westbound (velocity_mph < 0): compare to nearest doppler end_epoch.
  - Unknown direction (velocity_mph NULL): try both, keep whichever is closer.

Whether that nearest match counts as "reliable" is then decided by a
physical model fit separately for eastbound/westbound traffic (camera
and radar are on the same transverse line but the two lanes have
different camera-to-radar gap distances):

    threshold(speed) = gap_ft / speed_fps + intercept + sigma * MAD

fit via robust (median-based) regression on the candidate matches
themselves, so a handful of mislabeled/coincident-event matches don't
inflate the tolerance. Below --fit-min-mph (where cars can legitimately
pause, e.g. pedestrians) a flat --cutoff is used instead, since the
constant-speed assumption breaks down.

Reliable matches (matched=1) are written into a `cama_labels` table in
doppler1_events.db, linking each doppler event to the CAMA image(s)
that identify it.

Usage:
    python3 batch_match_doppler.py --cama-db cama_images.db --doppler-db doppler1_events.db \
        --out matches.csv
    python3 batch_match_doppler.py --limit 5000 --sample random --no-write-labels
"""

import argparse
import bisect
import csv
import sqlite3
from collections import Counter
from datetime import datetime


def load_doppler(doppler_con):
    rows = doppler_con.execute("""
        SELECT id, start_epoch, end_epoch, max_mph, direction, type, artifact_flag
        FROM events
    """).fetchall()
    starts = sorted((r[1], r[0]) for r in rows)  # (start_epoch, id)
    ends = sorted((r[2], r[0]) for r in rows)    # (end_epoch, id)
    meta = {r[0]: {'max_mph': r[3], 'direction': r[4], 'type': r[5], 'artifact_flag': r[6]}
            for r in rows}
    return starts, ends, meta


def nearest_candidates(sorted_pairs, epochs, target):
    """sorted_pairs: list of (epoch, id) sorted by epoch; epochs: parallel
    plain list of the epoch values (for bisect). Returns up to 2 nearby
    (epoch, id) pairs bracketing target."""
    idx = bisect.bisect_left(epochs, target)
    out = []
    if idx < len(sorted_pairs):
        out.append(sorted_pairs[idx])
    if idx > 0:
        out.append(sorted_pairs[idx - 1])
    return out


def best_match(cama_epoch, velocity_mph, starts, start_epochs, ends, end_epochs):
    """Return (abs_delta, doppler_id, match_field) for the closest doppler
    event to cama_epoch, respecting direction when known."""
    candidates = []

    if velocity_mph is None or velocity_mph > 0:
        for epoch, eid in nearest_candidates(starts, start_epochs, cama_epoch):
            candidates.append((abs(epoch - cama_epoch), eid, 'start'))

    if velocity_mph is None or velocity_mph < 0:
        for epoch, eid in nearest_candidates(ends, end_epochs, cama_epoch):
            candidates.append((abs(epoch - cama_epoch), eid, 'end'))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0]


def direction_of(doppler_direction):
    d = (doppler_direction or '').strip().lower()
    if 'east' in d:
        return 'east'
    if 'west' in d:
        return 'west'
    return None


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def fit_direction_model(rows, fit_min_mph, fit_bins):
    """Robust (median-binned) fit of delta ~= gap_ft/speed_fps + intercept
    for one direction's candidate matches. Returns dict with slope,
    intercept, mad, gap_ft, n_fit - or None if not enough data."""
    fit_rows = [r for r in rows if r['speed'] >= fit_min_mph]
    n_fit = len(fit_rows)
    if n_fit < 15:
        return None

    nbins = max(3, min(fit_bins, n_fit // 15))
    fit_sorted = sorted(fit_rows, key=lambda r: r['speed'])
    edges = [round(i * n_fit / nbins) for i in range(nbins + 1)]
    bin_pts = []
    for i in range(nbins):
        chunk = fit_sorted[edges[i]:edges[i + 1]]
        if len(chunk) < 3:
            continue
        bin_pts.append((median([c['speed'] for c in chunk]),
                         median([c['delta_s'] for c in chunk]),
                         len(chunk)))
    if len(bin_pts) < 2:
        return None

    xs = [1.0 / s for s, d, w in bin_pts]
    ys = [d for s, d, w in bin_pts]
    ws = [w for s, d, w in bin_pts]
    W = sum(ws)
    mean_x = sum(w * x for x, w in zip(xs, ws)) / W
    mean_y = sum(w * y for y, w in zip(ys, ws)) / W
    var_x = sum(w * (x - mean_x) ** 2 for x, w in zip(xs, ws))
    if var_x == 0:
        return None
    cov = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in zip(xs, ys, ws))
    slope = cov / var_x
    intercept = mean_y - slope * mean_x

    residuals = [r['delta_s'] - (slope / r['speed'] + intercept) for r in fit_rows]
    mad = median([abs(r) for r in residuals]) * 1.4826

    return {'slope': slope, 'intercept': intercept, 'mad': mad,
            'gap_ft': slope * 1.46667, 'n_fit': n_fit}


def ensure_labels_table(doppler_con):
    doppler_con.executescript("""
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

        -- Tracks the highest images.id / events.id seen on the last full
        -- (unfiltered) run, so a run with nothing new to match can bail
        -- out immediately instead of reloading and reprocessing the
        -- entire history every time cron fires with no new traffic.
        CREATE TABLE IF NOT EXISTS match_run_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            max_cama_id INTEGER NOT NULL,
            max_doppler_id INTEGER NOT NULL,
            last_run_at REAL NOT NULL
        );
    """)


def get_run_state(doppler_con):
    row = doppler_con.execute(
        "SELECT max_cama_id, max_doppler_id FROM match_run_state WHERE id=1").fetchone()
    return row  # (max_cama_id, max_doppler_id) or None


def set_run_state(doppler_con, max_cama_id, max_doppler_id):
    doppler_con.execute("""
        INSERT INTO match_run_state (id, max_cama_id, max_doppler_id, last_run_at)
        VALUES (1, ?, ?, strftime('%s','now'))
        ON CONFLICT(id) DO UPDATE SET
            max_cama_id=excluded.max_cama_id,
            max_doppler_id=excluded.max_doppler_id,
            last_run_at=excluded.last_run_at
    """, (max_cama_id, max_doppler_id))
    doppler_con.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cama-db', default='cama_images.db')
    ap.add_argument('--doppler-db', default='doppler1_events.db')
    ap.add_argument('--limit', type=int, default=None, help='max number of CAMA images to test')
    ap.add_argument('--sample', choices=['all', 'random'], default='all')
    ap.add_argument('--cutoff', type=float, default=15.0,
                     help='delta above this is never plausible, regardless of speed; also used '
                          'as the flat threshold below --fit-min-mph (default 15.0 s)')
    ap.add_argument('--fit-min-mph', type=float, default=10.0,
                     help='minimum doppler speed used to fit the per-direction speed-scaled '
                          'model (default 10.0)')
    ap.add_argument('--sigma', type=float, default=3.0,
                     help='number of MAD-based spread units above the fitted delta(speed) curve '
                          'allowed before a match is rejected (default 3.0)')
    ap.add_argument('--fit-bins', type=int, default=12,
                     help='number of equal-count speed bins used for the robust median fit '
                          '(default 12)')
    ap.add_argument('--out', default='doppler_matches.csv', help='CSV output path')
    ap.add_argument('--write-labels', dest='write_labels', action='store_true', default=True,
                     help='write reliable matches into a cama_labels table in the doppler DB (default on)')
    ap.add_argument('--no-write-labels', dest='write_labels', action='store_false')
    ap.add_argument('--force', action='store_true',
                     help='run even if a full run already covered the current data '
                          '(bypasses the no-new-data early exit)')
    args = ap.parse_args()

    cama_con = sqlite3.connect(args.cama_db)
    cama_con.row_factory = sqlite3.Row
    doppler_con = sqlite3.connect(args.doppler_db)
    ensure_labels_table(doppler_con)

    is_full_run = args.limit is None and args.sample == 'all'
    max_cama_id = cama_con.execute("SELECT COALESCE(MAX(id), 0) FROM images").fetchone()[0]
    max_doppler_id = doppler_con.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()[0]

    if is_full_run and not args.force and args.write_labels:
        prev = get_run_state(doppler_con)
        if prev is not None and prev == (max_cama_id, max_doppler_id):
            print(f"No new CAMA images or doppler events since the last full run "
                  f"(max_cama_id={max_cama_id}, max_doppler_id={max_doppler_id}) - nothing to do. "
                  f"Use --force to run anyway.")
            return

    starts, ends, doppler_meta = load_doppler(doppler_con)
    start_epochs = [p[0] for p in starts]
    end_epochs = [p[0] for p in ends]
    print(f"Loaded {len(starts)} doppler events")

    sql = "SELECT id, filename, capture_dt, velocity_mph FROM images"
    if args.sample == 'random':
        sql += " ORDER BY RANDOM()"
    else:
        sql += " ORDER BY capture_dt"
    if args.limit:
        sql += f" LIMIT {args.limit}"

    cama_rows = cama_con.execute(sql).fetchall()
    print(f"Testing {len(cama_rows)} CAMA images")

    # --- pass 1: nearest-neighbor candidate match for every image ---
    results = []
    for row in cama_rows:
        try:
            cama_dt = datetime.fromisoformat(row['capture_dt'])
        except (TypeError, ValueError):
            continue
        cama_epoch = cama_dt.timestamp()
        m = best_match(cama_epoch, row['velocity_mph'], starts, start_epochs, ends, end_epochs)
        if m is None:
            delta, doppler_id, field = None, None, None
        else:
            delta, doppler_id, field = m
        meta = doppler_meta.get(doppler_id, {})
        speed = meta.get('max_mph')
        results.append({
            'cama_id': row['id'],
            'filename': row['filename'],
            'capture_dt': row['capture_dt'],
            'velocity_mph': row['velocity_mph'],
            'doppler_id': doppler_id,
            'doppler_max_mph': speed,
            'doppler_direction': meta.get('direction'),
            'doppler_type': meta.get('type'),
            'doppler_artifact_flag': meta.get('artifact_flag'),
            'delta_s': delta,
            'match_field': field,
            'speed': abs(speed) if speed is not None else None,
            'dir': direction_of(meta.get('direction')),
        })

    if not results:
        print("No CAMA rows processed.")
        return

    # --- pass 2: fit per-direction speed-scaled models on plausible candidates ---
    candidate_pool = [r for r in results
                       if r['delta_s'] is not None and r['delta_s'] <= args.cutoff
                       and r['speed'] is not None and r['dir'] is not None]

    models = {}
    for d in ('east', 'west'):
        rows_d = [r for r in candidate_pool if r['dir'] == d]
        model = fit_direction_model(rows_d, args.fit_min_mph, args.fit_bins)
        models[d] = model
        if model:
            print(f"\n{d.upper()} model (n={model['n_fit']}): "
                  f"gap={model['gap_ft']:.2f}ft  intercept={model['intercept']:.3f}s  "
                  f"MAD={model['mad']:.3f}s")
        else:
            print(f"\n{d.upper()}: not enough data to fit a speed-scaled model "
                  f"- falling back to flat {args.cutoff}s cutoff")

    def adaptive_threshold(speed, direction):
        model = models.get(direction)
        if model is None or speed is None or speed < args.fit_min_mph:
            return args.cutoff
        t = model['slope'] / speed + model['intercept'] + args.sigma * model['mad']
        return max(0.05, min(t, args.cutoff))

    # --- pass 3: classify every row against its direction's threshold ---
    for r in results:
        thresh = adaptive_threshold(r['speed'], r['dir'])
        r['match_threshold_s'] = round(thresh, 3)
        r['matched'] = int(r['delta_s'] is not None and r['delta_s'] <= thresh)

    out_fields = ['cama_id', 'filename', 'capture_dt', 'velocity_mph', 'doppler_id',
                  'doppler_max_mph', 'doppler_direction', 'doppler_type',
                  'doppler_artifact_flag', 'delta_s', 'match_field',
                  'match_threshold_s', 'matched']
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in out_fields})
    print(f"\nWrote {len(results)} rows to {args.out}")

    matched = [r for r in results if r['matched']]
    unmatched = [r for r in results if not r['matched']]
    print(f"\nMatched (adaptive per-direction threshold): {len(matched)} / {len(results)} "
          f"({100.0 * len(matched) / len(results):.1f}%)")
    print(f"No match found: {len(unmatched)}")

    if matched:
        deltas = sorted(r['delta_s'] for r in matched)
        n = len(deltas)

        def pct(p):
            return deltas[min(n - 1, int(p * n))]

        print(f"\nDelta stats (matched only, seconds):")
        print(f"  min    : {deltas[0]:.3f}")
        print(f"  median : {pct(0.50):.3f}")
        print(f"  p90    : {pct(0.90):.3f}")
        print(f"  p99    : {pct(0.99):.3f}")
        print(f"  max    : {deltas[-1]:.3f}")

    field_counts = Counter(r['match_field'] for r in matched)
    print(f"\nMatch field breakdown: {dict(field_counts)}")

    vel_known = [r for r in results if r['velocity_mph'] is not None]
    vel_unknown = [r for r in results if r['velocity_mph'] is None]
    if vel_unknown and vel_known:
        m_known = sum(1 for r in vel_known if r['matched'])
        m_unknown = sum(1 for r in vel_unknown if r['matched'])
        print(f"\nMatch rate with known velocity_mph:   {m_known}/{len(vel_known)} "
              f"({100.0 * m_known / len(vel_known):.1f}%)")
        print(f"Match rate with unknown velocity_mph: {m_unknown}/{len(vel_unknown)} "
              f"({100.0 * m_unknown / len(vel_unknown):.1f}%)")

    if args.write_labels:
        rows_to_write = [
            (r['doppler_id'], r['cama_id'], r['filename'], r['delta_s'], r['match_field'])
            for r in matched if r['doppler_id'] is not None
        ]
        doppler_con.executemany("""
            INSERT OR REPLACE INTO cama_labels
                (doppler_event_id, cama_id, cama_filename, delta_s, match_field)
            VALUES (?, ?, ?, ?, ?)
        """, rows_to_write)
        doppler_con.commit()
        n_events_labeled = doppler_con.execute(
            "SELECT COUNT(DISTINCT doppler_event_id) FROM cama_labels").fetchone()[0]
        print(f"\nWrote {len(rows_to_write)} label rows to cama_labels table in {args.doppler_db} "
              f"({n_events_labeled} distinct doppler events now labeled)")

        if is_full_run:
            set_run_state(doppler_con, max_cama_id, max_doppler_id)


if __name__ == '__main__':
    main()
