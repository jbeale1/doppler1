#!/usr/bin/env python3
"""
doppler_db_build.py -- build/update the DOPPLER1 events database.

Scans all doppler1_speed_daily_*.csv files in DATA_DIR and (re)computes
tracks for any day whose source file is new or has changed (by mtime+size)
since it was last processed. Today's date is ALWAYS reprocessed, since that
file keeps growing throughout the day. Safe to run repeatedly -- unchanged
historical days are skipped, so incremental runs after the first full
backfill are fast.

Usage:
    python3 doppler_db_build.py               # incremental update
    python3 doppler_db_build.py --full         # reprocess every day regardless
                                                # of change detection (use after
                                                # an algorithm change, e.g. a new
                                                # artifact-detection rule, so
                                                # historical stats reflect it)
    python3 doppler_db_build.py --quiet        # suppress per-day progress lines

Suggested crontab entry on jbeale-mini (every 10 minutes, so "today" stays
reasonably fresh):
    */10 * * * * /usr/bin/python3 /path/to/doppler_db_build.py --quiet >> /var/log/doppler_db_build.log 2>&1
"""

import argparse
import os
import time

import doppler_common as dc
import doppler_db as db


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                     help="Reprocess every day, ignoring change detection")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-day progress lines")
    args = ap.parse_args()

    conn = db.get_connection()
    days = dc.list_available_days()
    today = dc.today_str()

    n_processed = 0
    n_skipped = 0
    for day_str in days:
        path = dc.day_csv_path(day_str)
        st = os.stat(path)
        mtime, size = st.st_mtime, st.st_size

        existing = db.get_day_record(conn, day_str)
        needs_update = (
            args.full
            or day_str == today
            or existing is None
            or existing["source_mtime"] != mtime
            or existing["source_size"] != size
        )
        if not needs_update:
            n_skipped += 1
            continue

        df = dc.load_day_df(day_str)
        tracks = dc.segment_tracks(df)
        db.upsert_day(conn, day_str, mtime, size, tracks, processed_at=time.time())
        n_processed += 1
        if not args.quiet:
            print(f"{day_str}: {len(tracks)} events (source {size} bytes)")

    conn.close()
    if not args.quiet:
        print(f"\nProcessed {n_processed} day(s), skipped {n_skipped} unchanged, "
              f"{len(days)} total day(s) found.")


if __name__ == "__main__":
    main()
