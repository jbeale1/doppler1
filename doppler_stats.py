#!/usr/bin/env python3
"""
doppler_stats.py -- print the compiled DOPPLER1 stats report from the
events database. Run doppler_db_build.py first (and periodically, via cron)
to keep the database up to date.

Usage:
    python3 doppler_stats.py [DAY | START END] [--start-time HH:MM] [--end-time HH:MM]

Examples:
    python3 doppler_stats.py                                  # today
    python3 doppler_stats.py 20260819                         # one day
    python3 doppler_stats.py 20260801 20260831                # a date range
    python3 doppler_stats.py 20260801 20260831 \\
        --start-time 07:00 --end-time 09:00                   # ...restricted to
                                                                # 7-9am on each day
                                                                # in that range
"""

import argparse

import doppler_common as dc
import doppler_db as db


def print_report(events, label, time_from=None, time_to=None):
    stats = dc.compute_stats(events)
    if stats is None:
        print(f"No events for {label}.")
        return

    start_str = dc.local_datetime_str_ms(stats["t_min"])
    end_str = dc.local_datetime_str_ms(stats["t_max"])
    print(f"Data period: {start_str} to {end_str}  ({stats['period_hours']:.2f} hours)")
    if time_from or time_to:
        print(f"Time-of-day filter: {time_from or '00:00'} to {time_to or '23:59:59'} (each day)")

    print("Summary (by direction / type):")
    header = f"{'':12}" + "".join(f"{t:>12}" for t in dc.TYPES) + f"{'total':>12}"
    print(header)
    col_totals = {t: 0 for t in dc.TYPES}
    for d in dc.DIRECTIONS:
        row_vals = []
        row_total = 0
        for t in dc.TYPES:
            n = stats["matrix"][d][t]
            row_vals.append(n)
            row_total += n
            col_totals[t] += n
        print(f"{d:12}" + "".join(f"{v:>12}" for v in row_vals) + f"{row_total:>12}")
    grand_total = sum(col_totals.values())
    print(f"{'total':12}" + "".join(f"{col_totals[t]:>12}" for t in dc.TYPES) + f"{grand_total:>12}")

    print("\nSpeed statistics (mph, by type):")
    pct_headers = "".join(f"{'p'+str(p):>8}" for p in dc.PERCENTILES if p != 50)
    print(f"{'type':12}{'n':>6}{'median':>10}{'mean':>10}{'min':>10}{'max':>10}{'stddev':>10}{'skew':>8}{'kurt':>8}{pct_headers}")
    for ty in dc.TYPES:
        s = stats["speed_stats"][ty]
        if s["n"] == 0:
            print(f"{ty:12}{0:>6}")
            continue
        pct_vals = "".join(f"{s['p'+str(p)]:>8.1f}" for p in dc.PERCENTILES if p != 50)
        print(f"{ty:12}{s['n']:>6}{s['median']:>10.1f}{s['mean']:>10.1f}"
              f"{s['min']:>10.1f}{s['max']:>10.1f}{s['std']:>10.1f}{s['skew']:>8.2f}{s['kurtosis']:>8.2f}{pct_vals}")

    print("\nSpeed statistics (mph, by direction / type):")
    print(f"{'direction':12}{'type':12}{'n':>6}{'median':>10}{'mean':>10}{'min':>10}{'max':>10}{'stddev':>10}{'skew':>8}{'kurt':>8}{pct_headers}")
    for d in dc.DIRECTIONS:
        for ty in dc.TYPES:
            s = stats["speed_stats_by_direction"][d][ty]
            if s["n"] == 0:
                print(f"{d:12}{ty:12}{0:>6}")
                continue
            pct_vals = "".join(f"{s['p'+str(p)]:>8.1f}" for p in dc.PERCENTILES if p != 50)
            print(f"{d:12}{ty:12}{s['n']:>6}{s['median']:>10.1f}{s['mean']:>10.1f}"
                  f"{s['min']:>10.1f}{s['max']:>10.1f}{s['std']:>10.1f}{s['skew']:>8.2f}{s['kurtosis']:>8.2f}{pct_vals}")

    print("\nSlowest / fastest by type:")
    time_fmt, time_col_label, time_col_width = dc.time_fmt_for_range(stats["t_min"], stats["t_max"])
    print(f"{'type':12}{'extreme':10}{'speed_mph':>10}{time_col_label:>{time_col_width}}")
    for ty in dc.TYPES:
        ex = stats["extremes"][ty]
        if ex is None:
            print(f"{ty:12}{'--':10}")
            continue
        for label2, row in [("slowest", ex["slowest"]), ("fastest", ex["fastest"])]:
            print(f"{ty:12}{label2:10}{row['max_mph']:>10.1f}{time_fmt(row['start_epoch']):>{time_col_width}}")

    print("\nSlowest / fastest by direction / type:")
    print(f"{'direction':12}{'type':12}{'extreme':10}{'speed_mph':>10}{time_col_label:>{time_col_width}}")
    for d in dc.DIRECTIONS:
        for ty in dc.TYPES:
            ex = stats["extremes_by_direction"][d][ty]
            if ex is None:
                print(f"{d:12}{ty:12}{'--':10}")
                continue
            for label2, row in [("slowest", ex["slowest"]), ("fastest", ex["fastest"])]:
                print(f"{d:12}{ty:12}{label2:10}{row['max_mph']:>10.1f}{time_fmt(row['start_epoch']):>{time_col_width}}")


def print_exceedance(events):
    exc = dc.compute_exceedance(events)
    print("\nExceedance thresholds (vehicle):")
    print(f"{'>= mph':>8}{'count':>10}{'pct':>10}")
    for row in exc["vehicle"]["thresholds"]:
        print(f"{row['mph']:>8}{row['count']:>10}{row['pct']:>9.1f}%")


def print_top_events(events, n, obj_type="vehicle"):
    top = dc.top_events(events, obj_type=obj_type, n=n)
    if not top:
        print(f"\nNo {obj_type} events found.")
        return
    t_min = min(e["start_epoch"] for e in top)
    t_max = max(e["start_epoch"] for e in top)
    time_fmt, time_col_label, time_col_width = dc.time_fmt_for_range(t_min, t_max)
    print(f"\nTop {len(top)} fastest {obj_type} events:")
    print(f"{'speed_mph':>10}{'direction':>12}{time_col_label:>{time_col_width}}")
    for e in top:
        print(f"{e['max_mph']:>10.1f}{e['direction']:>12}{time_fmt(e['start_epoch']):>{time_col_width}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dates", nargs="*", help="DAY, or START END for an inclusive date range (default: today)")
    ap.add_argument("--start-time", help="Only include events at/after this local time each day, e.g. 07:00")
    ap.add_argument("--end-time", help="Only include events at/before this local time each day, e.g. 09:00")
    ap.add_argument("--top", type=int, metavar="N",
                     help="Also list the N fastest vehicle events (with date/time) for direct inspection")
    args = ap.parse_args()

    if len(args.dates) == 0:
        start = end = dc.today_str()
        label = start
    elif len(args.dates) == 1:
        start = end = args.dates[0]
        label = start
    elif len(args.dates) == 2:
        start, end = args.dates
        label = f"{start}-{end}"
    else:
        ap.error("expected 0, 1, or 2 date arguments")
        return

    try:
        if args.start_time:
            dc.parse_time_of_day(args.start_time)
        if args.end_time:
            dc.parse_time_of_day(args.end_time)
    except ValueError as e:
        ap.error(str(e))
        return

    conn = db.get_connection()
    events = db.query_events(conn, date_from=start, date_to=end,
                              time_from=args.start_time, time_to=args.end_time)
    print_report(events, label, time_from=args.start_time, time_to=args.end_time)
    if events:
        print_exceedance(events)
    if args.top:
        print_top_events(events, args.top)
    conn.close()


if __name__ == "__main__":
    main()
