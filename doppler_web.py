#!/usr/bin/env python3
"""
doppler_web.py -- browse DOPPLER1 vehicle speed data (radar) in a web browser.

Reads from the events database (doppler_db.py) for fast per-day and
date-range lookups -- built/kept up to date by doppler_db_build.py, which
should run periodically via cron. Falls back to live CSV parsing
(doppler_common.py) for any day not yet present in the database, so the
app still works correctly before a first backfill or if the builder falls
behind.

Per-track plots (/plot/) always read directly from the source CSV --
that's cheap (one track at a time, on demand) and avoids storing raw
samples in the database at all, keeping it lean.

Runs as its own standalone Flask app on its own port -- separate from any
existing web server (e.g. a bird-call page on port 5000).
"""

import math
import re
from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template_string, send_file, abort, redirect, url_for, request, jsonify

import doppler_common as dc
import doppler_db as db

PORT = 5001

app = Flask(__name__)


# ---------------------------------------------------------------------------
# DB-backed day lookup, with CSV fallback
# ---------------------------------------------------------------------------
def get_day_tracks(day_str):
    """Returns (tracks, file_exists). DB-backed if the day has been
    processed into the database; falls back to live CSV computation
    otherwise (e.g. a day not yet picked up by doppler_db_build.py)."""
    conn = db.get_connection()
    try:
        if db.has_day(conn, day_str):
            events = db.query_events(conn, date_from=day_str, date_to=day_str, min_duration=None)
            return events, True
    finally:
        conn.close()

    df = dc.load_day_df(day_str)
    if df is None:
        return [], False
    tracks = dc.segment_tracks(df)
    tracks = [t for t in tracks if t["max_mph"] <= dc.MAX_PLAUSIBLE_SPEED_MPH]
    return tracks, True


# ---------------------------------------------------------------------------
# Templates (inline -- single-file app)
# ---------------------------------------------------------------------------
BASE_CSS = """
<style>
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; color: #222; }
  h1 { font-size: 1.4em; }
  h3 { font-size: 1.05em; margin-top: 1.6em; }
  table { border-collapse: collapse; width: 100%; margin-top: 1em; font-size: 0.75em; }
  th, td { text-align: left; padding: 2px 10px; border-bottom: 1px solid #ddd; vertical-align: top; }
  th { background: #f4f4f4; }
  th a { color: #222; text-decoration: none; }
  th a:hover { text-decoration: underline; }
  tr:hover { background: #fafafa; }
  td.plotcell { width: 2em; }
  .nav { margin-bottom: 1em; }
  .nav a { margin-right: 1em; text-decoration: none; color: #06c; }
  .stats { color: #555; margin: 0.5em 0 1em; }
  .east { color: #b45309; }
  .west { color: #1d4ed8; }
  details summary { cursor: pointer; color: #06c; list-style: none; }
  details summary::-webkit-details-marker { display: none; }
  tr.plotrow td { border-bottom: 1px solid #ddd; padding: 4px 10px 12px; }
  img.trackplot { max-width: 650px; width: 100%; display: block; margin-top: 6px; border: 1px solid #ddd; }
  .daylist a { display: inline-block; margin: 2px 8px 2px 0; }
  form.rangeform input { width: 9em; }
</style>
<script>
  function togglePlot(i) {
    var row = document.getElementById('plotrow-' + i);
    row.style.display = (row.style.display === 'none') ? '' : 'none';
  }

  function updateHistograms() {
    var log = document.getElementById('hist-log').checked ? '1' : '0';
    var gaussChecked = document.getElementById('hist-gaussian').checked;
    var gauss = gaussChecked ? '1' : '0';
    var fitMin = document.getElementById('hist-fit-min').value.trim();
    var fitMax = document.getElementById('hist-fit-max').value.trim();
    ['vehicle', 'pedestrian'].forEach(function(type) {
      var img = document.getElementById('hist-' + type);
      var nanbDiv = document.getElementById('hist-' + type + '-nanb');
      if (!img) return;
      var url = new URL(img.src, window.location.href);
      url.searchParams.set('log', log);
      url.searchParams.set('gaussian', gauss);
      if (fitMin) { url.searchParams.set('fit_min', fitMin); } else { url.searchParams.delete('fit_min'); }
      if (fitMax) { url.searchParams.set('fit_max', fitMax); } else { url.searchParams.delete('fit_max'); }
      img.src = url.toString();

      if (!nanbDiv) return;
      if (!gaussChecked) {
        nanbDiv.textContent = '';
        return;
      }
      var statsUrl = new URL('/histogram_stats.json', window.location.href);
      ['start', 'end', 'start_time', 'end_time'].forEach(function(p) {
        var v = url.searchParams.get(p);
        if (v) statsUrl.searchParams.set(p, v);
      });
      statsUrl.searchParams.set('type', type);
      if (fitMin) statsUrl.searchParams.set('fit_min', fitMin);
      if (fitMax) statsUrl.searchParams.set('fit_max', fitMax);
      fetch(statsUrl).then(function(r) { return r.json(); }).then(function(d) {
        if (d.error) { nanbDiv.textContent = ''; return; }
        var pct = (100 * d.nb / d.n_total).toFixed(2);
        nanbDiv.textContent = 'Na (fits Gaussian) = ' + d.na + '   Nb (excess above fit, high side) = ' +
          d.nb + '  (' + pct + '% of ' + d.n_total + ' total)';
      }).catch(function() { nanbDiv.textContent = ''; });
    });
  }
</script>
"""

DAY_TEMPLATE = BASE_CSS + """
<h1>DOPPLER1 vehicle speeds &mdash; {{ display_date }}</h1>
<div class="nav">
  <a href="{{ url_for('day_view', day_str=prev_day) }}">&larr; prev day</a>
  <a href="{{ url_for('index') }}">today</a>
  <a href="{{ url_for('day_view', day_str=next_day) }}">next day &rarr;</a>
  <a href="{{ url_for('days_list') }}">browse all days</a>
  <a href="{{ url_for('stats_day', day_str=day_str) }}">full stats for this day &rarr;</a>
</div>

{% if is_today %}
<p style="color:#888; font-size:0.9em;">
  Auto-refreshes every 5 minutes. Underlying data syncs from the radar
  hourly (plus once more at midnight); the database is only as fresh as the
  last time doppler_db_build.py ran. Neither is instantaneous.
</p>
{% endif %}

{% if tracks %}
<div class="stats">
  {{ tracks|length }} events &middot;
  fastest: {{ "%.1f"|format(max_mph) }} mph &middot;
  {{ east_count }} east / {{ west_count }} west
</div>
<table>
  <tr>
    <th>#</th>
    <th><a href="{{ url_for('day_view', day_str=day_str, sort='time', dir=headers.time.next_dir) }}">Time (PDT){{ headers.time.arrow }}</a></th>
    <th>Dir</th>
    <th>Type</th>
    <th><a href="{{ url_for('day_view', day_str=day_str, sort='speed', dir=headers.speed.next_dir) }}">Max speed{{ headers.speed.arrow }}</a></th>
    <th><a href="{{ url_for('day_view', day_str=day_str, sort='duration', dir=headers.duration.next_dir) }}">Duration{{ headers.duration.arrow }}</a></th>
    <th>Samples</th>
    <th></th>
  </tr>
  {% for t in tracks %}
  <tr>
    <td>{{ loop.index }}</td>
    <td>{{ local_time_str(t.start_epoch) }}</td>
    <td class="{{ t.direction }}">{{ t.direction }}</td>
    <td>{{ t.type }}</td>
    <td>{{ "%.1f"|format(t.max_mph) }} mph{% if t.artifact_flag %} <span title="Raw data included what looks like a sensor artifact (a run of identical readings plus a physically implausible jump); those points were excluded from this max-speed calculation. Expand the plot to see them marked.">&#9888;</span>{% endif %}</td>
    <td>{{ "%.1f"|format(t.duration_s) }} s</td>
    <td>{{ t.n_samples }}</td>
    <td class="plotcell">
      <details ontoggle="togglePlot({{ t.idx }})">
        <summary>&#9654;</summary>
      </details>
    </td>
  </tr>
  <tr class="plotrow" id="plotrow-{{ t.idx }}" style="display:none;">
    <td colspan="8">
      <img class="trackplot" loading="lazy"
           src="{{ url_for('track_plot', day_str=day_str, track_idx=t.idx) }}">
    </td>
  </tr>
  {% endfor %}
</table>
{% else %}
<p>No vehicle data for this day{% if not file_exists %} (no file found){% endif %}.</p>
{% endif %}
"""

DAYS_LIST_TEMPLATE = BASE_CSS + """
<h1>DOPPLER1 &mdash; browse by day</h1>
<div class="nav">
  <a href="{{ url_for('index') }}">today</a>
  <a href="{{ url_for('stats_range') }}">date-range stats</a>
</div>
<div class="daylist">
{% for d in days %}
  <a href="{{ url_for('day_view', day_str=d) }}">{{ d }}</a>
{% endfor %}
</div>
{% if not days %}<p>No data files found in {{ data_dir }}.</p>{% endif %}
"""

STATS_TEMPLATE = BASE_CSS + """
<h1>DOPPLER1 stats &mdash; {{ label }}</h1>
<div class="nav">
  {% if day_str %}<a href="{{ url_for('day_view', day_str=day_str) }}">&larr; back to day view</a>{% endif %}
  <a href="{{ url_for('days_list') }}">browse all days</a>
</div>

<form class="rangeform" method="get" action="{{ url_for('stats_range') }}">
  from <input type="text" name="start" placeholder="YYYYMMDD" value="{{ start or '' }}">
  to <input type="text" name="end" placeholder="YYYYMMDD" value="{{ end or '' }}">
  time-of-day
  <input type="text" name="start_time" placeholder="HH:MM" value="{{ start_time or '' }}" style="width:5em;">
  to
  <input type="text" name="end_time" placeholder="HH:MM" value="{{ end_time or '' }}" style="width:5em;">
  <button type="submit">go</button>
</form>
{% if time_error %}<p style="color:#b91c1c;">{{ time_error }}</p>{% endif %}

{% if stats %}
<p>Data period: {{ start_str }} to {{ end_str }}  ({{ "%.2f"|format(stats.period_hours) }} hours)</p>
{% if start_time or end_time %}<p>Time-of-day filter: {{ start_time or '00:00' }} to {{ end_time or '23:59:59' }} (each day)</p>{% endif %}

<h3>Summary (by direction / type)</h3>
<table>
  <tr><th></th>{% for ty in types %}<th>{{ ty }}</th>{% endfor %}<th>total</th></tr>
  {% for d in directions %}
  <tr>
    <td>{{ d }}</td>
    {% for ty in types %}<td>{{ stats.matrix[d][ty] }}</td>{% endfor %}
    <td>{{ direction_totals[d] }}</td>
  </tr>
  {% endfor %}
  <tr>
    <td><b>total</b></td>
    {% for ty in types %}<td><b>{{ type_totals[ty] }}</b></td>{% endfor %}
    <td><b>{{ stats.n_total }}</b></td>
  </tr>
</table>

<h3>Speed statistics (mph, by type)</h3>
<table>
  <tr><th>type</th><th>n</th><th>median</th><th>mean</th><th>min</th><th>max</th><th>stddev</th><th>skew</th><th>kurt</th>{% for p in percentiles %}<th>p{{ p }}</th>{% endfor %}</tr>
  {% for ty in types %}
  {% set s = stats.speed_stats[ty] %}
  <tr>
    <td>{{ ty }}</td>
    {% if s.n %}
    <td>{{ s.n }}</td><td>{{ "%.1f"|format(s.median) }}</td><td>{{ "%.1f"|format(s.mean) }}</td>
    <td>{{ "%.1f"|format(s.min) }}</td><td>{{ "%.1f"|format(s.max) }}</td>
    <td>{{ "%.1f"|format(s.std) if s.n > 1 else "--" }}</td>
    <td>{{ "%.2f"|format(s.skew) if s.n > 2 else "--" }}</td>
    <td>{{ "%.2f"|format(s.kurtosis) if s.n > 3 else "--" }}</td>
    {% for p in percentiles %}<td>{{ "%.1f"|format(s["p"+p|string]) }}</td>{% endfor %}
    {% else %}
    <td colspan="{{ 8 + percentiles|length }}">0</td>
    {% endif %}
  </tr>
  {% endfor %}
</table>

<h3>Speed statistics (mph, by direction / type)</h3>
<table>
  <tr><th>direction</th><th>type</th><th>n</th><th>median</th><th>mean</th><th>min</th><th>max</th><th>stddev</th><th>skew</th><th>kurt</th>{% for p in percentiles %}<th>p{{ p }}</th>{% endfor %}</tr>
  {% for d in directions %}
  {% for ty in types %}
  {% set s = stats.speed_stats_by_direction[d][ty] %}
  <tr>
    <td class="{{ d }}">{{ d }}</td><td>{{ ty }}</td>
    {% if s.n %}
    <td>{{ s.n }}</td><td>{{ "%.1f"|format(s.median) }}</td><td>{{ "%.1f"|format(s.mean) }}</td>
    <td>{{ "%.1f"|format(s.min) }}</td><td>{{ "%.1f"|format(s.max) }}</td>
    <td>{{ "%.1f"|format(s.std) if s.n > 1 else "--" }}</td>
    <td>{{ "%.2f"|format(s.skew) if s.n > 2 else "--" }}</td>
    <td>{{ "%.2f"|format(s.kurtosis) if s.n > 3 else "--" }}</td>
    {% for p in percentiles %}<td>{{ "%.1f"|format(s["p"+p|string]) }}</td>{% endfor %}
    {% else %}
    <td colspan="{{ 8 + percentiles|length }}">0</td>
    {% endif %}
  </tr>
  {% endfor %}
  {% endfor %}
</table>

<h3>Slowest / fastest by type</h3>
<table>
  <tr><th>type</th><th>extreme</th><th>speed</th><th>{{ time_col_label }}</th></tr>
  {% for ty in types %}
    {% set ex = stats.extremes[ty] %}
    {% if ex %}
    <tr><td>{{ ty }}</td><td>slowest</td><td>{{ "%.1f"|format(ex.slowest.max_mph) }} mph</td><td>{{ time_fmt(ex.slowest.start_epoch) }}</td></tr>
    <tr><td>{{ ty }}</td><td>fastest</td><td>{{ "%.1f"|format(ex.fastest.max_mph) }} mph</td><td>{{ time_fmt(ex.fastest.start_epoch) }}</td></tr>
    {% endif %}
  {% endfor %}
</table>

<h3>Slowest / fastest by direction / type</h3>
<table>
  <tr><th>direction</th><th>type</th><th>extreme</th><th>speed</th><th>{{ time_col_label }}</th></tr>
  {% for d in directions %}
  {% for ty in types %}
    {% set ex = stats.extremes_by_direction[d][ty] %}
    {% if ex %}
    <tr><td class="{{ d }}">{{ d }}</td><td>{{ ty }}</td><td>slowest</td><td>{{ "%.1f"|format(ex.slowest.max_mph) }} mph</td><td>{{ time_fmt(ex.slowest.start_epoch) }}</td></tr>
    <tr><td class="{{ d }}">{{ d }}</td><td>{{ ty }}</td><td>fastest</td><td>{{ "%.1f"|format(ex.fastest.max_mph) }} mph</td><td>{{ time_fmt(ex.fastest.start_epoch) }}</td></tr>
    {% endif %}
  {% endfor %}
  {% endfor %}
</table>

<h3>Speed distribution</h3>
<div style="margin-bottom:6px;">
  <label><input type="checkbox" id="hist-log" onchange="updateHistograms()"> log scale (count axis)</label>
  &nbsp;&nbsp;
  <label><input type="checkbox" id="hist-gaussian" onchange="updateHistograms()"> show Gaussian fit</label>
  &nbsp;&nbsp;
  fit range (mph):
  <input type="text" id="hist-fit-min" placeholder="min" style="width:4em;" onchange="updateHistograms()">
  to
  <input type="text" id="hist-fit-max" placeholder="max" style="width:4em;" onchange="updateHistograms()">
</div>
{% if stats.speed_stats.vehicle.n %}
<img class="trackplot" loading="lazy" id="hist-vehicle"
     src="{{ url_for('speed_histogram', start=hist_start, end=hist_end, start_time=start_time, end_time=end_time, type='vehicle', log='0', gaussian='0') }}">
<div id="hist-vehicle-nanb" style="font-size:0.85em; color:#333; margin:4px 0 12px;"></div>
{% endif %}
{% if stats.speed_stats.pedestrian.n %}
<img class="trackplot" loading="lazy" id="hist-pedestrian"
     src="{{ url_for('speed_histogram', start=hist_start, end=hist_end, start_time=start_time, end_time=end_time, type='pedestrian', log='0', gaussian='0') }}">
<div id="hist-pedestrian-nanb" style="font-size:0.85em; color:#333; margin:4px 0 12px;"></div>
{% endif %}

<h3>Exceedance thresholds (vehicle)</h3>
<table>
  <tr><th>&ge; mph</th><th>count</th><th>pct</th></tr>
  {% for row in exceedance.vehicle.thresholds %}
  <tr><td>{{ row.mph }}</td><td>{{ row.count }}</td><td>{{ "%.1f"|format(row.pct) }}%</td></tr>
  {% endfor %}
</table>

<h3>Top {{ top_vehicle_events|length }} fastest vehicle events</h3>
<table>
  <tr><th>#</th><th>{{ time_col_label }}</th><th>Dir</th><th>Max speed</th><th></th></tr>
  {% for e in top_vehicle_events %}
  <tr>
    <td>{{ loop.index }}</td>
    <td>{{ time_fmt(e.start_epoch) }}</td>
    <td class="{{ e.direction }}">{{ e.direction }}</td>
    <td>{{ "%.1f"|format(e.max_mph) }} mph</td>
    <td class="plotcell">
      <details ontoggle="togglePlot('top-{{ loop.index0 }}')">
        <summary>&#9654;</summary>
      </details>
    </td>
  </tr>
  <tr class="plotrow" id="plotrow-top-{{ loop.index0 }}" style="display:none;">
    <td colspan="5">
      <img class="trackplot" loading="lazy"
           src="{{ url_for('track_plot', day_str=e.date, track_idx=e.idx) }}">
    </td>
  </tr>
  {% endfor %}
</table>
{% else %}
<p>No events found for this period.</p>
{% endif %}
"""


def render_stats_page(events, label, day_str=None, start=None, end=None,
                       start_time=None, end_time=None, time_error=None, top_n=20):
    stats = dc.compute_stats(events)
    direction_totals = {}
    type_totals = {ty: 0 for ty in dc.TYPES}
    start_str = end_str = None
    time_fmt, time_col_label = dc.local_time_str_ms, "PDT time"
    exceedance = None
    top_vehicle_events = []
    if stats:
        for d in dc.DIRECTIONS:
            direction_totals[d] = sum(stats["matrix"][d].values())
        for ty in dc.TYPES:
            type_totals[ty] = sum(stats["matrix"][d][ty] for d in dc.DIRECTIONS)
        start_str = dc.local_datetime_str_ms(stats["t_min"])
        end_str = dc.local_datetime_str_ms(stats["t_max"])
        time_fmt, time_col_label, _ = dc.time_fmt_for_range(stats["t_min"], stats["t_max"])
        exceedance = dc.compute_exceedance(events)
        top_vehicle_events = dc.top_events(events, obj_type="vehicle", n=top_n)

    return render_template_string(
        STATS_TEMPLATE,
        label=label, day_str=day_str, start=start, end=end,
        hist_start=(start or day_str), hist_end=(end or day_str),
        start_time=start_time, end_time=end_time, time_error=time_error,
        stats=stats, start_str=start_str, end_str=end_str,
        direction_totals=direction_totals, type_totals=type_totals,
        directions=dc.DIRECTIONS, types=dc.TYPES,
        percentiles=[p for p in dc.PERCENTILES if p != 50],
        exceedance=exceedance, top_vehicle_events=top_vehicle_events,
        time_fmt=time_fmt, time_col_label=time_col_label,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return redirect(url_for("day_view", day_str=dc.today_str()))


SORT_FIELDS = {
    "time": ("start_epoch", "asc"),
    "speed": ("max_mph", "desc"),
    "duration": ("duration_s", "desc"),
}


def build_sort_headers(sort_key, sort_dir):
    headers = {}
    for key, (_, default_dir) in SORT_FIELDS.items():
        if key == sort_key:
            next_dir = "asc" if sort_dir == "desc" else "desc"
            arrow = " \u25b2" if sort_dir == "asc" else " \u25bc"
        else:
            next_dir = default_dir
            arrow = ""
        headers[key] = {"next_dir": next_dir, "arrow": arrow}
    return headers


@app.route("/days")
def days_list():
    return render_template_string(DAYS_LIST_TEMPLATE, days=dc.list_available_days(), data_dir=dc.DATA_DIR)


@app.route("/day/<day_str>")
def day_view(day_str):
    if not re.fullmatch(r"\d{8}", day_str):
        abort(404)

    sort_key = request.args.get("sort", "time")
    if sort_key not in SORT_FIELDS:
        sort_key = "time"
    sort_dir = request.args.get("dir", SORT_FIELDS[sort_key][1])
    if sort_dir not in ("asc", "desc"):
        sort_dir = SORT_FIELDS[sort_key][1]

    tracks, file_exists = get_day_tracks(day_str)
    tracks = [t for t in tracks if t["duration_s"] >= dc.MIN_TRACK_DURATION_S]

    field, _ = SORT_FIELDS[sort_key]
    display_tracks = sorted(tracks, key=lambda t: t[field], reverse=(sort_dir == "desc"))

    try:
        display_date = datetime.strptime(day_str, "%Y%m%d").strftime("%A, %B %d, %Y")
    except ValueError:
        display_date = day_str

    east_count = sum(1 for t in tracks if t["direction"] == "east")
    west_count = sum(1 for t in tracks if t["direction"] == "west")
    max_mph = max((t["max_mph"] for t in tracks), default=0.0)

    return render_template_string(
        DAY_TEMPLATE,
        day_str=day_str,
        display_date=display_date,
        tracks=display_tracks,
        headers=build_sort_headers(sort_key, sort_dir),
        file_exists=file_exists,
        is_today=(day_str == dc.today_str()),
        prev_day=dc.adjacent_day(day_str, -1),
        next_day=dc.adjacent_day(day_str, 1),
        local_time_str=dc.local_time_str,
        east_count=east_count,
        west_count=west_count,
        max_mph=max_mph,
    )


def _parse_top_n():
    try:
        n = int(request.args.get("top", "20"))
        return max(1, min(n, 200))
    except (TypeError, ValueError):
        return 20


@app.route("/stats/<day_str>")
def stats_day(day_str):
    if not re.fullmatch(r"\d{8}", day_str):
        abort(404)
    start_time = request.args.get("start_time", "").strip() or None
    end_time = request.args.get("end_time", "").strip() or None
    time_error = None
    try:
        if start_time:
            dc.parse_time_of_day(start_time)
        if end_time:
            dc.parse_time_of_day(end_time)
    except ValueError as e:
        time_error = str(e)
        start_time = end_time = None

    conn = db.get_connection()
    events = db.query_events(conn, date_from=day_str, date_to=day_str,
                              time_from=start_time, time_to=end_time)
    conn.close()
    return render_stats_page(events, label=day_str, day_str=day_str,
                              start_time=start_time, end_time=end_time, time_error=time_error,
                              top_n=_parse_top_n())


@app.route("/stats_range")
def stats_range():
    start = request.args.get("start", "").strip()
    end = request.args.get("end", "").strip()
    start_time = request.args.get("start_time", "").strip() or None
    end_time = request.args.get("end_time", "").strip() or None
    time_error = None
    try:
        if start_time:
            dc.parse_time_of_day(start_time)
        if end_time:
            dc.parse_time_of_day(end_time)
    except ValueError as e:
        time_error = str(e)
        start_time = end_time = None

    events = []
    label = "custom range"
    if re.fullmatch(r"\d{8}", start) and re.fullmatch(r"\d{8}", end):
        conn = db.get_connection()
        events = db.query_events(conn, date_from=start, date_to=end,
                                  time_from=start_time, time_to=end_time)
        conn.close()
        label = f"{start} to {end}"
    return render_stats_page(events, label=label, start=start, end=end,
                              start_time=start_time, end_time=end_time, time_error=time_error,
                              top_n=_parse_top_n())


def _parse_float_arg(name):
    raw = request.args.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def gaussian_pdf(x, mean, std):
    return (1.0 / (std * (2 * math.pi) ** 0.5)) * math.exp(-0.5 * ((x - mean) / std) ** 2)


def fit_gaussian(speeds, fit_min, fit_max):
    """Fit mean/std from only the samples within [fit_min, fit_max] (or all
    of them, if both are None). Returns None if there's not enough data to
    fit. Shared by the histogram plot and the Na/Nb stats endpoint so they
    can never disagree with each other."""
    fit_speeds = speeds
    if fit_min is not None or fit_max is not None:
        lo_f = fit_min if fit_min is not None else -1e9
        hi_f = fit_max if fit_max is not None else 1e9
        fit_speeds = [s for s in speeds if lo_f <= s <= hi_f]
    if len(fit_speeds) < 2:
        return None
    n_fit = len(fit_speeds)
    mean = sum(fit_speeds) / n_fit
    var = sum((s - mean) ** 2 for s in fit_speeds) / (n_fit - 1)
    std = var ** 0.5
    if std <= 0:
        return None
    return {"mean": mean, "std": std, "n_fit": n_fit}


def equal_width_bins(speeds, n_bins=30):
    """Same binning matplotlib's ax.hist(speeds, bins=n_bins) uses (equal-
    width bins spanning the data's own min/max), reimplemented without a
    numpy/matplotlib dependency so the Na/Nb JSON endpoint can compute the
    identical bins the plot draws without rendering a figure."""
    lo, hi = min(speeds), max(speeds)
    if lo == hi:
        hi = lo + 1.0
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for s in speeds:
        idx = int((s - lo) / width)
        if idx >= n_bins:
            idx = n_bins - 1
        elif idx < 0:
            idx = 0
        counts[idx] += 1
    edges = [lo + i * width for i in range(n_bins + 1)]
    return counts, edges


def compute_na_nb(speeds, fit_min, fit_max, n_bins=30):
    """Split the total event count into Na (consistent with the Gaussian
    fit) and Nb (excess above the fit on the high/right side only): every
    event beyond fit_max counts fully toward Nb; for bins between the
    fitted mean and fit_max, only the portion of that bin's bar ABOVE the
    fitted curve counts (never negative). If fit_max wasn't given, every
    bin right of the mean uses the excess-only rule out to the data's own
    max. Uses the same bins the histogram plot draws, so this always
    matches what's visually shown."""
    n_total = len(speeds)
    if n_total == 0:
        return None
    fit = fit_gaussian(speeds, fit_min, fit_max)
    if fit is None:
        return None
    mean, std, n_fit = fit["mean"], fit["std"], fit["n_fit"]

    counts, bin_edges = equal_width_bins(speeds, n_bins)
    bin_width = bin_edges[1] - bin_edges[0]

    nb = 0.0
    for i, observed in enumerate(counts):
        center = (bin_edges[i] + bin_edges[i + 1]) / 2.0
        if center <= mean:
            continue
        if fit_max is not None and center > fit_max:
            nb += observed  # fully outside the fitted range -- counts in full
        else:
            predicted = n_fit * bin_width * gaussian_pdf(center, mean, std)
            nb += max(0.0, observed - predicted)

    nb = int(round(nb))
    na = n_total - nb
    return {"n_total": n_total, "na": na, "nb": nb, "mean": mean, "std": std, "n_fit": n_fit}


@app.route("/histogram_stats.json")
def histogram_stats():
    start = request.args.get("start", "").strip()
    end = request.args.get("end", "").strip()
    start_time = request.args.get("start_time", "").strip() or None
    end_time = request.args.get("end_time", "").strip() or None
    obj_type = request.args.get("type", "vehicle")
    if obj_type not in dc.TYPES:
        obj_type = "vehicle"
    fit_min = _parse_float_arg("fit_min")
    fit_max = _parse_float_arg("fit_max")

    if not (re.fullmatch(r"\d{8}", start) and re.fullmatch(r"\d{8}", end)):
        abort(404)

    conn = db.get_connection()
    events = db.query_events(conn, date_from=start, date_to=end,
                              time_from=start_time, time_to=end_time)
    conn.close()

    speeds = [e["max_mph"] for e in events if e["type"] == obj_type]
    result = compute_na_nb(speeds, fit_min, fit_max)
    if result is None:
        return jsonify({"error": "not enough data to fit"})
    return jsonify(result)


@app.route("/histogram.png")
def speed_histogram():
    start = request.args.get("start", "").strip()
    end = request.args.get("end", "").strip()
    start_time = request.args.get("start_time", "").strip() or None
    end_time = request.args.get("end_time", "").strip() or None
    obj_type = request.args.get("type", "vehicle")
    if obj_type not in dc.TYPES:
        obj_type = "vehicle"
    log_scale = request.args.get("log", "0") == "1"
    show_gaussian = request.args.get("gaussian", "0") == "1"
    fit_min = _parse_float_arg("fit_min")
    fit_max = _parse_float_arg("fit_max")

    if not (re.fullmatch(r"\d{8}", start) and re.fullmatch(r"\d{8}", end)):
        abort(404)

    conn = db.get_connection()
    events = db.query_events(conn, date_from=start, date_to=end,
                              time_from=start_time, time_to=end_time)
    conn.close()

    speeds = [e["max_mph"] for e in events if e["type"] == obj_type]

    title = f"{obj_type} speed distribution -- {start}"
    if end != start:
        title += f" to {end}"
    if start_time or end_time:
        title += f"  [{start_time or '00:00'}-{end_time or '23:59'}]"
    title += f"  (n={len(speeds)})"

    fig, ax = plt.subplots(figsize=(7, 3.5))
    if speeds:
        counts, bin_edges, _ = ax.hist(speeds, bins=30, color="#1d4ed8", edgecolor="white",
                                        label="observed", zorder=2)

        if show_gaussian and len(speeds) > 1:
            # Fit mean/std/amplitude from only the samples within
            # [fit_min, fit_max] if given (letting you test e.g. whether
            # the CORE of the distribution is narrower than a single fit
            # to everything) -- but still draw the resulting curve across
            # the full x-range, so it's visible where it diverges from the
            # tails outside the fit window.
            fit = fit_gaussian(speeds, fit_min, fit_max)
            if fit is not None:
                mean, std, n_fit = fit["mean"], fit["std"], fit["n_fit"]
                bin_width = bin_edges[1] - bin_edges[0]
                lo, hi = bin_edges[0], bin_edges[-1]
                pad = (hi - lo) * 0.15
                xs = [lo - pad + i * (hi - lo + 2 * pad) / 300 for i in range(301)]
                ys = [n_fit * bin_width * gaussian_pdf(x, mean, std) for x in xs]
                fit_label = "Gaussian fit"
                if fit_min is not None or fit_max is not None:
                    fit_label += f" [{fit_min if fit_min is not None else 'min'}-{fit_max if fit_max is not None else 'max'} mph]"
                fit_label += f" (\u03bc={mean:.1f}, \u03c3={std:.1f}, n={n_fit})"
                ax.plot(xs, ys, linestyle=":", color="#111", linewidth=1.8,
                        label=fit_label, zorder=3)

        if log_scale:
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.5)  # so zero-count bins don't collapse the axis

        ax.legend(fontsize=8, loc="upper right")

    ax.set_xlabel("Max speed (mph)")
    ax.set_ylabel("Count" + (" (log scale)" if log_scale else ""))
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/plot/<day_str>/<int:track_idx>.png")
def track_plot(day_str, track_idx):
    df = dc.load_day_df(day_str)
    tracks = dc.segment_tracks(df)
    if track_idx < 0 or track_idx >= len(tracks):
        abort(404)
    t = tracks[track_idx]

    epochs = [e for e, _, _ in t["samples"]]
    raw_kmh = [v for _, v, _ in t["samples"]]
    artifact_mask = [a for _, _, a in t["samples"]]
    raw_mph = [abs(v) / dc.KMH_PER_MPH for v in raw_kmh]

    clean_kmh = [v for v, bad in zip(raw_kmh, artifact_mask) if not bad]
    clean_epochs = [e for e, bad in zip(epochs, artifact_mask) if not bad]
    if clean_kmh:
        med_kmh = dc.median_filter(clean_kmh)
        smooth_kmh = dc.rts_smooth_velocity(clean_epochs, med_kmh)
        smooth_epochs = clean_epochs
    else:
        med_kmh = dc.median_filter(raw_kmh)
        smooth_kmh = dc.rts_smooth_velocity(epochs, med_kmh)
        smooth_epochs = epochs
    smooth_mph = [abs(v) / dc.KMH_PER_MPH for v in smooth_kmh]

    t0 = epochs[0]
    rel_t = [e - t0 for e in epochs]
    smooth_rel_t = [e - t0 for e in smooth_epochs]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    normal_rel_t = [rt for rt, bad in zip(rel_t, artifact_mask) if not bad]
    normal_mph = [m for m, bad in zip(raw_mph, artifact_mask) if not bad]
    ax.scatter(normal_rel_t, normal_mph, s=8, color="#bbb", label="raw", zorder=1)
    if any(artifact_mask):
        bad_rel_t = [rt for rt, bad in zip(rel_t, artifact_mask) if bad]
        bad_mph = [m for m, bad in zip(raw_mph, artifact_mask) if bad]
        ax.scatter(bad_rel_t, bad_mph, s=28, color="#e11d48", marker="x",
                   label="flagged artifact", zorder=2)
    ax.plot(smooth_rel_t, smooth_mph, color="#dc2626", linewidth=1.3, label="RTS smoothed", zorder=3)
    ax.set_xlabel("Time (s, relative to track start)")
    ax.set_ylabel("Speed (mph)")
    ax.set_title(f"{t['direction']} @ {dc.local_time_str(t0)} PDT  --  max {t['max_mph']:.1f} mph (RTS smoothed)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
