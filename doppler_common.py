#!/usr/bin/env python3
"""
doppler_common.py -- shared core logic for DOPPLER1 radar speed data.

Everything about turning raw (epoch, kmh) readings into per-vehicle
"tracks" (segmentation, outlier filtering, smoothing, artifact detection,
type/direction classification) lives here, so both the live web app
(doppler_web.py) and the database builder (doppler_db_build.py) use
identical logic and can't drift apart.

The radar goes fully silent (no output at all) both at a genuine end of
track AND during a brief mid-track signal dropout -- the only way to tell
them apart is how long the silence lasts. So tracks are segmented purely by
time gap between consecutive VALID (nonzero) readings: a gap longer than
TRACK_GAP_THRESHOLD_S starts a new track; shorter gaps (a momentary dropout)
stay part of the same track. Zero rows themselves carry no usable speed
info and are simply dropped before segmenting.
"""

import glob
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "/mnt/bluecherry/DOPPLER1"
FILE_RE = re.compile(r"doppler1_speed_daily_(\d{8})\.csv$")

LOCAL_TZ = ZoneInfo("America/Los_Angeles")

# Consecutive valid (nonzero) readings more than this many seconds apart are
# treated as two separate vehicle tracks, not one track with a brief signal
# dropout in the middle. Tune this if tracks look wrongly split or wrongly
# merged once you have more real data to check it against.
TRACK_GAP_THRESHOLD_S = 2.0

# Sign convention for this radar's mounting/orientation. Confirmed from one
# cross-checked event (2026-08-19 10:33:48 PDT, independently verified
# eastbound by video) that showed NEGATIVE readings -- only one data point
# so far, flip this if it turns out backwards once you've checked more.
SIGN_LABELS = {-1: "east", 1: "west"}

KMH_PER_MPH = 1.60934
KMH_TO_MPS = 1.0 / 3.6
FT_PER_MPH_S = 5280.0 / 3600.0  # 1 mph = this many feet/second

# Speed below which a track is classified "pedestrian" rather than
# "vehicle" -- same convention used for the LIDAR station matching (an
# arbitrary but reasonable rule of thumb, not derived from this data).
PEDESTRIAN_MPH_CUTOFF = 8.0

# Rolling median filter applied to every track's raw signed-kmh series before
# anything else. Robust (order-statistic based) rejection of brief outlier
# spikes -- e.g. a pedestrian's swinging arm/leg momentarily returning a
# stronger, faster-moving reflection than the torso -- without needing any
# noise-model tuning. Window is in SAMPLES, not seconds: since sample rate
# can itself be bursty/irregular for weak-signal targets (pedestrians tend
# to have far sparser, less regular timing than vehicles), a sample-count
# window can span more or less real time depending on how bursty that
# particular track is.
MEDIAN_FILTER_WINDOW = 9

# Tracks shorter than this are hidden from display entirely (too brief to
# be a meaningful reading -- noise/edge-of-range blips). Filtering happens
# at display/query time, not in segment_tracks() itself, so the full
# computed set is always available if needed.
MIN_TRACK_DURATION_S = 0.6

# Speed readings above this are treated as known-bad -- e.g. a period of RF
# interference producing spurious very-high readings -- and excluded from
# all queries by default. Not a real vehicle speed under any plausible
# circumstance on a residential street. Filtered at query time (not at
# storage/build time), so the raw rows stay recoverable in the database if
# this threshold ever needs revisiting.
MAX_PLAUSIBLE_SPEED_MPH = 75.0

# Artifact detection: on rare occasions the raw data itself looks physically
# impossible -- e.g. several consecutive samples reading the EXACT same
# value (real Doppler noise essentially never repeats identically that many
# times), paired with a discontinuous jump to/from the rest of the track
# that would require an implausible deceleration for a real vehicle. Points
# matching both signatures are excluded from smoothing/max-speed
# calculations (but still available, marked, in the samples list).
MIN_STUCK_RUN = 4                    # consecutive identical raw readings this long is suspicious
MAX_PLAUSIBLE_ACCEL_MPS2 = 10.0      # ~1g -- a transition into/out of a suspicious run implying
                                      # more than this is treated as an implausible discontinuity
ARTIFACT_MAX_SPEED_IMPACT_MPH = 2.0  # only visibly flag a track (vs. just quietly excluding the
                                      # points) if doing so actually changed the reported max
                                      # speed by more than this

# General coherence check, applied across EVERY consecutive raw-sample pair
# in a candidate track (unlike detect_artifact_mask() above, which only
# checks this same MAX_PLAUSIBLE_ACCEL_MPS2 bound at the edges of a
# stuck-value run). A single vehicle or pedestrian accelerates/decelerates
# smoothly; readings that repeatedly swing wildly (e.g. 3 -> 36 -> 3 -> 74
# mph within a couple of samples) aren't a coherent single target at all --
# more likely electronic interference or another unreliable source. A FEW
# implausible jumps are normal and expected, especially right at a track's
# start/end where signal is weakest (a long good track will dilute a
# couple of noisy edge samples to a low percentage); what distinguishes
# real noise is a HIGH PROPORTION of the whole track being incoherent, not
# a fixed count. Checked against two real all-noise examples (29% and 52%
# implausible pairs) vs. a realistic good track with ordinary edge noise
# (~4%) -- there's a wide, clean gap between those, which is what these
# two constants are chosen to sit inside.
MAX_IMPLAUSIBLE_JUMP_FRACTION = 0.20   # reject if implausible jumps exceed this share of pairs
MIN_IMPLAUSIBLE_JUMPS_TO_REJECT = 3    # ...but never reject on fewer than this many in absolute
                                        # terms, so a short track with just 1-2 bad pairs (which
                                        # could easily exceed the fraction above on a small
                                        # denominator) isn't wrongly discarded

# Overlap detection: two vehicles travelling in OPPOSITE directions that
# pass close enough together in time can land in the same time-gap group
# above (segmentation only looks at time gaps, not sign) -- e.g. a
# westbound car passing just as an eastbound car is still in range. A
# real single vehicle's raw signed reading should not flip sign under
# this sensor's road-aligned geometry, so a SUSTAINED sign reversal
# within one group (not just a sample or two of noise near a magnitude
# dip at closest approach, which is ordinary) is treated as two
# overlapping tracks and split apart at the crossing. Tune this if real
# single-vehicle passes start getting incorrectly split, or genuine
# overlaps go uncaught.
MIN_DIRECTION_RUN_SAMPLES = 6

# Kalman/RTS smoother: nearly-constant-acceleration model (state = [velocity,
# acceleration], process noise on jerk), applied on top of the median-
# filtered series. See doppler_web.py's history/tuning notes for why these
# specific values were chosen -- they came from checking against several
# real tracks (a validated deceleration event, a cosine-angle antenna
# artifact, a sensor-glitch case), not just picked blind.
KALMAN_JERK_NOISE = 20.0
KALMAN_MEASUREMENT_NOISE = 1.0


# ---------------------------------------------------------------------------
# Filtering / smoothing
# ---------------------------------------------------------------------------
def _implied_accel_mps2(v1_kmh, v2_kmh, dt):
    dv_mps = abs(v2_kmh - v1_kmh) * KMH_TO_MPS
    return dv_mps / max(dt, 1e-3)


def detect_artifact_mask(epochs, raw_kmh):
    """Flag samples that are part of a run of identical consecutive raw
    values AND border an implausible speed discontinuity -- the signature
    of a sensor artifact (e.g. a spurious lock before/after the real
    target), not real vehicle motion. Returns a bool list, same length as
    the input, True where a sample should be excluded from smoothing/stats."""
    n = len(raw_kmh)
    mask = [False] * n
    if n == 0:
        return mask

    i = 0
    while i < n:
        j = i
        while j + 1 < n and raw_kmh[j + 1] == raw_kmh[i]:
            j += 1
        run_len = j - i + 1

        if run_len >= MIN_STUCK_RUN:
            implausible = False
            if i > 0 and _implied_accel_mps2(raw_kmh[i - 1], raw_kmh[i], epochs[i] - epochs[i - 1]) > MAX_PLAUSIBLE_ACCEL_MPS2:
                implausible = True
            if j + 1 < n and _implied_accel_mps2(raw_kmh[j], raw_kmh[j + 1], epochs[j + 1] - epochs[j]) > MAX_PLAUSIBLE_ACCEL_MPS2:
                implausible = True
            if implausible:
                for k in range(i, j + 1):
                    mask[k] = True

        i = j + 1
    return mask


def count_implausible_jumps(epochs, raw_kmh):
    """Number of consecutive raw-sample pairs whose implied acceleration
    exceeds MAX_PLAUSIBLE_ACCEL_MPS2, checked across the WHOLE track (not
    just at stuck-run boundaries like detect_artifact_mask() above)."""
    n = len(raw_kmh)
    count = 0
    for i in range(1, n):
        if _implied_accel_mps2(raw_kmh[i - 1], raw_kmh[i], epochs[i] - epochs[i - 1]) > MAX_PLAUSIBLE_ACCEL_MPS2:
            count += 1
    return count


def median_filter(values, window=MEDIAN_FILTER_WINDOW):
    return pd.Series(values).rolling(window, center=True, min_periods=1).median().tolist()


def split_direction_overlaps(g):
    """Given one time-contiguous group of raw readings (already segmented
    by time gap), detect a SUSTAINED sign reversal in the raw kmh values
    -- the signature of two vehicles travelling in opposite directions
    whose tracks overlapped closely enough in time that gap-based
    segmentation alone couldn't separate them (see
    MIN_DIRECTION_RUN_SAMPLES). A brief flip of just a sample or two right
    at a magnitude dip is ordinary near-zero noise, not a second vehicle,
    so both sides of a split must have a solid run of consistent sign
    before it's treated as a genuine overlap.

    Any brief, weak samples sitting between two solid opposite-sign runs
    are DROPPED entirely rather than assigned to either side: they're
    exactly where the two vehicles' signals overlap/cancel, so they
    aren't reliably either vehicle's true reading, and including them
    would drag each half's smoothed curve toward a spurious dip/spike at
    its own edge.

    Returns a list of (sub_dataframe, was_split) pairs: [(g, False)] if no
    genuine reversal is found (the overwhelmingly common case), or two or
    more entries with was_split=True if the group was split."""
    kmh = g["kmh"].tolist()
    n = len(kmh)
    if n < 2 * MIN_DIRECTION_RUN_SAMPLES:
        return [(g, False)]

    signs = [1 if v > 0 else -1 for v in kmh]
    runs = []  # [sign, start_idx, end_idx_inclusive]
    start = 0
    for i in range(1, n + 1):
        if i == n or signs[i] != signs[start]:
            runs.append([signs[start], start, i - 1])
            start = i

    solid = [r for r in runs if (r[2] - r[1] + 1) >= MIN_DIRECTION_RUN_SAMPLES]
    if len({r[0] for r in solid}) < 2:
        return [(g, False)]  # everything's one sign, or only brief noise flips

    segments = []  # [(start_idx, end_idx_inclusive), ...]
    seg_start = 0
    current_sign = solid[0][0]
    for i in range(1, len(solid)):
        if solid[i][0] != current_sign:
            segments.append((seg_start, solid[i - 1][2]))  # end at prior solid run's own extent
            seg_start = solid[i][1]                         # resume at this solid run's start
            current_sign = solid[i][0]
    segments.append((seg_start, n - 1))

    return [(g.iloc[s:e + 1], True) for s, e in segments]


def kalman_filter_velocity(epochs, values, q=KALMAN_JERK_NOISE, r=KALMAN_MEASUREMENT_NOISE):
    """Causal (forward-only) nearly-constant-acceleration Kalman filter.
    Kept for reference/potential future real-time use; the web app and DB
    builder both use rts_smooth_velocity() instead, since a full track is
    always available before it's ever processed."""
    n = len(values)
    if n == 0:
        return []
    v, a = values[0], 0.0
    p_vv, p_va, p_aa = float(r), 0.0, 1.0
    out = [v]
    for i in range(1, n):
        dt = max(epochs[i] - epochs[i - 1], 1e-3)
        v_pred = v + a * dt
        a_pred = a
        q_vv = q * dt ** 3 / 3
        q_va = q * dt ** 2 / 2
        q_aa = q * dt
        p_vv_pred = p_vv + 2 * dt * p_va + dt * dt * p_aa + q_vv
        p_va_pred = p_va + dt * p_aa + q_va
        p_aa_pred = p_aa + q_aa

        z = values[i]
        y = z - v_pred
        s = p_vv_pred + r
        k_v = p_vv_pred / s
        k_a = p_va_pred / s

        v = v_pred + k_v * y
        a = a_pred + k_a * y
        p_vv = (1 - k_v) * p_vv_pred
        p_va = (1 - k_v) * p_va_pred
        p_aa = p_aa_pred - k_a * p_va_pred

        out.append(v)
    return out


def rts_smooth_velocity(epochs, values, q=KALMAN_JERK_NOISE, r=KALMAN_MEASUREMENT_NOISE):
    """Rauch-Tung-Striebel smoother: the non-causal counterpart to
    kalman_filter_velocity. Same physical model (nearly-constant
    acceleration) and same forward pass, but followed by a backward pass
    that revisits every point using everything that happened AFTER it too
    -- the maximum-likelihood state estimate given the WHOLE track, not
    just what could be inferred causally up to that instant."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [float(values[0])]

    H = np.array([[1.0, 0.0]])
    Rm = np.array([[r]])
    I2 = np.eye(2)

    x_filt = [None] * n
    P_filt = [None] * n
    x_pred = [None] * n
    P_pred = [None] * n
    Fs = [None] * n

    x = np.array([values[0], 0.0])
    P = np.array([[float(r), 0.0], [0.0, 1.0]])
    x_filt[0] = x
    P_filt[0] = P

    for k in range(1, n):
        dt = max(epochs[k] - epochs[k - 1], 1e-3)
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        Fs[k] = F

        xp = F @ x
        Pp = F @ P @ F.T + Q
        x_pred[k] = xp
        P_pred[k] = Pp

        z = np.array([values[k]])
        y = z - H @ xp
        S = H @ Pp @ H.T + Rm
        K = Pp @ H.T @ np.linalg.inv(S)
        x = xp + K @ y
        P = (I2 - K @ H) @ Pp

        x_filt[k] = x
        P_filt[k] = P

    x_smooth = [None] * n
    P_smooth = [None] * n
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for k in range(n - 2, -1, -1):
        F = Fs[k + 1]
        C = P_filt[k] @ F.T @ np.linalg.inv(P_pred[k + 1])
        x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + C @ (P_smooth[k + 1] - P_pred[k + 1]) @ C.T

    return [float(xs[0]) for xs in x_smooth]


# ---------------------------------------------------------------------------
# Data loading / segmentation
# ---------------------------------------------------------------------------
def list_available_days():
    days = []
    for path in glob.glob(os.path.join(DATA_DIR, "doppler1_speed_daily_*.csv")):
        m = FILE_RE.search(path)
        if m:
            days.append(m.group(1))
    return sorted(days, reverse=True)


def day_csv_path(day_str):
    return os.path.join(DATA_DIR, f"doppler1_speed_daily_{day_str}.csv")


def load_day_df(day_str):
    path = day_csv_path(day_str)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["kmh"] = df["kmh"].astype(float)
    return df


def segment_tracks(df):
    """Split a day's raw readings into per-vehicle tracks. See module
    docstring for the gap-based segmentation rationale. Returns a list of
    dicts, chronologically ordered, each with a stable 'idx' field."""
    if df is None or df.empty:
        return []
    valid = df[df["kmh"] != 0].sort_values("epoch").reset_index(drop=True)
    if valid.empty:
        return []

    gap = valid["epoch"].diff()
    new_track = gap.isna() | (gap > TRACK_GAP_THRESHOLD_S)
    group_id = new_track.cumsum()

    tracks = []
    for _, g in valid.groupby(group_id):
        for sub_g, was_split in split_direction_overlaps(g):
            raw_kmh = sub_g["kmh"].tolist()
            epochs_list = sub_g["epoch"].tolist()

            n_bad_jumps = count_implausible_jumps(epochs_list, raw_kmh)
            n_pairs = len(raw_kmh) - 1
            if n_pairs > 0 and n_bad_jumps >= MIN_IMPLAUSIBLE_JUMPS_TO_REJECT \
                    and (n_bad_jumps / n_pairs) > MAX_IMPLAUSIBLE_JUMP_FRACTION:
                continue  # too erratic to be one coherent target - discard the whole segment

            artifact_mask = detect_artifact_mask(epochs_list, raw_kmh)
            clean_kmh = [v for v, bad in zip(raw_kmh, artifact_mask) if not bad]
            clean_epochs = [e for e, bad in zip(epochs_list, artifact_mask) if not bad]

            raw_max_abs = max(abs(v) for v in raw_kmh)
            if clean_kmh:
                med_kmh = median_filter(clean_kmh)
                smooth_kmh = rts_smooth_velocity(clean_epochs, med_kmh)
            else:
                med_kmh = median_filter(raw_kmh)
                smooth_kmh = rts_smooth_velocity(epochs_list, med_kmh)
            max_abs = max(abs(v) for v in smooth_kmh)
            max_mph = max_abs / KMH_PER_MPH
            # Median of the same final (smoothed, artifact-excluded) speed
            # series max_mph is drawn from, so the two are directly
            # comparable - not computed from the raw or intermediate signal.
            median_abs = float(np.median([abs(v) for v in smooth_kmh]))
            median_mph = median_abs / KMH_PER_MPH
            duration_s = float(sub_g["epoch"].max() - sub_g["epoch"].min())
            # Crude distance estimate: treats the whole track as if it moved
            # at its median speed for its full duration. Ignores acceleration
            # within the track and any time before/after the radar's first/
            # last valid sample, so it's a rough figure, not a precise one.
            track_distance_ft = median_mph * FT_PER_MPH_S * duration_s

            artifact_flag = any(artifact_mask) and (raw_max_abs - max_abs) > (ARTIFACT_MAX_SPEED_IMPACT_MPH * KMH_PER_MPH)

            sign_source = clean_kmh if clean_kmh else raw_kmh
            n_pos = sum(1 for v in sign_source if v > 0)
            sign = 1 if n_pos >= len(sign_source) / 2 else -1

            obj_type = "pedestrian" if max_mph < PEDESTRIAN_MPH_CUTOFF else "vehicle"

            tracks.append({
                "start_epoch": float(sub_g["epoch"].min()),
                "end_epoch": float(sub_g["epoch"].max()),
                "duration_s": duration_s,
                "n_samples": int(len(sub_g)),
                "max_kmh": float(max_abs),
                "max_mph": float(max_mph),
                "median_mph": float(median_mph),
                "track_distance_ft": float(track_distance_ft),
                "direction": SIGN_LABELS.get(sign, "?"),
                "type": obj_type,
                "artifact_flag": artifact_flag,
                "overlap_split": was_split,
                "samples": list(zip(epochs_list, raw_kmh, artifact_mask)),
            })
    tracks.sort(key=lambda t: t["start_epoch"])
    for i, t in enumerate(tracks):
        t["idx"] = i  # stable chronological index -- used for /plot/ URLs regardless
                       # of how a caller sorts its own copy for display
    return tracks


def local_time_str(epoch):
    return datetime.fromtimestamp(epoch, tz=LOCAL_TZ).strftime("%H:%M:%S")


def local_time_str_ms(epoch):
    return datetime.fromtimestamp(epoch, tz=LOCAL_TZ).strftime("%H:%M:%S.%f")[:-3]


def local_datetime_str_ms(epoch):
    return datetime.fromtimestamp(epoch, tz=LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def today_str():
    return datetime.now(tz=LOCAL_TZ).strftime("%Y%m%d")


def adjacent_day(day_str, delta_days):
    d = datetime.strptime(day_str, "%Y%m%d") + timedelta(days=delta_days)
    return d.strftime("%Y%m%d")


def parse_time_of_day(s):
    """Parse 'HH:MM' or 'HH:MM:SS' (optionally '.fff') into seconds since
    local midnight. Raises ValueError on bad input."""
    parts = s.strip().split(":")
    if len(parts) == 2:
        h, m = parts
        sec = 0.0
    elif len(parts) == 3:
        h, m, sec = parts
        sec = float(sec)
    else:
        raise ValueError(f"invalid time-of-day: {s!r} (expected HH:MM or HH:MM:SS)")
    h, m = int(h), int(m)
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= sec < 60):
        raise ValueError(f"invalid time-of-day: {s!r} (expected HH:MM or HH:MM:SS)")
    return h * 3600 + m * 60 + sec


def time_of_day_seconds(epoch):
    """Seconds since local midnight for the given epoch (fractional)."""
    t = datetime.fromtimestamp(epoch, tz=LOCAL_TZ)
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def time_fmt_for_range(t_min, t_max):
    """Pick the right formatter/label/column-width for displaying event
    timestamps, based on whether [t_min, t_max] spans more than one
    calendar day -- a bare time-of-day is ambiguous over a multi-day range."""
    multi_day = local_datetime_str_ms(t_min)[:10] != local_datetime_str_ms(t_max)[:10]
    if multi_day:
        return local_datetime_str_ms, "PDT date/time", 26
    return local_time_str_ms, "PDT time", 16


# ---------------------------------------------------------------------------
# Aggregate stats (used by both the CLI report and the web /stats pages)
# ---------------------------------------------------------------------------
DIRECTIONS = ["east", "west"]
TYPES = ["vehicle", "pedestrian"]
PERCENTILES = [50, 85, 95, 99]  # 50th = median, already shown separately; kept here
                                 # so PERCENTILE_KEYS below stays in sync with it
PERCENTILE_KEYS = [f"p{p}" for p in PERCENTILES]

# Round-number speed thresholds (mph) for exceedance counts -- "how many/
# what % of events were at or above X mph". Deliberately generic (not tied
# to any one street's posted limit) since this is a shared module; a caller
# that wants "how many over the limit" can pick the right entries out.
EXCEEDANCE_THRESHOLDS_MPH = [25, 30, 35, 40, 45, 50]


# Round-number speed thresholds (mph) for exceedance counts -- "how many/
# what % of events were at or above X mph". Deliberately generic (not tied
# to any one street's posted limit) since this is a shared module; a caller
# that wants "how many over the limit" can pick the right entries out.
EXCEEDANCE_THRESHOLDS_MPH = [25, 30, 35, 40, 45, 50]


def _speed_stats_and_extremes_by_type(events):
    """Given a list of events (already filtered to whatever scope the
    caller wants -- all directions, or just one), compute per-type speed
    stats and slowest/fastest. Shared by the combined and per-direction
    breakdowns so they can't drift apart."""
    speed_stats = {}
    extremes = {}
    for ty in TYPES:
        sub = [e for e in events if e["type"] == ty]
        speeds = pd.Series([e["max_mph"] for e in sub])
        if len(speeds):
            speed_stats[ty] = {
                "n": int(len(speeds)),
                "median": float(speeds.median()),
                "mean": float(speeds.mean()),
                "min": float(speeds.min()),
                "max": float(speeds.max()),
                "std": float(speeds.std()) if len(speeds) > 1 else float("nan"),
                # Fisher (excess) kurtosis: 0 for a Gaussian, positive means
                # more sharply peaked AND heavier-tailed than a Gaussian of
                # the same variance (leptokurtic) -- the combination we saw
                # in the log-scale histogram: a narrower core than the
                # single-Gaussian fit, with heavier tails on both sides.
                "skew": float(speeds.skew()) if len(speeds) > 2 else float("nan"),
                "kurtosis": float(speeds.kurt()) if len(speeds) > 3 else float("nan"),
            }
            for p in PERCENTILES:
                speed_stats[ty][f"p{p}"] = float(speeds.quantile(p / 100.0))
            extremes[ty] = {
                "slowest": min(sub, key=lambda e: e["max_mph"]),
                "fastest": max(sub, key=lambda e: e["max_mph"]),
            }
        else:
            speed_stats[ty] = {"n": 0}
            extremes[ty] = None
    return speed_stats, extremes


def compute_stats(events):
    """events: list of dicts with at least start_epoch, direction, type,
    max_mph (e.g. rows from doppler_db.query_events(), or track dicts from
    segment_tracks()). Returns None if events is empty.

    Includes both combined (speed_stats/extremes) and per-direction
    (speed_stats_by_direction/extremes_by_direction) breakdowns, in the
    same shape -- {type: {...}} for combined, {direction: {type: {...}}}
    for per-direction."""
    if not events:
        return None

    t_min = min(e["start_epoch"] for e in events)
    t_max = max(e["start_epoch"] for e in events)
    period_hours = (t_max - t_min) / 3600.0
    n_days = len({local_datetime_str_ms(e["start_epoch"])[:10] for e in events})

    matrix = {d: {ty: 0 for ty in TYPES} for d in DIRECTIONS}
    for e in events:
        if e["direction"] in matrix and e["type"] in TYPES:
            matrix[e["direction"]][e["type"]] += 1

    speed_stats, extremes = _speed_stats_and_extremes_by_type(events)

    speed_stats_by_direction = {}
    extremes_by_direction = {}
    for d in DIRECTIONS:
        sub = [e for e in events if e["direction"] == d]
        speed_stats_by_direction[d], extremes_by_direction[d] = _speed_stats_and_extremes_by_type(sub)

    return {
        "t_min": t_min,
        "t_max": t_max,
        "period_hours": period_hours,
        "n_days": n_days,
        "matrix": matrix,
        "speed_stats": speed_stats,
        "extremes": extremes,
        "speed_stats_by_direction": speed_stats_by_direction,
        "extremes_by_direction": extremes_by_direction,
        "n_total": len(events),
    }


def compute_exceedance(events, thresholds=EXCEEDANCE_THRESHOLDS_MPH):
    """For each type, how many/what % of events had max_mph >= each of the
    given thresholds. A more concrete, real-world-units way to look at the
    tail than percentiles alone -- e.g. 'how many vehicles were doing 40+?'"""
    result = {}
    for ty in TYPES:
        speeds = [e["max_mph"] for e in events if e["type"] == ty]
        n = len(speeds)
        rows = []
        for t in thresholds:
            count = sum(1 for s in speeds if s >= t)
            rows.append({"mph": t, "count": count, "pct": (count / n * 100.0) if n else 0.0})
        result[ty] = {"n": n, "thresholds": rows}
    return result


def top_events(events, obj_type="vehicle", n=20):
    """The n fastest events of the given type, descending by max_mph. Each
    retains its 'date'/'idx' fields (present on DB-sourced events), so
    callers can link straight to that event's /plot/<date>/<idx>.png."""
    sub = [e for e in events if e["type"] == obj_type]
    return sorted(sub, key=lambda e: e["max_mph"], reverse=True)[:n]
