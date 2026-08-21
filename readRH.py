#!/usr/bin/env python
# 2026-08-19: rewritten to write to tmpfs (/dev/shm) instead of the SD card,
# with daily-rotated CSV files and an hourly push to jbeale-mini via rsync
# -- mirrors the push/rotate pattern from lidar_events3.py (LIDAR3 station):
#   - a background thread rsyncs the still-growing file every hour (plain
#     overwrite, local copy kept) so a tmpfs power-loss loses at most ~1h
#   - at the 00:00 tick the finished day's file is pushed with
#     --remove-source-files and a fresh file is opened for the new day

import board
import adafruit_ahtx0
import time
import os
import threading
import subprocess
import queue

OUTPUT_DIR = '/dev/shm/D1'      # tmpfs directory to log data in (RAM-backed)
REMOTE_DEST = "jbeale@jbeale-mini.local:/mnt/bluecherry/DOPPLER1/"
FILE_PREFIX = 'doppler1_env_daily_'  # -> doppler1_env_daily_YYYYMMDD.csv

VERSION = "1.0.0 (readRH.py: daily-rotated CSV, hourly push)"

os.makedirs(OUTPUT_DIR, exist_ok=True)

_csv_file = None
_csv_path = None
_csv_lock = threading.Lock()

_transfer_queue = queue.Queue()


def _transfer_worker():
    while True:
        fname = _transfer_queue.get()
        try:
            if os.path.exists(fname):
                result = subprocess.run(
                    ['rsync', '-a', '--remove-source-files', fname, REMOTE_DEST],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode != 0:
                    print(f"[transfer] rsync failed for {fname}: {result.stderr.strip()}")
                else:
                    print(f"[transfer] pushed {fname} -> {REMOTE_DEST}")
        except Exception as e:
            print(f"[transfer] error sending {fname}: {e}")
        finally:
            _transfer_queue.task_done()


def _fetch_remote_if_exists(local_fname):
    """At startup (local file doesn't exist yet in tmpfs), check if a copy for
    today already exists on the archive host -- e.g. after a reboot -- and
    pull it down first so we append to the full day's history instead of
    overwriting it with a fresh stub."""
    remote_path = REMOTE_DEST + os.path.basename(local_fname)
    try:
        result = subprocess.run(
            ['rsync', '-a', '--timeout=10', remote_path, local_fname],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and os.path.exists(local_fname):
            print(f"[startup] pulled existing {remote_path} -> {local_fname}")
        else:
            print(f"[startup] no existing remote file pulled ({result.stderr.strip()})")
    except Exception as e:
        print(f"[startup] error checking remote for existing file: {e}")


def _open_env_csv(day_str):
    global _csv_file, _csv_path
    fname = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}{day_str}.csv")
    _csv_file = open(fname, "a")
    if _csv_file.tell() == 0:
        _csv_file.write("epoch,degC,RH\n")
        _csv_file.flush()
    _csv_path = fname
    return fname


def _seconds_until_next_local_hour():
    now = time.time()
    lt = time.localtime(now)
    secs_into_hour = lt.tm_min * 60 + lt.tm_sec + (now - int(now))
    remaining = 3600.0 - secs_into_hour
    return remaining if remaining > 0 else remaining + 3600.0


def _finalize_and_rotate():
    with _csv_lock:
        old_path = None
        if _csv_file is not None:
            old_path = _csv_file.name
            _csv_file.close()
        day_str = time.strftime("%Y%m%d")  # local date, now that midnight has passed
        fname = _open_env_csv(day_str)
        print(f"[daily] midnight rotation -> now logging to {fname}")

    if old_path is not None:
        _transfer_queue.put(old_path)
        print(f"[daily] queued {old_path} for push+remove")


def _hourly_push_loop():
    while True:
        time.sleep(_seconds_until_next_local_hour())

        if time.localtime().tm_hour == 0:
            _finalize_and_rotate()
            continue

        with _csv_lock:
            path = _csv_path
            if path is not None and _csv_file is not None:
                _csv_file.flush()
        if path is None or not os.path.exists(path):
            continue
        try:
            result = subprocess.run(
                ['rsync', '-a', path, REMOTE_DEST],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                print(f"[hourly push] rsync failed for {path}: {result.stderr.strip()}")
            else:
                print(f"[hourly push] pushed {path} -> {REMOTE_DEST} (local copy kept)")
        except Exception as e:
            print(f"[hourly push] error: {e}")


def write_env_row(t_start, t_mean, rh_mean):
    with _csv_lock:
        _csv_file.write("%.1f,%5.4f,%5.4f\n" % (t_start, t_mean, rh_mean))
        _csv_file.flush()


i2c = board.I2C()
sensor = adafruit_ahtx0.AHTx0(i2c)
GROUP = 100
lCount = 0
tSum = 0.0
rSum = 0.0
tStart = 0.0

_startup_day = time.strftime("%Y%m%d")
_startup_fname = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}{_startup_day}.csv")
if not os.path.exists(_startup_fname):
    _fetch_remote_if_exists(_startup_fname)
_open_env_csv(_startup_day)
threading.Thread(target=_transfer_worker, daemon=True, name='transfer').start()
threading.Thread(target=_hourly_push_loop, daemon=True, name='hourly_push_rotate').start()

print(f"readRH.py version {VERSION}")
print(f"Logging to {_csv_path}")

try:
    while True:
        t = sensor.temperature
        r = sensor.relative_humidity
        if lCount == 0:
            tStart = time.time()
        tSum += t
        rSum += r
        lCount += 1
        if lCount >= GROUP:
            write_env_row(tStart, tSum / GROUP, rSum / GROUP)
            lCount = 0
            tSum = 0.0
            rSum = 0.0
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    with _csv_lock:
        if _csv_file:
            _csv_file.close()
