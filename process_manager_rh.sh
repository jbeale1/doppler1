#!/bin/bash

PROCESS_NAME="/home/pi/DOPPLER1/readRH.py"
PID_FILE="/tmp/rh_log.pid"

start_process() {
    echo "Starting process..."
    $PROCESS_NAME &
    echo $! > "$PID_FILE"
    echo "Process started with PID $(cat $PID_FILE)"
}

stop_process() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "Stopping process with PID $PID"
        kill "$PID"
        rm "$PID_FILE"
    else
        echo "PID file not found. Process might not be running."
    fi
}

case "$1" in
    start)
        start_process
        ;;
    stop)
        stop_process
        ;;
    restart)
        stop_process
        sleep 2
        start_process
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        ;;
esac
