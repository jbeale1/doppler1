#!/bin/bash
PROCESS_NAME="/home/pi/DOPPLER1/dlog1.py"
PID_FILE="/tmp/doppler_log.pid"
LOG_FILE="/home/pi/DOPPLER1/dlog1.log"

start_process() {
    echo "Starting process..."
    nohup "$PROCESS_NAME" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Process started with PID $(cat $PID_FILE), logging to $LOG_FILE"
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
status_process() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Running with PID $PID"
        else
            echo "PID file present ($PID) but process is not running - it likely crashed or was killed outside this script."
        fi
    else
        echo "Not running (no PID file)."
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
    status)
        status_process
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        ;;
esac
