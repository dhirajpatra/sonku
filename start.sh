#!/usr/bin/env bash
# Start (or restart) Sonku and follow its log. Ctrl+C leaves the service running.
set -e

# Ensure Bluetooth speaker/mic is connected and unmuted (ignore if already connected)
bluetoothctl connect 30:63:71:04:38:BB >/dev/null 2>&1 || true
sleep 1
wpctl set-volume @DEFAULT_SINK@ 1.0 >/dev/null 2>&1 || true
wpctl set-volume @DEFAULT_SOURCE@ 1.0 >/dev/null 2>&1 || true

sudo systemctl restart sonku.service
sleep 2

if systemctl is-active --quiet sonku.service; then
    echo "Sonku is running."
else
    echo "Sonku failed to start:"
    journalctl -u sonku.service -n 30 --no-pager
    exit 1
fi

journalctl -u sonku.service -f -n 40
