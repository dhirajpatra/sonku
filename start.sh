#!/usr/bin/env bash
# Start (or restart) Sonku and follow its log. Ctrl+C leaves the service running.
set -e
sudo systemctl restart sonku.service
sleep 2
if systemctl is-active --quiet sonku.service; then echo "Sonku is running."; else echo "Sonku failed to start:"; journalctl -u sonku.service -n 30 --no-pager; exit 1; fi
journalctl -u sonku.service -f -n 40
