#!/usr/bin/env bash
# Start (or restart) Sonku and follow its log. Ctrl+C leaves the service running.
set -e

source /home/dhirajpatra/robot_env/bin/activate
sudo systemctl restart tabletop-rbot.service
sleep 2
if systemctl is-active --quiet tabletop-rbot.service; then echo "Tabletop-Rbot is running."; else echo "Tabletop-Rbot failed to start:"; journalctl -u tabletop-rbot.service -n 30 --no-pager; exit 1; fi
journalctl -u tabletop-rbot.service -f -n 40
