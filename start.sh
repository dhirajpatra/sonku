#!/usr/bin/env bash
# Start (or restart) Sonku and follow its log. Ctrl+C leaves the service running.
set -e

source /home/dhirajpatra/robot_env/bin/activate
sudo systemctl restart tabletop-robot.service
sleep 2
if systemctl is-active --quiet tabletop-robot.service; then echo "Tabletop-Robot is running."; else echo "Tabletop-Robot failed to start:"; journalctl -u tabletop-robot.service -n 30 --no-pager; exit 1; fi
journalctl -u tabletop-robot.service -f -n 40
