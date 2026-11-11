#!/bin/bash
set -e

echo "[1/4] Restarting Sonku Service..."

# Ensure we use an absolute path to the virtualenv if needed
source /home/dhirajpatra/robot_env/bin/activate

# Restart the service
sudo systemctl restart tabletop-robot.service

# Verify the service started cleanly (non-blocking status check)
if systemctl is-active --quiet tabletop-robot.service; then
    echo "[✓] Sonku service restarted successfully!"
else
    echo "[X] Service failed to start. Printing recent error logs:"
    journalctl -u tabletop-robot.service -n 20 --no-pager
    exit 1
fi

echo "[2/4] Streaming live journal (Press Ctrl+C to exit log view)..."
# Stream live logs as the final step
journalctl -u tabletop-robot.service -f -n 50