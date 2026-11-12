#!/usr/bin/env bash
# Installs the systemd unit and retires the old tabletop-robot.service so two programs never fight over the mic/camera.
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/robot_env}"
TARGET=/etc/systemd/system/sonku.service
sed -e "s#@USER@#$USER#g" -e "s#@APP_DIR@#$APP_DIR#g" -e "s#@VENV@#$VENV#g" -e "s#@UID@#$(id -u)#g" "$APP_DIR/systemd/sonku.service" | sudo tee "$TARGET" >/dev/null
if systemctl list-unit-files | grep -q '^tabletop-robot.service'; then
  echo "Disabling old tabletop-robot.service"
  sudo systemctl disable --now tabletop-robot.service || true
fi
sudo systemctl daemon-reload
sudo systemctl enable --now sonku.service
echo "Installed. Admin panel: http://$(hostname -I | awk '{print $1}'):$(grep -E '^WEB_PORT=' "$APP_DIR/.env" | cut -d= -f2- || echo 8000)"
