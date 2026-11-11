source ~/robot_env/bin/activate

# 1. Check you really have the new code (all three should exist)
ls app/main.py systemd/sonku.service scripts/install_service.sh
grep -c sonku.service start.sh     # should print 1 or more; 0 means start.sh is still the old one

# 2. Stop the old service so it doesn't hold the mic and camera
sudo systemctl disable --now tabletop-robot.service

# 3. Config and dependencies
[ -f .env ] || cp .env.example .env
nano .env                          # set ADMIN_PASSWORD and MODEL_API_KEY
~/robot_env/bin/pip install -r requirements.txt
~/robot_env/bin/python scripts/download_models.py

# 4. Run it in the foreground first, to see any errors directly
~/robot_env/bin/python -m app      # Ctrl+C to stop