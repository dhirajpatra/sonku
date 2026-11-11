#!/usr/bin/env bash
journalctl -u sonku.service -f -n "${1:-100}"
