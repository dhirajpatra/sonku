"""Console + rotating file + in-memory ring buffer (served to the admin panel)."""
from __future__ import annotations

import logging
import logging.handlers
from collections import deque

from app.config import Settings


class RingHandler(logging.Handler):
    def __init__(self, capacity: int = 1000):
        super().__init__()
        self.buf: deque[dict] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + logging.Formatter().formatException(record.exc_info)
        self.buf.append({"ts": record.created, "level": record.levelname, "logger": record.name, "msg": msg})

    def tail(self, n: int = 200, min_level: str = "INFO") -> list[dict]:
        floor = logging.getLevelName(min_level.upper())
        floor = floor if isinstance(floor, int) else logging.INFO
        rows = [r for r in self.buf if logging.getLevelName(r["level"]) >= floor]
        return rows[-n:]


RING = RingHandler()


def setup_logging(s: Settings) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, s.log_level, logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    try:
        log_dir = s.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(log_dir / "sonku.log", maxBytes=5_000_000, backupCount=4)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError as exc:  # read-only disk etc: keep running with console only
        root.warning("file logging disabled: %s", exc)
    root.addHandler(RING)
    for noisy in ("urllib3", "multipart", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
