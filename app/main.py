"""Process entry point: `python -m app`."""
import uvicorn

from app.config import Settings
from app.logging_setup import setup_logging
from app.runtime import Runtime
from app.web.server import create_app


def main() -> None:
    s = Settings.from_env()
    setup_logging(s)
    runtime = Runtime(s)
    app = create_app(runtime)
    uvicorn.run(app, host=s.web_host, port=s.web_port, log_level="warning", access_log=False)
