"""Admin web panel + JSON API. Everything sits behind HTTP Basic auth; state-changing calls also
need an `X-Sonku: 1` header, which a cross-site page cannot send (CSRF protection)."""
from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from app import __version__
from app.logging_setup import RING
from app.runtime import Runtime

log = logging.getLogger("sonku.web")
STATIC = Path(__file__).parent / "static"


class NameBody(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class SayBody(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    lang: str = "en"


class ChatBody(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    speak: bool = False


def create_app(rt: Runtime) -> FastAPI:
    s = rt.s

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        rt.start()
        yield
        rt.stop()

    app = FastAPI(title="Sonku admin", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)

    password = s.admin_password
    if not password:
        password = secrets.token_urlsafe(9)
        log.warning("ADMIN_PASSWORD is not set. Temporary admin login for this run -> user: %s  password: %s", s.admin_user, password)
    basic = HTTPBasic()

    def auth(creds: HTTPBasicCredentials = Depends(basic)) -> None:
        ok_user = secrets.compare_digest(creds.username.encode(), s.admin_user.encode())
        ok_pass = secrets.compare_digest(creds.password.encode(), password.encode())
        if not (ok_user and ok_pass):
            raise HTTPException(401, "Wrong username or password", headers={"WWW-Authenticate": 'Basic realm="Sonku"'})

    def csrf(request: Request) -> None:
        if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get("x-sonku") != "1":
            raise HTTPException(403, "Missing X-Sonku header")

    api = APIRouter(prefix="/api", dependencies=[Depends(auth), Depends(csrf)])

    # ---- status ---------------------------------------------------------------
    @api.get("/status")
    def status() -> dict:
        return rt.status()

    @api.get("/camera/snapshot")
    def snapshot() -> Response:
        data = (rt.watcher.last_jpeg if rt.watcher else None) or (rt.camera.jpeg() if rt.camera else None)
        if not data:
            raise HTTPException(503, "Camera is not available")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @api.post("/audio/rescan")
    def rescan_audio() -> dict:
        rt.audio.scan()
        return rt.audio.status()

    # ---- people ---------------------------------------------------------------
    @api.get("/people")
    def people() -> list[dict]:
        return rt.store.list_people()

    @api.post("/people/enroll")
    def enroll(body: NameBody) -> dict:
        if not rt.watcher:
            raise HTTPException(503, "Camera or face models are not available")
        embs = rt.watcher.capture_embeddings(max(1, min(s.enroll_templates, s.faces_per_person)), spacing=0.7)
        if not embs:
            raise HTTPException(422, "No face seen. Stand in front of Sonku and try again.")
        known, _ = rt.faces.match(embs[0], rt.store.gallery())
        if known:
            raise HTTPException(409, f"This face is already stored as '{rt.store.get_person(known)['name']}'")
        pid = rt.store.add_person(body.name)
        for e in embs:
            rt.store.add_face(pid, e)
        rt.store.mark_greeted(pid)
        log.info("admin enrolled %s", body.name)
        return {"id": pid, "faces": len(embs)}

    @api.patch("/people/{pid}")
    def rename(pid: int, body: NameBody) -> dict:
        if not rt.store.rename_person(pid, body.name):
            raise HTTPException(404, "No such person")
        return {"ok": True}

    @api.delete("/people/{pid}")
    def delete_person(pid: int) -> dict:
        if not rt.store.delete_person(pid):
            raise HTTPException(404, "No such person")
        if rt.watcher:
            rt.watcher.reset_identity()
        log.info("admin deleted person %s (face + conversations)", pid)
        return {"ok": True}

    @api.delete("/people")
    def delete_everyone(confirm: str = "") -> dict:
        if confirm != "yes":
            raise HTTPException(400, "Add ?confirm=yes to delete every person, face and conversation")
        n = rt.store.delete_all_people()
        if rt.watcher:
            rt.watcher.reset_identity()
        log.warning("admin deleted ALL people (%d)", n)
        return {"deleted": n}

    @api.delete("/people/{pid}/summary")
    def clear_summary(pid: int) -> dict:
        rt.store.set_summary(pid, "")
        return {"ok": True}

    # ---- conversations --------------------------------------------------------
    @api.get("/interactions")
    def interactions(limit: int = 100, person_id: int | None = None) -> list[dict]:
        return rt.store.list_interactions(min(max(limit, 1), 500), person_id)

    @api.delete("/interactions")
    def clear_interactions(person_id: int | None = None) -> dict:
        return {"deleted": rt.store.clear_interactions(person_id)}

    # ---- memory ---------------------------------------------------------------
    @api.get("/memory")
    def memory() -> dict:
        return {**rt.store.usage(), "alert": rt.maint.alert, "last_maintenance": rt.maint.last_run, "last_result": rt.maint.last_result}

    @api.post("/memory/maintenance")
    def run_maintenance() -> dict:
        return rt.maint.trigger()

    # ---- settings -------------------------------------------------------------
    @api.get("/settings")
    def get_settings() -> list[dict]:
        return s.editable_view()

    @api.put("/settings")
    def put_settings(values: dict[str, Any]) -> dict:
        applied = {}
        for key, raw in values.items():
            try:
                applied[key] = s.set_value(key, raw)
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, str(exc))
        for key, val in applied.items():
            rt.store.save_setting(key, val)
        log.info("settings changed from admin panel: %s", applied)
        return {"applied": applied}

    # ---- robot control / testing ---------------------------------------------
    @api.post("/control/{action}")
    def control(action: str) -> dict:
        if action == "pause":
            rt.orch.paused = True
        elif action == "resume":
            rt.orch.paused = False
        elif action == "end-session":
            rt.orch.end_session()
        else:
            raise HTTPException(404, "Unknown action")
        return rt.orch.describe()

    @api.post("/say")
    def say(body: SayBody) -> dict:
        rt.orch.say(body.text, body.lang)
        return {"ok": True}

    @api.post("/chat")
    def chat(body: ChatBody) -> dict:
        return rt.orch.text_turn(body.text, speak=body.speak)

    @api.get("/logs")
    def logs(n: int = 200, level: str = "INFO") -> list[dict]:
        return RING.tail(min(max(n, 1), 1000), level)

    app.include_router(api)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/", dependencies=[Depends(auth)])
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    return app
