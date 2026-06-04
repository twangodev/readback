from __future__ import annotations

from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

NO_STORE = {"Cache-Control": "no-store"}
AUDIO_CACHE = "public, max-age=31536000, immutable"


class StudioService(Protocol):
    def queue(self) -> dict: ...
    def plan(self) -> dict: ...
    def stats(self) -> dict: ...
    def context(self, utterance_id: str) -> dict | None: ...
    def audio(self, utterance_id: str) -> bytes: ...
    def record(self, utterance_id: str, payload: dict) -> dict: ...
    def undo(self, utterance_id: str) -> dict | None: ...


class ReviewRequest(BaseModel):
    decision: str
    transcript: str
    base_hyp: str


def create_app(service: StudioService, web_dir: Path | None = None) -> FastAPI:
    app = FastAPI()

    @app.get("/api/queue")
    def queue() -> Response:
        return JSONResponse(service.queue(), headers=NO_STORE)

    @app.get("/api/plan")
    def plan() -> Response:
        return JSONResponse(service.plan(), headers=NO_STORE)

    @app.get("/api/queue/stats")
    def stats() -> Response:
        return JSONResponse(service.stats(), headers=NO_STORE)

    @app.get("/api/clips/{utterance_id:path}/audio")
    def audio(utterance_id: str, request: Request) -> Response:
        etag = f'"{utterance_id}"'
        headers = {"ETag": etag, "Cache-Control": AUDIO_CACHE}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        try:
            data = service.audio(utterance_id)
        except KeyError:
            return Response(status_code=404)
        return Response(content=data, media_type="audio/wav", headers=headers)

    @app.get("/api/clips/{utterance_id:path}")
    def clip(utterance_id: str) -> Response:
        context = service.context(utterance_id)
        if context is None:
            return Response(status_code=404)
        return JSONResponse(context, headers=NO_STORE)

    @app.post("/api/review/{utterance_id:path}")
    def review(utterance_id: str, body: ReviewRequest) -> Response:
        row = service.record(utterance_id, body.model_dump())
        return JSONResponse({**row, "reviewed": True}, headers=NO_STORE)

    @app.delete("/api/review/{utterance_id:path}")
    def unreview(utterance_id: str) -> Response:
        restored = service.undo(utterance_id)
        return JSONResponse(
            {"utterance_id": utterance_id, "restored": restored}, headers=NO_STORE
        )

    if web_dir is not None:
        from readback.server.spa import mount_spa

        mount_spa(app, web_dir)

    return app
