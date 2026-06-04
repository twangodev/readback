from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope


class SpaStaticFiles(StaticFiles):
    def __init__(self, web_dir: Path) -> None:
        super().__init__(directory=str(web_dir), html=True)
        self._fallback = web_dir / "200.html"

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code == 404:
                return FileResponse(self._fallback)
            raise


def mount_spa(app: FastAPI, web_dir: Path) -> None:
    app.mount("/", SpaStaticFiles(web_dir), name="spa")
