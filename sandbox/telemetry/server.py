"""REST Telemetry microservice on port 8090 with Bearer Token authentication.

Timestamp: 2026-11-11 13:00 (Asia/Shanghai)
"""

import json
import os
from typing import Any, Dict, Optional

from aiohttp import web

# Lazy reference to engine
_engine_instance = None


def set_engine_instance(engine: Any) -> None:
    global _engine_instance
    _engine_instance = engine


def get_engine() -> Any:
    global _engine_instance
    if _engine_instance is None:
        from daemon.engine import StreamingEngine  # Bug 7 fix: absolute import (sandbox root in sys.path)
        _engine_instance = StreamingEngine()
    return _engine_instance


def verify_bearer_auth(request: web.Request) -> None:
    """Check Authorization: Bearer <token> header."""
    expected_token = os.environ.get("TELEMETRY_BEARER_TOKEN", "change_me_telemetry_bearer_token")
    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        raise web.HTTPUnauthorized(
            text=json.dumps({"detail": "Missing Authorization header"}),
            content_type="application/json",
            headers={"WWW-Authenticate": "Bearer"},
        )
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise web.HTTPUnauthorized(
            text=json.dumps({"detail": "Invalid Bearer token format"}),
            content_type="application/json",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if parts[1] != expected_token:
        raise web.HTTPUnauthorized(
            text=json.dumps({"detail": "Unauthorized: invalid telemetry token"}),
            content_type="application/json",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "hybrid-cascade-sandbox-telemetry"})


async def handle_overview(request: web.Request) -> web.Response:
    verify_bearer_auth(request)
    engine = get_engine()
    snapshot = engine.get_telemetry_snapshot()
    return web.json_response(snapshot["overview"])


async def handle_portfolio(request: web.Request) -> web.Response:
    verify_bearer_auth(request)
    engine = get_engine()
    snapshot = engine.get_telemetry_snapshot()
    return web.json_response(snapshot["portfolio"])


async def handle_paired_metrics(request: web.Request) -> web.Response:
    verify_bearer_auth(request)
    engine = get_engine()
    snapshot = engine.get_telemetry_snapshot()
    return web.json_response(snapshot["paired_metrics"])


async def handle_reasoning_stream(request: web.Request) -> web.Response:
    verify_bearer_auth(request)
    limit_str = request.query.get("limit", "20")
    try:
        limit = max(1, min(100, int(limit_str)))
    except ValueError:
        limit = 20

    engine = get_engine()
    snapshot = engine.get_telemetry_snapshot()
    stream = snapshot["reasoning_stream"]
    return web.json_response(stream[-limit:])


def create_telemetry_app() -> web.Application:
    """Factory creating the aiohttp web Application."""
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/api/v1/telemetry/overview", handle_overview)
    app.router.add_get("/api/v1/telemetry/portfolio", handle_portfolio)
    app.router.add_get("/api/v1/telemetry/paired_metrics", handle_paired_metrics)
    app.router.add_get("/api/v1/telemetry/reasoning_stream", handle_reasoning_stream)
    return app
