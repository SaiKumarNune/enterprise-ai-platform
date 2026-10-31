"""HTTP middleware for request IDs, JSON access logs, and metrics."""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match

from app.core.logging import log_event
from app.core.metrics import observe_http
from app.core.request_context import REQUEST_ID_HEADER, bind_request_id, reset_request_id, validate_request_id

logger = logging.getLogger("app.http")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = validate_request_id(request.headers.get(REQUEST_ID_HEADER))
        token = bind_request_id(request_id)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            duration_ms = int((time.perf_counter() - started) * 1000)
            route = _route_template(request)
            observe_http(request.method, route, status_code, (time.perf_counter() - started))
            log_event(
                logger,
                "http.request",
                method=request.method,
                route=route,
                status=status_code,
                duration_ms=duration_ms,
            )
            reset_request_id(token)


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    for attr in ("path_format", "path"):
        path = getattr(route, attr, None)
        if isinstance(path, str) and path.startswith("/"):
            return path
    matched = _match_registered_route(request)
    if matched:
        return matched
    return "unmatched"


def _match_registered_route(request: Request) -> str | None:
    """Resolve templates for routes that never set scope['route'] (e.g. /docs)."""
    routes = getattr(getattr(request, "app", None), "routes", None) or []
    for route in routes:
        matches = getattr(route, "matches", None)
        if not callable(matches):
            continue
        try:
            match, _child = matches(request.scope)
        except Exception:
            continue
        if match is not Match.FULL:
            continue
        for attr in ("path_format", "path"):
            path = getattr(route, attr, None)
            if isinstance(path, str) and path.startswith("/"):
                return path
    return None
