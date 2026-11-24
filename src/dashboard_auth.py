"""Dashboard authentication.

Two layers of defence for every endpoint:

1. Localhost-only: `request.client.host` must be a loopback address.
2. Bearer token: `Authorization: Bearer <DASHBOARD_SECRET_KEY>`.

If `DASHBOARD_SECRET_KEY` is unset, all protected requests return 503
(fail closed). An unset token is a configuration error, not a
permissive default. This is deliberate — the dashboard cannot be
operated until the operator sets a token.
"""

import hmac
from typing import Optional, Tuple

from fastapi import HTTPException, Request

from src.config_loader import config
from src.logger import get_logger

logger = get_logger(__name__)


LOOPBACK_HOSTS = frozenset({
    "127.0.0.1",
    "localhost",
    "::1",
    "::ffff:127.0.0.1",
})


def check_request_allowed(
    client_host: Optional[str],
    auth_header: Optional[str],
    configured_token: Optional[str],
) -> Tuple[bool, int, str]:
    """Pure function; testable without FastAPI."""
    if client_host not in LOOPBACK_HOSTS:
        return False, 403, "Access denied: localhost only."

    if not configured_token:
        return False, 503, (
            "Dashboard token is not configured. "
            "Set DASHBOARD_SECRET_KEY in .env."
        )

    if not auth_header:
        return False, 401, "Missing Authorization header."

    if not auth_header.startswith("Bearer "):
        return False, 401, "Authorization header must be 'Bearer <token>'."

    provided = auth_header[7:].strip()

    if not hmac.compare_digest(provided, configured_token):
        return False, 401, "Invalid dashboard token."

    return True, 200, "OK"


def check_websocket_allowed(
    client_host: Optional[str],
    provided_token: Optional[str],
    configured_token: Optional[str],
) -> Tuple[bool, int, str]:
    """WS variant: token as query param. Return (allowed, close_code, reason)."""
    if client_host not in LOOPBACK_HOSTS:
        return False, 1008, "WebSocket refused: localhost only."

    if not configured_token:
        return False, 1011, (
            "WebSocket refused: dashboard token not configured."
        )

    if not provided_token:
        return False, 1008, "WebSocket refused: missing token."

    if not hmac.compare_digest(provided_token, configured_token):
        return False, 1008, "WebSocket refused: invalid token."

    return True, 1000, "OK"


async def require_auth(request: Request) -> bool:
    """FastAPI dependency; enforces localhost + token."""
    client_host = request.client.host if request.client else None
    auth_header = request.headers.get("authorization")
    configured = config.dashboard.secret_key

    allowed, status, reason = check_request_allowed(
        client_host, auth_header, configured
    )
    if not allowed:
        if client_host not in LOOPBACK_HOSTS:
            logger.warning(
                "Dashboard auth rejected | host=%s | reason=%s",
                client_host, reason,
            )
        raise HTTPException(status_code=status, detail=reason)
    return True