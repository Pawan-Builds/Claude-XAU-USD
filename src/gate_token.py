"""One-time, HMAC-signed authorization tokens for opening orders.

Purpose
-------
``EntryGate.check()`` issues a short-lived token on success. Any code
that opens a position — today only ``OrderExecutor.execute_signal``,
but the token also defends any *future* caller of
``mt5_connector.send_order`` — must supply that token. The connector
verifies and consumes it before calling ``mt5.order_send`` for an
opening action. Closing, modifying, and cancelling actions do NOT
require a token, so emergency exits always work.

Properties
----------
- HMAC-SHA256 over ``<nonce>:<expires>`` using a per-process random key.
- Per-process, in-memory consumed set; tokens do not survive restart.
  This is intentional: after a restart the agent has not authorized
  any submission yet.
- TTL 30 seconds. Expired entries are purged lazily on each consume.
- Replay resistant within TTL: a consumed nonce cannot be reused.
"""

import hashlib
import hmac
import os
import secrets
import time
from threading import Lock
from typing import Optional

_SECRET: bytes = os.urandom(32)
_TTL_SECONDS: int = 30
_CONSUMED: dict = {}
_LOCK = Lock()


def issue() -> str:
    """Return a fresh token: ``<nonce>.<expires_epoch>.<sig_hex>``."""
    nonce = secrets.token_hex(16)
    expires = int(time.time()) + _TTL_SECONDS
    message = f"{nonce}:{expires}".encode("ascii")
    signature = hmac.new(_SECRET, message, hashlib.sha256).hexdigest()
    return f"{nonce}.{expires}.{signature}"


def consume(token: Optional[str]) -> bool:
    """Validate and consume ``token``.

    Returns True only if the token is well formed, unexpired,
    signature-valid, and has not been consumed before. On success the
    nonce is recorded so the same token cannot be used twice.
    """
    if not token or not isinstance(token, str):
        return False

    parts = token.split(".")
    if len(parts) != 3:
        return False

    nonce, expires_str, signature = parts
    try:
        expires = int(expires_str)
    except ValueError:
        return False

    now = int(time.time())
    if expires < now:
        return False

    with _LOCK:
        # Purge any expired nonces before checking.
        if _CONSUMED:
            for key in [k for k, v in _CONSUMED.items() if v < now]:
                del _CONSUMED[key]

        if nonce in _CONSUMED:
            return False

        message = f"{nonce}:{expires}".encode("ascii")
        expected = hmac.new(
            _SECRET, message, hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected):
            return False

        _CONSUMED[nonce] = expires
        return True


def reset_for_tests() -> None:
    """Test-only. Clears the consumed set so tests are independent."""
    with _LOCK:
        _CONSUMED.clear()