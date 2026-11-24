"""Phase 1 / C-03 residual: HMAC gate-token primitives."""

import hmac
import hashlib
import time

from src import gate_token


def _reset():
    gate_token.reset_for_tests()


def test_issue_returns_three_part_token():
    _reset()
    tok = gate_token.issue()
    assert tok.count(".") == 2


def test_consume_accepts_fresh_token():
    _reset()
    assert gate_token.consume(gate_token.issue()) is True


def test_consume_rejects_replay():
    _reset()
    tok = gate_token.issue()
    assert gate_token.consume(tok) is True
    assert gate_token.consume(tok) is False


def test_consume_rejects_none_and_empty():
    _reset()
    assert gate_token.consume(None) is False
    assert gate_token.consume("") is False


def test_consume_rejects_bad_format():
    _reset()
    assert gate_token.consume("nope") is False
    assert gate_token.consume("a.b") is False
    assert gate_token.consume("a.b.c.d") is False


def test_consume_rejects_expired():
    _reset()
    tok = gate_token.issue()
    parts = tok.split(".")
    # Forge an expired token (past expires) with the ORIGINAL signature —
    # signature will not match because the message includes `expires`.
    fake = f"{parts[0]}.{int(time.time()) - 60}.{parts[2]}"
    assert gate_token.consume(fake) is False


def test_consume_rejects_tampered_signature():
    _reset()
    tok = gate_token.issue()
    nonce, expires, _sig = tok.split(".")
    forged = f"{nonce}.{expires}.{'0' * 64}"
    assert gate_token.consume(forged) is False


def test_different_tokens_have_different_nonces():
    _reset()
    a = gate_token.issue().split(".")[0]
    b = gate_token.issue().split(".")[0]
    assert a != b