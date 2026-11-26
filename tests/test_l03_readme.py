"""Focused L-03 test: README replaced placeholder with operational docs."""
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"


@pytest.fixture(scope="module")
def text() -> str:
    assert README.is_file(), f"README.md not found at {README}"
    return README.read_text(encoding="utf-8")


def test_no_placeholder_markers(text):
    for marker in (
        "[Insert project description here]",
        "[List dependencies here]",
        "[Step 1]",
        "[Step 2]",
        "[Explain how to use the project]",
        "[Guidelines for contributing]",
        "[Project license information]",
    ):
        assert marker not in text, f"placeholder still present: {marker}"


def test_critical_operational_sections_present(text):
    for heading in (
        "## Description",
        "## Prerequisites",
        "## Configuration",
        "## Startup",
        "## Shutdown",
        "## Recovery",
        "## Risk configuration",
    ):
        assert heading in text, f"missing section: {heading}"


def test_documented_commands_are_accurate(text):
    for needle in (
        "python main.py",
        "uvicorn dashboard.backend.main:app",
        "bot.lock",
        "/api/health",
        "DASHBOARD_SECRET_KEY",
    ):
        assert needle in text, f"missing accurate command/identifier: {needle}"


def test_live_guard_documented(text):
    for needle in (
        "LIVE_TRADING_ENABLED",
        "LIVE_ALLOWED_ACCOUNT",
        "LIVE_TRADING_CONFIRMATION",
        "I_ACCEPT_LIVE_TRADING_RISK",
        "demo_only",
    ):
        assert needle in text, f"LIVE guard element missing: {needle}"


def test_recovery_and_kill_switch_documented(text):
    for needle in (
        "reconciliation",
        "Kill switch",
        "I_UNDERSTAND",
    ):
        assert needle in text, f"operational concept missing: {needle}"

def test_readme_starts_with_project_title(text):
    """Guard: a paste that includes a chat wrapper must fail loudly."""
    stripped = text.lstrip()
    assert stripped.startswith("# Claude-XAU-USD"), (
        "README does not begin with the project title; it may have "
        f"chat-wrapper text above it. First 80 chars: {stripped[:80]!r}"
    )
    assert "Replace the entire contents" not in text
    assert "## Complete `README.md`" not in text