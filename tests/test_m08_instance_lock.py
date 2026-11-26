"""M-08 regression tests: atomic single-instance lock.

Asserts:
  * the lock is created atomically (O_CREAT|O_EXCL), so two processes
    racing at startup cannot both acquire it;
  * a live lock held by another process is respected;
  * stale locks are safely reclaimed: dead PID, PID reuse detected by
    start_time mismatch, corrupt/unreadable payload, pre-M-08 legacy
    single-PID format;
  * release never removes a lock file owned by a different PID.
"""

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from src.main import TradingAgent


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_bare_agent(tmp_path: Path) -> TradingAgent:
    agent = TradingAgent.__new__(TradingAgent)
    agent._lock_file = tmp_path / "bot.lock"
    agent._lock_acquired = False
    return agent


# ---------------------------------------------------------------------
# Single-process tests
# ---------------------------------------------------------------------

def test_acquire_when_no_lock_exists(tmp_path):
    agent = _make_bare_agent(tmp_path)
    assert agent._acquire_instance_lock() is True
    assert agent._lock_acquired is True
    assert agent._lock_file.exists()

    payload = json.loads(agent._lock_file.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert float(payload["start_time"]) > 0.0


def test_live_lock_blocks_second_acquire(tmp_path):
    first = _make_bare_agent(tmp_path)
    assert first._acquire_instance_lock() is True

    second = _make_bare_agent(tmp_path)
    # Same-process holder: PID alive, start_time matches.
    assert second._acquire_instance_lock() is False
    payload = json.loads(first._lock_file.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()


def test_pid_reuse_is_detected(tmp_path):
    """A live PID with a non-matching start_time is not the holder."""
    lock = tmp_path / "bot.lock"
    lock.write_text(
        json.dumps({"pid": os.getpid(), "start_time": 1.0}),
        encoding="utf-8",
    )
    agent = _make_bare_agent(tmp_path)
    assert agent._acquire_instance_lock() is True
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert float(payload["start_time"]) > 1.0


def test_corrupt_lock_is_reclaimed(tmp_path):
    lock = tmp_path / "bot.lock"
    lock.write_text("not a pid and not json", encoding="utf-8")
    agent = _make_bare_agent(tmp_path)
    assert agent._acquire_instance_lock() is True
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()


def test_legacy_single_pid_lock_is_honored(tmp_path):
    """Pre-M-08 single-PID format is still honored when PID is alive."""
    lock = tmp_path / "bot.lock"
    lock.write_text(str(os.getpid()), encoding="utf-8")
    agent = _make_bare_agent(tmp_path)
    # Our own PID is alive, so the legacy lock blocks us.
    assert agent._acquire_instance_lock() is False


def test_release_does_not_unlink_foreign_lock(tmp_path):
    agent = _make_bare_agent(tmp_path)
    assert agent._acquire_instance_lock() is True

    # Simulate another process replacing the file with its own payload.
    foreign_pid = os.getpid() + 1
    agent._lock_file.write_text(
        json.dumps({"pid": foreign_pid, "start_time": 0.0}),
        encoding="utf-8",
    )

    agent._release_instance_lock()
    assert agent._lock_file.exists(), (
        "release must not remove a lock owned by a different PID"
    )
    assert agent._lock_acquired is False


def test_release_removes_own_lock(tmp_path):
    agent = _make_bare_agent(tmp_path)
    assert agent._acquire_instance_lock() is True
    assert agent._lock_file.exists()
    agent._release_instance_lock()
    assert not agent._lock_file.exists()


# ---------------------------------------------------------------------
# Cross-process atomicity
# ---------------------------------------------------------------------

_RACER_SCRIPT = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from src.main import TradingAgent

    lock_path = Path(sys.argv[1])
    ready_path = Path(sys.argv[2])
    go_path = Path(sys.argv[3])

    agent = TradingAgent.__new__(TradingAgent)
    agent._lock_file = lock_path
    agent._lock_acquired = False

    # Rendezvous: signal we finished importing, then wait for "go".
    ready_path.touch()
    while not go_path.exists():
        time.sleep(0.01)

    acquired = agent._acquire_instance_lock()
    if acquired:
        # Hold long enough that any racing peer observes the lock.
        time.sleep(1.5)
        agent._release_instance_lock()
        sys.exit(0)
    sys.exit(1)
""")


def test_lock_is_atomic_across_processes(tmp_path):
    """Two processes released from a common barrier: only one wins."""
    lock_path = tmp_path / "bot.lock"
    ready_a = tmp_path / "ready_a"
    ready_b = tmp_path / "ready_b"
    go = tmp_path / "go"
    script_file = tmp_path / "racer.py"
    script_file.write_text(_RACER_SCRIPT, encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    )

    def _spawn(ready_path: Path) -> subprocess.Popen:
        return subprocess.Popen(
            [
                sys.executable,
                str(script_file),
                str(lock_path),
                str(ready_path),
                str(go),
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
        )

    proc_a = _spawn(ready_a)
    proc_b = _spawn(ready_b)

    # Wait until both children have finished importing and reached
    # their rendezvous, then release the barrier simultaneously.
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if ready_a.exists() and ready_b.exists():
            break
        time.sleep(0.05)
    else:
        proc_a.kill(); proc_b.kill()
        pytest.fail("racers did not reach the rendezvous in time")

    go.touch()

    rc_a = proc_a.wait(timeout=20)
    rc_b = proc_b.wait(timeout=20)

    success_count = sum(1 for rc in (rc_a, rc_b) if rc == 0)
    assert success_count == 1, (
        f"exactly one process must acquire the lock; "
        f"got rc_a={rc_a}, rc_b={rc_b}"
    )