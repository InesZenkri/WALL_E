"""Tests for the optional API key and movement limit.

Run from the brain/ folder: pip install -e . pytest httpx && pytest ../tests
(set the env vars below before brain is imported, so this file does it itself).
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ["BRAIN_API_KEY"] = "secret"
os.environ["MAX_MOVE_METERS"] = "5"

from fastapi.testclient import TestClient  # noqa: E402

from brain.controller import controller  # noqa: E402
from brain.routes import app  # noqa: E402

client = TestClient(app)


def test_stop_rejects_missing_key():
    assert client.post("/stop").status_code == 401


def test_stop_rejects_wrong_key():
    assert client.post("/stop", headers={"X-API-Key": "nope"}).status_code == 401


def test_command_rejects_missing_key():
    assert client.post("/command", json={"message": "hi"}).status_code == 401


def test_stop_accepts_right_key(monkeypatch):
    called = []
    monkeypatch.setattr(controller, "stop_event", lambda: called.append(True))
    r = client.post("/stop", headers={"X-API-Key": "secret"})
    assert r.status_code == 200
    assert called == [True]


def test_root_stays_open():
    assert client.get("/").status_code == 200


def test_move_over_limit_is_rejected():
    result = controller.move(10, 0)
    assert result["status"] is False
    assert "limit" in result["error"]
