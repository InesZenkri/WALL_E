"""Tiny fake robot, so the backend can run without hardware.

It serves the two endpoints the backend uses:
  GET  /api/ros/sew_bot/pose       current pose
  POST /api/ros/sew_bot/goal_pose  new target; the fake robot drives there at SPEED m/s

Start it:   python mock_robot.py            (listens on port 8081)
Point the backend at it:   URL=http://localhost:8081 python run.py
"""
import math
import threading
import time

import uvicorn
from fastapi import FastAPI

SPEED = 1.0  # meters per second
TICK = 0.1  # seconds

app = FastAPI(title="Mock robot")
_lock = threading.Lock()
_pose = {"x": 0.0, "y": 0.0}
_goal = {"x": 0.0, "y": 0.0}


def _drive():
    while True:
        time.sleep(TICK)
        with _lock:
            dx, dy = _goal["x"] - _pose["x"], _goal["y"] - _pose["y"]
            dist = math.hypot(dx, dy)
            step = SPEED * TICK
            if dist <= step:
                _pose.update(_goal)
            else:
                _pose["x"] += dx / dist * step
                _pose["y"] += dy / dist * step


threading.Thread(target=_drive, daemon=True).start()


@app.get("/api/ros/sew_bot/pose")
def get_pose():
    with _lock:
        x, y = _pose["x"], _pose["y"]
    return {
        "transform": {
            "translation": {"x": x, "y": y, "z": 0.0},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        }
    }


@app.post("/api/ros/sew_bot/goal_pose")
def goal_pose(payload: dict):
    position = payload["pose"]["position"]
    with _lock:
        _goal["x"], _goal["y"] = float(position["x"]), float(position["y"])
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081, log_level="warning")
