from __future__ import annotations

import json
import mimetypes
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from src.dashboard.catalog import (
    get_state_record,
    load_reward_profiles,
    load_state_library,
    save_reward_profile,
    save_state_record,
)
from src.dashboard.jobs import JobManager
from src.dashboard.metadata import build_dashboard_bootstrap
from src.dashboard.runs import get_run_detail, list_runs, update_run_meta


class DashboardApplication:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.static_dir = self.project_root / "dashboard_static"
        self.job_manager = JobManager(self.project_root)

    def bootstrap_payload(self) -> dict[str, Any]:
        payload = build_dashboard_bootstrap(self.project_root)
        payload["runs"] = list_runs(self.project_root)
        payload["jobs"] = self.job_manager.list_jobs()
        payload["replay_sources"] = self._replay_sources()
        return payload

    def _replay_sources(self) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for candidate in [self.project_root / "replays_ssl_like", self.project_root / "xern_replays", self.project_root.parent / "DataState"]:
            if not candidate.exists():
                continue
            sources.append(
                {
                    "path": str(candidate.resolve()),
                    "name": candidate.name,
                    "replay_count": len(list(candidate.glob("*.replay"))),
                    "csv_count": len(list(candidate.rglob("*.csv"))),
                }
            )
        return sources

    def list_states(self) -> list[dict[str, Any]]:
        return load_state_library(self.project_root)

    def load_state(self, state_id: str) -> dict[str, Any]:
        return get_state_record(self.project_root, state_id)

    def save_state(self, state_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return save_state_record(self.project_root, state_id, payload)

    def load_reward_profiles(self) -> dict[str, Any]:
        return load_reward_profiles(self.project_root)

    def save_reward_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        return save_reward_profile(
            self.project_root,
            str(payload.get("name") or payload.get("id") or "Nouveau profil"),
            dict(payload.get("weights", {})),
            profile_id=payload.get("id"),
            set_active=bool(payload.get("set_active", True)),
        )

    def mirror_scenario(self, payload: dict[str, Any]) -> dict[str, Any]:
        mirrored = json.loads(json.dumps(payload))
        ball = mirrored.get("ball", {}).get("position", {})
        if ball:
            ball["x"] = -float(ball.get("x", 0.0))
            ball["y"] = -float(ball.get("y", 0.0))
        ball_random = mirrored.get("ball", {}).get("position_random", {})
        if ball_random.get("enabled"):
            min_x = -float(ball_random.get("max_x", 0.0))
            max_x = -float(ball_random.get("min_x", 0.0))
            min_y = -float(ball_random.get("max_y", 0.0))
            max_y = -float(ball_random.get("min_y", 0.0))
            ball_random["min_x"] = min(min_x, max_x)
            ball_random["max_x"] = max(min_x, max_x)
            ball_random["min_y"] = min(min_y, max_y)
            ball_random["max_y"] = max(min_y, max_y)
        for car in mirrored.get("cars", []):
            position = car.get("position", {})
            position["x"] = -float(position.get("x", 0.0))
            position["y"] = -float(position.get("y", 0.0))
            yaw = car.get("yaw", {})
            yaw["value"] = (float(yaw.get("value", 0.0)) + 3.141592653589793) % (2.0 * 3.141592653589793)
            if yaw.get("enabled"):
                min_yaw = (float(yaw.get("min", yaw["value"])) + 3.141592653589793) % (2.0 * 3.141592653589793)
                max_yaw = (float(yaw.get("max", yaw["value"])) + 3.141592653589793) % (2.0 * 3.141592653589793)
                yaw["min"] = min(min_yaw, max_yaw)
                yaw["max"] = max(min_yaw, max_yaw)
            pos_random = car.get("position_random", {})
            if pos_random.get("enabled"):
                min_x = -float(pos_random.get("max_x", 0.0))
                max_x = -float(pos_random.get("min_x", 0.0))
                min_y = -float(pos_random.get("max_y", 0.0))
                max_y = -float(pos_random.get("min_y", 0.0))
                pos_random["min_x"] = min(min_x, max_x)
                pos_random["max_x"] = max(min_x, max_x)
                pos_random["min_y"] = min(min_y, max_y)
                pos_random["max_y"] = max(min_y, max_y)
            car["team"] = 1 - int(car.get("team", 0))
            if str(car.get("name", "")).startswith("blue"):
                car["name"] = str(car["name"]).replace("blue", "orange", 1)
            elif str(car.get("name", "")).startswith("orange"):
                car["name"] = str(car["name"]).replace("orange", "blue", 1)
        return mirrored


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "RocketRLDashboard/2.0"

    @property
    def app(self) -> DashboardApplication:
        return self.server.app  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/bootstrap":
                return self._json(self.app.bootstrap_payload())
            if path == "/api/runs":
                return self._json({"runs": list_runs(self.app.project_root)})
            if path.startswith("/api/runs/"):
                run_name = urllib.parse.unquote(path.split("/api/runs/", 1)[1])
                return self._json(get_run_detail(self.app.project_root, run_name))
            if path == "/api/jobs":
                return self._json({"jobs": self.app.job_manager.list_jobs()})
            if path.startswith("/api/jobs/") and path.endswith("/log"):
                job_id = path.split("/api/jobs/", 1)[1].rsplit("/log", 1)[0]
                return self._json({"job_id": job_id, "log": self.app.job_manager.read_log_tail(job_id)})
            if path.startswith("/api/jobs/"):
                job_id = path.split("/api/jobs/", 1)[1]
                return self._json(self.app.job_manager.get_job(job_id))
            if path == "/api/states":
                return self._json({"states": self.app.list_states()})
            if path.startswith("/api/states/"):
                state_id = urllib.parse.unquote(path.split("/api/states/", 1)[1])
                return self._json(self.app.load_state(state_id))
            if path == "/api/reward-profiles":
                return self._json(self.app.load_reward_profiles())
            return self._serve_static(path)
        except FileNotFoundError as exc:
            if self._client_disconnected(exc):
                return None
            return self._json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            if self._client_disconnected(exc):
                return None
            return self._json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pragma: no cover
            if self._client_disconnected(exc):
                return None
            return self._json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        payload = self._read_json_body()
        try:
            if path == "/api/jobs/train":
                record = self.app.job_manager.launch("train", payload.get("title", "Training"), payload)
                return self._json(record.__dict__, status=HTTPStatus.ACCEPTED)
            if path == "/api/jobs/pretrain":
                record = self.app.job_manager.launch("pretrain", payload.get("title", "Replay pretraining"), payload)
                return self._json(record.__dict__, status=HTTPStatus.ACCEPTED)
            if path == "/api/jobs/evaluate":
                record = self.app.job_manager.launch("evaluate", payload.get("title", "Evaluation"), payload)
                return self._json(record.__dict__, status=HTTPStatus.ACCEPTED)
            if path == "/api/jobs/export":
                record = self.app.job_manager.launch("export", payload.get("title", "RLBot export"), payload)
                return self._json(record.__dict__, status=HTTPStatus.ACCEPTED)
            if path.startswith("/api/runs/") and path.endswith("/meta"):
                run_name = urllib.parse.unquote(path.split("/api/runs/", 1)[1].rsplit("/meta", 1)[0])
                return self._json(update_run_meta(self.app.project_root, run_name, payload))
            if path == "/api/states/save":
                state_id = str(payload.get("id") or payload.get("state_id") or "custom_state")
                return self._json(self.app.save_state(state_id, dict(payload.get("state", payload))))
            if path == "/api/reward-profiles/save":
                return self._json(self.app.save_reward_profile(payload))
            if path == "/api/scenarios/mirror":
                return self._json({"scenario": self.app.mirror_scenario(dict(payload.get("scenario", {})))})
            return self._json({"error": f"Unknown POST route: {path}"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pragma: no cover
            if self._client_disconnected(exc):
                return None
            return self._json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return None

    def _serve_static(self, raw_path: str) -> None:
        path = raw_path or "/"
        if path == "/":
            path = "/index.html"
        candidate = (self.app.static_dir / path.lstrip("/")).resolve()
        if not str(candidate).startswith(str(self.app.static_dir.resolve())) or not candidate.exists() or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        mime_type, _encoding = mimetypes.guess_type(str(candidate))
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.end_headers()
            self.wfile.write(candidate.read_bytes())
        except OSError as exc:
            if not self._client_disconnected(exc):
                raise

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        payload = self.rfile.read(content_length)
        return json.loads(payload.decode("utf-8")) if payload else {}

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8")
        try:
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError as exc:
            if not self._client_disconnected(exc):
                raise

    @staticmethod
    def _client_disconnected(exc: BaseException) -> bool:
        return isinstance(exc, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError))


def create_server(project_root: Path, host: str, port: int) -> ThreadingHTTPServer:
    app = DashboardApplication(Path(project_root))
    server = ThreadingHTTPServer((host, int(port)), DashboardHandler)
    server.app = app  # type: ignore[attr-defined]
    return server
