from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.dashboard.runs import list_runs
from src.utils.config_loader import load_project_configs


REWARD_PROFILE_PATH = Path("configs") / "dashboard_reward_profiles.json"
STATE_LIBRARY_PATH = Path("configs") / "state_library.json"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug or "item"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _reward_profile_file(project_root: Path) -> Path:
    return Path(project_root) / REWARD_PROFILE_PATH


def _state_library_file(project_root: Path) -> Path:
    return Path(project_root) / STATE_LIBRARY_PATH


def ensure_reward_profiles(project_root: Path) -> dict[str, Any]:
    path = _reward_profile_file(Path(project_root))
    payload = _read_json(path, None)
    if payload:
        return payload
    _config, rewards, _curriculum = load_project_configs(Path(project_root))
    default_weights = dict(rewards.get("weights", {}))
    payload = {
        "active_profile": "default",
        "profiles": [
            {"id": "default", "name": "Default", "weights": default_weights},
            {
                "id": "touch_focus",
                "name": "Touch Focus",
                "weights": {
                    **default_weights,
                    "touch_reward": float(default_weights.get("touch_reward", 0.0)) * 1.35,
                    "ball_touch_reward": max(float(default_weights.get("ball_touch_reward", 0.0)), 0.15),
                },
            },
            {
                "id": "defense_focus",
                "name": "Defense Focus",
                "weights": {
                    **default_weights,
                    "defense_position": max(float(default_weights.get("defense_position", 0.0)), 0.04),
                    "save_reward": max(float(default_weights.get("save_reward", 0.0)), 0.20),
                },
            },
        ],
    }
    _write_json(path, payload)
    return payload


def load_reward_profiles(project_root: Path) -> dict[str, Any]:
    payload = ensure_reward_profiles(Path(project_root))
    payload.setdefault("active_profile", "default")
    payload.setdefault("profiles", [])
    return payload


def save_reward_profile(
    project_root: Path,
    profile_name: str,
    weights: dict[str, Any],
    *,
    profile_id: str | None = None,
    set_active: bool = True,
) -> dict[str, Any]:
    payload = load_reward_profiles(Path(project_root))
    resolved_id = str(profile_id or _slugify(profile_name))
    profiles = list(payload.get("profiles", []))
    updated = False
    for profile in profiles:
        if profile.get("id") == resolved_id:
            profile["name"] = str(profile_name)
            profile["weights"] = {key: float(value) for key, value in dict(weights).items()}
            updated = True
            break
    if not updated:
        profiles.append({
            "id": resolved_id,
            "name": str(profile_name),
            "weights": {key: float(value) for key, value in dict(weights).items()},
        })
    payload["profiles"] = sorted(profiles, key=lambda item: str(item.get("name", item.get("id", ""))).lower())
    if set_active:
        payload["active_profile"] = resolved_id
    _write_json(_reward_profile_file(Path(project_root)), payload)
    return payload


def load_state_library(project_root: Path) -> list[dict[str, Any]]:
    payload = _read_json(_state_library_file(Path(project_root)), {"states": []})
    states = list(payload.get("states", []))
    for state in states:
        state.setdefault("enabled", True)
        state.setdefault("supported_team_sizes", [1, 2, 3])
        state.setdefault("definition", {})
        state.setdefault("reward_profile", "default")
        if not state.get("preview"):
            definition = dict(state.get("definition", {}))
            ball_position = definition.get("ball", {}).get("position", {})
            state["preview"] = {
                "ball": [
                    float(ball_position.get("x", 0.0)),
                    float(ball_position.get("y", 0.0)),
                    float(ball_position.get("z", 93.0)),
                ],
                "cars": [
                    {
                        "team": int(car.get("team", 0)),
                        "x": float(car.get("position", {}).get("x", 0.0)),
                        "y": float(car.get("position", {}).get("y", 0.0)),
                        "yaw": float(car.get("yaw", {}).get("value", 1.57 if int(car.get("team", 0)) == 0 else -1.57)),
                    }
                    for car in definition.get("cars", [])
                ],
            }
    return sorted(states, key=lambda item: (str(item.get("category", "custom")), str(item.get("name", item.get("id", "")))))


def get_state_record(project_root: Path, state_id: str) -> dict[str, Any]:
    for state in load_state_library(Path(project_root)):
        if str(state.get("id")) == str(state_id):
            return state
    raise FileNotFoundError(f"Unknown state: {state_id}")


def save_state_record(project_root: Path, state_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = _state_library_file(Path(project_root))
    library = _read_json(path, {"states": []})
    states = list(library.get("states", []))
    payload = dict(payload)
    payload["id"] = str(state_id)
    updated = False
    for index, state in enumerate(states):
        if str(state.get("id")) == str(state_id):
            states[index] = payload
            updated = True
            break
    if not updated:
        states.append(payload)
    library["states"] = sorted(states, key=lambda item: str(item.get("name", item.get("id", ""))).lower())
    _write_json(path, library)
    return get_state_record(Path(project_root), state_id)


def build_model_catalog(project_root: Path) -> list[dict[str, Any]]:
    config, _rewards, _curriculum = load_project_configs(Path(project_root))
    default_prefix = str(config.get("project", {}).get("run_name_prefix", "ppo_dashboard"))
    models: dict[str, dict[str, Any]] = {
        "new": {"id": "new", "name": "Nouveau modele", "run_name_prefix": default_prefix, "source": "default"}
    }
    for run in list_runs(Path(project_root)):
        display_name = str(run.get("display_name") or run.get("run_name") or "modele")
        model_id = _slugify(display_name)
        models[model_id] = {
            "id": model_id,
            "name": display_name,
            "run_name_prefix": str(run.get("run_name", default_prefix)).split("_20", 1)[0],
            "source": "run_history",
        }
    return sorted(models.values(), key=lambda item: (item["id"] != "new", item["name"].lower()))
