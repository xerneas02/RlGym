from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def bootstrap_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def ensure_rocketsim_available() -> None:
    try:
        import RocketSim  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "RocketSim n'est pas installe. Lance `pip install rocketsim`, puis relance `pip install -r requirements.txt`."
        ) from exc


def ensure_rocketsim_arena_ready(project_root: Path, timeout_seconds: int = 8) -> None:
    probe = (
        "import RocketSim as rs; "
        "print('probe_before_arena', flush=True); "
        "arena = rs.Arena(rs.GameMode.SOCCAR); "
        "print('probe_after_arena', flush=True)"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            "RocketSim bloque pendant l'initialisation de l'arena. "
            "Cause la plus probable: les assets Rocket League dumpes par `RLArenaCollisionDumper` ne sont pas presents "
            "a la racine de `rocket_rl_bot/`. Place les fichiers d'assets dans ce dossier puis relance."
        ) from exc

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0 or "probe_after_arena" not in stdout:
        details = stderr.strip() or stdout.strip() or f"code={completed.returncode}"
        raise SystemExit(
            "RocketSim n'a pas reussi a initialiser l'arena. "
            "Verifie que les assets Rocket League dumpes par `RLArenaCollisionDumper` sont bien presents a la racine de `rocket_rl_bot/`. "
            f"Detail: {details}"
        )
