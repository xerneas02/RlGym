from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

BASE_URL = "https://ballchasing.com/api"
DEFAULT_OUT_DIR = Path("replays_ssl_like")
DEFAULT_STATE_FILE = "_download_state.json"
DEFAULT_INDEX_FILE = "_downloaded_ids.txt"
DEFAULT_MANIFEST_FILE = "_manifest.jsonl"
UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Telecharge des replays ballchasing en plusieurs pages, "
            "sans re-telecharger ceux deja presents localement."
        )
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("BALLCHASING_API_KEY"),
        help="Cle API ballchasing. Par defaut, lit BALLCHASING_API_KEY.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Dossier de sortie. Par defaut: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=20_000,
        help=(
            "Nombre total de replays a posseder dans le dossier a la fin. "
            "Par defaut: 20000."
        ),
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="Limite le nombre de nouveaux replays a telecharger sur ce run.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="Taille de page demandee a l'API. Ballchasing accepte 1 a 200.",
    )
    parser.add_argument(
        "--playlist",
        default="ranked-duels",
        help="Playlist a filtrer. Exemple: ranked-duels",
    )
    parser.add_argument(
        "--min-rank",
        default="grand-champion",
        help="Rang minimum ballchasing. Exemple: grand-champion",
    )
    parser.add_argument(
        "--pro",
        action="store_true",
        help="Ne garde que les replays contenant au moins un joueur pro.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("replay-date", "upload-date"),
        default="replay-date",
        help="Champ de tri de l'API.",
    )
    parser.add_argument(
        "--sort-dir",
        choices=("asc", "desc"),
        default="desc",
        help="Sens de tri de l'API.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Ne demande que les replays ajoutes depuis le dernier run, "
            "avec une petite fenetre de recouvrement pour eviter les oublis."
        ),
    )
    parser.add_argument(
        "--overlap-hours",
        type=float,
        default=24.0,
        help="Fenetre de recouvrement appliquee en mode incremental. Par defaut: 24h.",
    )
    parser.add_argument(
        "--page-delay",
        type=float,
        default=0.4,
        help="Pause entre deux requetes de liste. Par defaut: 0.4s.",
    )
    parser.add_argument(
        "--download-delay",
        type=float,
        default=2.0,
        help="Pause entre deux telechargements. Par defaut: 2.0s.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Nombre maximum de tentatives sur 429/5xx. Par defaut: 5.",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            "Cle API manquante. Definis BALLCHASING_API_KEY ou passe --api-key."
        )
    if not 1 <= args.page_size <= 200:
        parser.error("--page-size doit etre entre 1 et 200.")
    if args.target_total is not None and args.target_total < 0:
        parser.error("--target-total doit etre >= 0.")
    if args.max_new is not None and args.max_new < 0:
        parser.error("--max-new doit etre >= 0.")
    return args


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"[WARN] Etat invalide ignore: {path}")
        return {}


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


def load_known_ids(index_path: Path, out_dir: Path) -> set[str]:
    known_ids: set[str] = set()

    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            replay_id = line.strip()
            if replay_id:
                known_ids.add(replay_id)

    imported_from_files: list[str] = []
    if out_dir.exists():
        for replay_path in out_dir.glob("*.replay"):
            match = UUID_RE.search(replay_path.stem)
            if not match:
                continue
            replay_id = match.group(0).lower()
            if replay_id not in known_ids:
                imported_from_files.append(replay_id)
            known_ids.add(replay_id)

    if imported_from_files:
        with index_path.open("a", encoding="utf-8", newline="\n") as handle:
            for replay_id in imported_from_files:
                handle.write(f"{replay_id}\n")
        print(
            f"[INFO] {len(imported_from_files)} replay(s) existants importes dans l'index."
        )

    return known_ids


def append_index(index_path: Path, replay_id: str) -> None:
    with index_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{replay_id}\n")


def append_manifest(manifest_path: Path, replay: dict[str, Any], file_name: str) -> None:
    record = {
        "id": replay["id"],
        "file_name": file_name,
        "replay_title": replay.get("replay_title"),
        "playlist": replay.get("playlist_id"),
        "replay_date": replay.get("date"),
        "created": replay.get("created"),
        "downloaded_at": utc_now_iso(),
    }
    with manifest_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=True))
        handle.write("\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def shift_iso8601(value: str, hours: float) -> str:
    parsed = parse_iso8601(value)
    if parsed is None:
        return value
    shifted = parsed - timedelta(hours=hours)
    return shifted.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def update_latest_timestamp(
    state: dict[str, Any], state_key: str, candidate_value: str | None
) -> None:
    current = parse_iso8601(state.get(state_key))
    candidate = parse_iso8601(candidate_value)
    if candidate is None:
        return
    if current is None or candidate > current:
        state[state_key] = candidate_value


def normalize_next_url(next_url: str | None) -> str | None:
    if not next_url:
        return None
    if next_url.startswith("http://") or next_url.startswith("https://"):
        return next_url
    return urljoin(f"{BASE_URL}/", next_url.lstrip("/"))


def build_initial_params(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {
        "playlist": args.playlist,
        "count": args.page_size,
        "sort-by": args.sort_by,
        "sort-dir": args.sort_dir,
    }

    if args.min_rank:
        params["min-rank"] = args.min_rank
    if args.pro:
        params["pro"] = "true"
    if args.incremental and state.get("latest_created"):
        params["created-after"] = shift_iso8601(
            state["latest_created"], args.overlap_hours
        )

    return params


def compute_retry_wait_seconds(
    response: requests.Response,
    attempt: int,
    *,
    min_seconds: float,
) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            wait_seconds = float(retry_after)
        except ValueError:
            try:
                retry_dt = parsedate_to_datetime(retry_after)
                if retry_dt.tzinfo is None:
                    retry_dt = retry_dt.replace(tzinfo=timezone.utc)
                wait_seconds = (retry_dt - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                wait_seconds = 0.0
        if wait_seconds > 0:
            return max(min_seconds, wait_seconds)
    return max(min_seconds, 2.0**attempt)


def request_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    stream: bool = False,
    timeout: int = 60,
    max_attempts: int = 5,
) -> requests.Response:
    is_file_download = url.endswith("/file")
    min_wait = 15.0 if is_file_download else 3.0
    last_response: requests.Response | None = None

    for attempt in range(1, max_attempts + 1):
        response = session.get(url, params=params, stream=stream, timeout=timeout)
        last_response = response
        if response.status_code == 429:
            if attempt >= max_attempts:
                return response
            wait_seconds = compute_retry_wait_seconds(
                response,
                attempt,
                min_seconds=min_wait,
            )
            print(
                f"[RATE LIMIT] 429 sur {url}. Attente {wait_seconds:.1f}s avant retry "
                f"({attempt}/{max_attempts})."
            )
            response.close()
            time.sleep(wait_seconds)
            continue
        if 500 <= response.status_code < 600 and attempt < max_attempts:
            wait_seconds = max(min_wait, 2.0**attempt)
            print(
                f"[SERVER] {response.status_code} sur {url}. Attente {wait_seconds:.1f}s "
                f"avant retry ({attempt}/{max_attempts})."
            )
            response.close()
            time.sleep(wait_seconds)
            continue
        return response

    if last_response is None:
        raise RuntimeError(f"Aucune reponse recue pour {url}")
    return last_response


def fetch_replay_page(
    session: requests.Session,
    *,
    next_url: str | None,
    params: dict[str, Any],
    max_attempts: int,
) -> tuple[list[dict[str, Any]], str | None]:
    if next_url:
        response = request_with_retry(
            session, next_url, stream=False, timeout=30, max_attempts=max_attempts
        )
    else:
        response = request_with_retry(
            session,
            f"{BASE_URL}/replays",
            params=params,
            stream=False,
            timeout=30,
            max_attempts=max_attempts,
        )

    response.raise_for_status()
    payload = response.json()
    return payload.get("list", []), normalize_next_url(payload.get("next"))


def download_replay(
    session: requests.Session,
    replay_id: str,
    out_dir: Path,
    max_attempts: int,
) -> tuple[bool, str]:
    url = f"{BASE_URL}/replays/{replay_id}/file"
    try:
        response = request_with_retry(
            session, url, stream=True, timeout=90, max_attempts=max_attempts
        )
    except requests.RequestException as exc:
        return False, f"Erreur reseau sur /file: {exc}"

    if response.status_code == 404:
        response.close()
        return False, "Replay introuvable ou non telechargeable"
    if response.status_code == 403:
        response.close()
        return False, "Replay prive ou acces refuse"
    if response.status_code == 429:
        response.close()
        return False, "Rate limit persistant sur /file, replay saute pour ce run"
    if 500 <= response.status_code < 600:
        status_code = response.status_code
        response.close()
        return False, f"Erreur serveur persistante ({status_code}) sur /file, replay saute pour ce run"

    response.raise_for_status()

    out_path = out_dir / f"{replay_id}.replay"
    tmp_path = out_dir / f"{replay_id}.part"

    with tmp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 128):
            if chunk:
                handle.write(chunk)
    response.close()

    tmp_path.replace(out_path)
    return True, out_path.name


def compute_run_limit(args: argparse.Namespace, already_known: int) -> int | None:
    limits = []
    if args.max_new is not None:
        limits.append(args.max_new)
    if args.target_total is not None:
        limits.append(max(0, args.target_total - already_known))
    if not limits:
        return None
    return min(limits)


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / DEFAULT_STATE_FILE
    index_path = out_dir / DEFAULT_INDEX_FILE
    manifest_path = out_dir / DEFAULT_MANIFEST_FILE

    state = read_state(state_path)
    known_ids = load_known_ids(index_path, out_dir)
    run_limit = compute_run_limit(args, len(known_ids))

    if run_limit == 0:
        print(
            f"[DONE] Rien a faire: {len(known_ids)} replay(s) deja connus, "
            f"objectif={args.target_total}."
        )
        return

    headers = {"Authorization": args.api_key}
    session = requests.Session()
    session.headers.update(headers)

    initial_params = build_initial_params(args, state)
    if args.incremental and "created-after" in initial_params:
        print(
            "[INFO] Mode incremental active, filtre created-after="
            f"{initial_params['created-after']}"
        )

    downloaded_now = 0
    skipped_known = 0
    skipped_unavailable = 0
    page_count = 0
    next_url: str | None = None

    print(f"[INFO] Replays deja connus: {len(known_ids)}")
    print(
        f"[INFO] Limite de nouveaux telechargements pour ce run: "
        f"{run_limit if run_limit is not None else 'illimitee'}"
    )

    while True:
        if run_limit is not None and downloaded_now >= run_limit:
            break

        page_count += 1
        replays, next_url = fetch_replay_page(
            session,
            next_url=next_url,
            params=initial_params,
            max_attempts=args.max_attempts,
        )

        if not replays:
            print("[INFO] Plus aucun replay retourne par l'API.")
            break

        print(f"[PAGE {page_count}] {len(replays)} replay(s) recus.")

        for replay in replays:
            replay_id = replay["id"].lower()
            replay_title = replay.get("replay_title") or replay_id

            if replay_id in known_ids:
                skipped_known += 1
                continue

            print(
                f"[GET] {replay_title} ({replay_id}) "
                f"[nouveau {downloaded_now + 1}]"
            )

            ok, message = download_replay(
                session=session,
                replay_id=replay_id,
                out_dir=out_dir,
                max_attempts=args.max_attempts,
            )
            if not ok:
                skipped_unavailable += 1
                print(f"[SKIP] {replay_id}: {message}")
                time.sleep(args.download_delay)
                continue

            known_ids.add(replay_id)
            append_index(index_path, replay_id)
            append_manifest(manifest_path, replay, message)
            update_latest_timestamp(state, "latest_created", replay.get("created"))
            update_latest_timestamp(state, "latest_replay_date", replay.get("date"))
            state["last_run_at"] = utc_now_iso()
            state["known_replay_count"] = len(known_ids)
            state["filters"] = {
                "playlist": args.playlist,
                "min-rank": args.min_rank,
                "pro": args.pro,
                "sort-by": args.sort_by,
                "sort-dir": args.sort_dir,
            }
            write_state(state_path, state)

            downloaded_now += 1
            print(f"[OK] {message}")

            if run_limit is not None and downloaded_now >= run_limit:
                break

            time.sleep(args.download_delay)

        if not next_url:
            print("[INFO] Fin de pagination: l'API n'a pas fourni de page suivante.")
            break

        if run_limit is not None and downloaded_now >= run_limit:
            break

        time.sleep(args.page_delay)

    print()
    print("[SUMMARY]")
    print(f"  Pages lues          : {page_count}")
    print(f"  Nouveaux replays    : {downloaded_now}")
    print(f"  Deja connus sautes  : {skipped_known}")
    print(f"  Indisponibles sautes: {skipped_unavailable}")
    print(f"  Total connu final   : {len(known_ids)}")
    print(f"  Dossier             : {out_dir}")


if __name__ == "__main__":
    main()

