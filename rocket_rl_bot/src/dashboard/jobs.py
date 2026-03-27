from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class JobRecord:
    job_id: str
    kind: str
    status: str
    title: str
    created_at: float
    updated_at: float
    payload_path: str
    log_path: str
    pid: int | None = None
    return_code: int | None = None
    run_name_hint: str | None = None


class JobManager:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.runtime_dir = self.project_root / "runtime" / "dashboard_jobs"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, tuple[JobRecord, subprocess.Popen[Any] | None]] = {}

    def launch(self, kind: str, title: str, payload: dict[str, Any]) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        payload_path = self.runtime_dir / f"{job_id}.json"
        log_path = self.runtime_dir / f"{job_id}.log"
        payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        command = [
            sys.executable,
            str(self.project_root / "scripts" / "dashboard_job_runner.py"),
            str(kind),
            "--payload",
            str(payload_path),
        ]
        log_handle = log_path.open("w", encoding="utf-8")
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen(
            command,
            cwd=str(self.project_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        record = JobRecord(
            job_id=job_id,
            kind=str(kind),
            status="running",
            title=str(title),
            created_at=time.time(),
            updated_at=time.time(),
            payload_path=str(payload_path),
            log_path=str(log_path),
            pid=int(process.pid),
            run_name_hint=str(payload.get("run_name_prefix", "")) or None,
        )
        self.jobs[job_id] = (record, process)
        return record

    def refresh(self) -> None:
        for job_id, (record, process) in list(self.jobs.items()):
            if process is None:
                continue
            return_code = process.poll()
            record.updated_at = time.time()
            if return_code is None:
                continue
            record.return_code = int(return_code)
            record.status = "completed" if return_code == 0 else "failed"
            self.jobs[job_id] = (record, None)

    def list_jobs(self) -> list[dict[str, Any]]:
        self.refresh()
        records = [asdict(record) for record, _process in self.jobs.values()]
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

    def get_job(self, job_id: str) -> dict[str, Any]:
        self.refresh()
        if job_id not in self.jobs:
            raise KeyError(f"Unknown job: {job_id}")
        record, _process = self.jobs[job_id]
        return asdict(record)

    def read_log_tail(self, job_id: str, max_chars: int = 24000) -> str:
        if job_id not in self.jobs:
            raise KeyError(f"Unknown job: {job_id}")
        record, _process = self.jobs[job_id]
        path = Path(record.log_path)
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:]
