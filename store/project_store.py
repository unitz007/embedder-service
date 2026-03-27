"""
Simple JSON-file-backed store for Project entities.

Projects are first-class entities independent of any particular index or
GitHub repository.  They can optionally be linked to a GitHub repo via the
github_owner / github_repo / github_ref fields.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ProjectStore:
    """Thread-safe, JSON-file-backed store for Project records."""

    def __init__(self, data_dir: str) -> None:
        self._path = Path(data_dir) / "projects.json"
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, dict]:
        if not self._path.exists():
            return {}
        try:
            with open(self._path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, data: Dict[str, dict]) -> None:
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        namespace: str,
        description: Optional[str] = None,
        github_owner: Optional[str] = None,
        github_repo: Optional[str] = None,
        github_ref: Optional[str] = None,
    ) -> dict:
        with self._lock:
            data = self._load()
            project_id = uuid4().hex
            now = _now_iso()
            record = {
                "id": project_id,
                "name": name,
                "namespace": namespace,
                "description": description,
                "github_owner": github_owner,
                "github_repo": github_repo,
                "github_ref": github_ref,
                "created_at": now,
                "updated_at": now,
            }
            data[project_id] = record
            self._save(data)
            return record

    def get(self, project_id: str) -> Optional[dict]:
        with self._lock:
            return self._load().get(project_id)

    def list_all(self, namespace: Optional[str] = None) -> List[dict]:
        with self._lock:
            projects = list(self._load().values())
        if namespace is not None:
            projects = [p for p in projects if p.get("namespace") == namespace]
        return sorted(projects, key=lambda p: p["created_at"], reverse=True)

    def update(self, project_id: str, **fields) -> Optional[dict]:
        allowed = {"name", "description", "namespace", "github_owner", "github_repo", "github_ref"}
        with self._lock:
            data = self._load()
            record = data.get(project_id)
            if record is None:
                return None
            for key, value in fields.items():
                if key in allowed:
                    record[key] = value
            record["updated_at"] = _now_iso()
            data[project_id] = record
            self._save(data)
            return record

    def delete(self, project_id: str) -> bool:
        with self._lock:
            data = self._load()
            if project_id not in data:
                return False
            del data[project_id]
            self._save(data)
            return True
