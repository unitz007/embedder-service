"""
File-hash tracking for incremental indexing.

Provides SHA-256 content hashing of source files and a lightweight JSON-based
cache so that only files whose content has actually changed are re-processed
on subsequent indexing runs.
"""

import hashlib
import json
from pathlib import Path


def compute_file_hash(filepath: str) -> str:
    """Return the SHA-256 hex digest of the file at *filepath*.

    Args:
        filepath: Absolute or relative path to the file.

    Returns:
        A 64-character lowercase hex string.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hashes(cache_path: str) -> dict[str, str]:
    """Load a previously saved ``{relative_path: hash}`` mapping from *cache_path*.

    If the file does not exist or contains invalid JSON, an empty dict is
    returned so callers can treat a missing / corrupt cache the same as a
    first run.

    Args:
        cache_path: Path to the JSON cache file.

    Returns:
        Dictionary mapping relative file paths to their SHA-256 hashes.
    """
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
        return {}
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}


def save_hashes(cache_path: str, hashes: dict[str, str]) -> None:
    """Persist *hashes* to *cache_path* as pretty-printed JSON.

    Parent directories are created automatically if they do not exist.

    Args:
        cache_path: Destination path for the JSON file.
        hashes:     Mapping of relative file paths to SHA-256 hex digests.
    """
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_changed_files(
    repo_path: str,
    all_files: list[str],
    cache_path: str,
) -> tuple[list[str], list[str]]:
    """Determine which files have changed compared to a cached hash map.

    For every entry in *all_files*, the current content hash is computed and
    compared against the value stored in *cache_path*.  Files are classified
    as either **changed** (hash differs or file is new) or **unchanged**
    (hash matches).  Files that exist in the cache but are no longer present
    in *all_files* are simply removed from the updated cache.

    After comparison the on-disk cache at *cache_path* is updated to reflect
    the current state.

    Args:
        repo_path:  Root directory of the repository (used to resolve
                    relative paths to absolute file system paths).
        all_files:  List of file paths **relative** to *repo_path*.
        cache_path: Path to the JSON cache file.

    Returns:
        A ``(changed_files, unchanged_files)`` tuple where each element is a
        list of paths relative to *repo_path*.
    """
    old_hashes: dict[str, str] = load_hashes(cache_path)
    new_hashes: dict[str, str] = {}
    changed: list[str] = []
    unchanged: list[str] = []

    repo = Path(repo_path)

    for rel in all_files:
        abs_path = str(repo / rel)
        try:
            current_hash = compute_file_hash(abs_path)
        except OSError:
            # File may have been removed between scan and hash; treat as changed
            # so it is not silently dropped.
            changed.append(rel)
            continue

        new_hashes[rel] = current_hash

        if rel not in old_hashes or old_hashes[rel] != current_hash:
            changed.append(rel)
        else:
            unchanged.append(rel)

    save_hashes(cache_path, new_hashes)
    return changed, unchanged