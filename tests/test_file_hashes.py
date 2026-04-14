"""Tests for :pymod:`utils.file_hashes`."""

import json
import os
import textwrap
from pathlib import Path

import pytest

from utils.file_hashes import (
    compute_file_hash,
    get_changed_files,
    load_hashes,
    save_hashes,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_repo(tmp_path):
    """A small repository-like directory with a few text files."""
    (tmp_path / "a.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / "b.go").write_text("package main\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("# config\nx = 1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def cache_path(tmp_path):
    return str(tmp_path / "cache" / "hashes.json")


def _files(repo):
    """Return the fixture repo contents as a list of relative paths."""
    all_files = []
    for root, _dirs, files in os.walk(repo):
        for f in files:
            all_files.append(os.path.relpath(os.path.join(root, f), repo))
    return sorted(all_files)


# ---------------------------------------------------------------------------
# compute_file_hash
# ---------------------------------------------------------------------------

class TestComputeFileHash:
    def test_deterministic(self, tmp_repo):
        """Hashing the same file twice returns the same digest."""
        path = str(tmp_repo / "a.py")
        assert compute_file_hash(path) == compute_file_hash(path)

    def test_different_content(self, tmp_repo):
        """Different file contents produce different hashes."""
        h_a = compute_file_hash(str(tmp_repo / "a.py"))
        h_b = compute_file_hash(str(tmp_repo / "b.go"))
        assert h_a != h_b

    def test_known_sha256(self, tmp_path):
        """Matches an independently computed SHA-256."""
        p = tmp_path / "known.txt"
        p.write_bytes(b"abc")
        expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        assert compute_file_hash(str(p)) == expected

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compute_file_hash(str(tmp_path / "nope.txt"))


# ---------------------------------------------------------------------------
# load_hashes / save_hashes
# ---------------------------------------------------------------------------

class TestLoadSaveHashes:
    def test_round_trip(self, cache_path):
        data = {"a.py": "aa", "b.go": "bb"}
        save_hashes(cache_path, data)
        loaded = load_hashes(cache_path)
        assert loaded == data

    def test_load_missing_returns_empty(self, cache_path):
        assert load_hashes(cache_path) == {}

    def test_save_creates_parent_dirs(self, tmp_path):
        nested = str(tmp_path / "x" / "y" / "hashes.json")
        save_hashes(nested, {"a": "b"})
        assert load_hashes(nested) == {"a": "b"}

    def test_corrupt_json_returns_empty(self, cache_path):
        Path(cache_path).write_text("NOT JSON{{{", encoding="utf-8")
        assert load_hashes(cache_path) == {}

    def test_non_dict_json_returns_empty(self, cache_path):
        Path(cache_path).write_text("[1,2,3]", encoding="utf-8")
        assert load_hashes(cache_path) == {}

    def test_saves_sorted(self, cache_path):
        data = {"z.py": "zz", "a.py": "aa"}
        save_hashes(cache_path, data)
        text = Path(cache_path).read_text(encoding="utf-8")
        obj = json.loads(text)
        assert list(obj.keys()) == ["a.py", "z.py"]

    def test_filters_non_string_values(self, cache_path):
        Path(cache_path).write_text(json.dumps({"good": "hash", "bad": 123}), encoding="utf-8")
        loaded = load_hashes(cache_path)
        assert loaded == {"good": "hash"}

    def test_filters_non_string_keys(self, cache_path):
        Path(cache_path).write_text(json.dumps({1: "hash"}), encoding="utf-8")
        loaded = load_hashes(cache_path)
        assert loaded == {}

    def test_trailing_newline(self, cache_path):
        save_hashes(cache_path, {"a": "b"})
        text = Path(cache_path).read_text(encoding="utf-8")
        assert text.endswith("\n")


# ---------------------------------------------------------------------------
# get_changed_files
# ---------------------------------------------------------------------------

class TestGetChangedFiles:
    def test_first_run_all_changed(self, tmp_repo, cache_path):
        """On the very first run (no cache), every file is reported as changed."""
        files = _files(tmp_repo)
        changed, unchanged = get_changed_files(str(tmp_repo), files, cache_path)
        assert sorted(changed) == files
        assert unchanged == []

    def test_second_run_no_changes(self, tmp_repo, cache_path):
        """If nothing changes between calls, no files are reported as changed."""
        files = _files(tmp_repo)
        get_changed_files(str(tmp_repo), files, cache_path)  # first run
        changed, unchanged = get_changed_files(str(tmp_repo), files, cache_path)  # second
        assert changed == []
        assert sorted(unchanged) == files

    def test_modified_file_detected(self, tmp_repo, cache_path):
        """Editing a file marks it as changed on the next run."""
        files = _files(tmp_repo)
        get_changed_files(str(tmp_repo), files, cache_path)

        (tmp_repo / "a.py").write_text("print('changed')\n", encoding="utf-8")

        changed, unchanged = get_changed_files(str(tmp_repo), files, cache_path)
        assert "a.py" in changed
        assert "b.go" not in changed
        assert "b.go" in unchanged

    def test_new_file_detected(self, tmp_repo, cache_path):
        """Adding a file marks it as changed on the next run."""
        files = _files(tmp_repo)
        get_changed_files(str(tmp_repo), files, cache_path)

        (tmp_repo / "new.txt").write_text("hello\n", encoding="utf-8")
        new_files = sorted(files + ["new.txt"])

        changed, _ = get_changed_files(str(tmp_repo), new_files, cache_path)
        assert "new.txt" in changed

    def test_removed_file_not_in_results(self, tmp_repo, cache_path):
        """Files removed from the all_files list are silently dropped."""
        files = _files(tmp_repo)
        get_changed_files(str(tmp_repo), files, cache_path)

        reduced = [f for f in files if f != "a.py"]
        changed, unchanged = get_changed_files(str(tmp_repo), reduced, cache_path)
        assert "a.py" not in changed
        assert "a.py" not in unchanged

    def test_cache_updated_after_each_call(self, tmp_repo, cache_path):
        """The cache file on disk reflects the latest hashes after every call."""
        files = _files(tmp_repo)
        get_changed_files(str(tmp_repo), files, cache_path)

        (tmp_repo / "a.py").write_text("modified\n", encoding="utf-8")
        get_changed_files(str(tmp_repo), files, cache_path)

        on_disk = load_hashes(cache_path)
        assert on_disk["a.py"] == compute_file_hash(str(tmp_repo / "a.py"))

    def test_multiple_changes(self, tmp_repo, cache_path):
        """Several simultaneous changes are all detected."""
        files = _files(tmp_repo)
        get_changed_files(str(tmp_repo), files, cache_path)

        (tmp_repo / "a.py").write_text("v2\n", encoding="utf-8")
        (tmp_repo / "b.go").write_text("package lib\n", encoding="utf-8")

        changed, unchanged = get_changed_files(str(tmp_repo), files, cache_path)
        assert set(changed) == {"a.py", "b.go"}
        assert "sub/c.py" in unchanged

    def test_all_files_deleted(self, tmp_repo, cache_path):
        """Passing an empty list returns no changed files."""
        files = _files(tmp_repo)
        get_changed_files(str(tmp_repo), files, cache_path)

        changed, unchanged = get_changed_files(str(tmp_repo), [], cache_path)
        assert changed == []
        assert unchanged == []

    def test_file_deleted_between_scan_and_hash(self, tmp_repo, cache_path):
        """A file removed after the file list was collected is treated as changed."""
        files = _files(tmp_repo)
        # Simulate deletion between scan and hash by providing a non-existent path
        nonexistent_files = files + ["phantom.py"]
        changed, unchanged = get_changed_files(str(tmp_repo), nonexistent_files, cache_path)
        assert "phantom.py" in changed
