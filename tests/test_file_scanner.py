# tests/test_file_scanner.py
"""Unit tests for utils/file_scanner.py — repository scanning logic."""

import os
import tempfile

import pytest

from utils.file_scanner import scan_repository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tree(base: str, files: dict) -> str:
    """Create a directory tree.  *files* maps relative path → content."""
    for rel, content in files.items():
        full = os.path.join(base, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
    return base


# ---------------------------------------------------------------------------
# Core scanning
# ---------------------------------------------------------------------------

class TestScanRepository:
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = scan_repository(tmp)
            assert result == []

    def test_single_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "main.py")
            with open(path, "w") as f:
                f.write("print('hello')\n")
            result = scan_repository(tmp)
            assert len(result) == 1
            assert result[0].endswith("main.py")

    def test_nested_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_tree(tmp, {
                "src/a.py": "a",
                "src/lib/b.py": "b",
                "README.md": "# readme",
            })
            result = scan_repository(tmp)
            assert len(result) == 3

    def test_skips_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_tree(tmp, {
                ".git/config": "",
                "main.py": "main",
                ".git/HEAD": "",
            })
            result = scan_repository(tmp)
            assert all(".git" not in f for f in result)

    def test_skips_node_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_tree(tmp, {
                "node_modules/lodash/index.js": "",
                "main.js": "main",
            })
            result = scan_repository(tmp)
            assert all("node_modules" not in f for f in result)

    def test_skips_binary_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_tree(tmp, {
                "image.png": b"\x89PNG",
                "main.py": "main",
            })
            result = scan_repository(tmp)
            assert all(not f.endswith(".png") for f in result)

    def test_skips_oversized_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            big = os.path.join(tmp, "big.py")
            with open(big, "w") as f:
                f.write("x" * (600 * 1024))  # > 512KB
            small = os.path.join(tmp, "small.py")
            with open(small, "w") as f:
                f.write("x = 1\n")
            result = scan_repository(tmp)
            assert len(result) == 1
            assert result[0].endswith("small.py")

    def test_skips_pycache(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_tree(tmp, {
                "__pycache__/mod.pyc": "",
                "main.py": "main",
            })
            result = scan_repository(tmp)
            assert all("__pycache__" not in f for f in result)

    def test_skips_generated_go_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_tree(tmp, {
                "gen.pb.go": "generated",
                "main.go": "package main",
            })
            result = scan_repository(tmp)
            assert all(".pb.go" not in f for f in result)

    def test_skips_hidden_files_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_tree(tmp, {
                ".hidden": "hidden",
                "visible.py": "visible",
            })
            result = scan_repository(tmp)
            assert all(not os.path.basename(f).startswith(".") for f in result)

    def test_include_dotfiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_tree(tmp, {
                ".zshrc": "export PATH=...",
                ".git/config": "",
                "visible.py": "visible",
            })
            result = scan_repository(tmp, include_dotfiles=True)
            basenames = {os.path.basename(f) for f in result}
            assert ".zshrc" in basenames
            # .git should still be excluded
            assert not any(".git" in f for f in result)

    def test_skips_lock_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_tree(tmp, {
                "package-lock.json": "{}",
                "main.js": "main",
            })
            result = scan_repository(tmp)
            assert all(".lock" not in f for f in result)

    def test_skips_common_ignore_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_tree(tmp, {
                ".venv/lib/site.py": "",
                "venv/lib/site.py": "",
                "dist/main.js": "",
                "build/out.js": "",
                "main.py": "main",
            })
            result = scan_repository(tmp)
            for ignored in (".venv", "venv", "dist", "build"):
                assert all(ignored not in f for f in result)
