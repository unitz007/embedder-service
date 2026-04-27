"""Tests for utils/file_scanner — scan_repository()."""

import os
import tempfile
import pytest

from utils.file_scanner import (
    BINARY_EXTENSIONS,
    DOTDIR_ALLOWLIST,
    GENERATED_FILE_PATTERNS,
    IGNORE_DIRS,
    MAX_FILE_SIZE_BYTES,
    scan_repository,
    _is_generated_file,
)


class TestScanRepository:
    """Tests for the scan_repository helper."""

    # ------------------------------------------------------------------
    # Binary / generated file filtering
    # ------------------------------------------------------------------

    def test_skips_binary_files(self, tmp_path):
        """Files with extensions in BINARY_EXTENSIONS are excluded."""
        (tmp_path / "photo.png").write_bytes(b"\x89PNG\r\n")
        (tmp_path / "readme.txt").write_text("hello")

        results = scan_repository(str(tmp_path))
        assert [os.path.basename(f) for f in results] == ["readme.txt"]

    def test_skips_wasm_files(self, tmp_path):
        """.wasm is a binary extension and must be excluded."""
        (tmp_path / "module.wasm").write_bytes(b"\x00asm")
        (tmp_path / "lib.js").write_text("export {};")

        results = scan_repository(str(tmp_path))
        basenames = [os.path.basename(f) for f in results]
        assert "module.wasm" not in basenames
        assert "lib.js" in basenames

    def test_skips_generated_lock_files(self, tmp_path):
        """Files matching GENERATED_FILE_PATTERNS (e.g. lock files) are skipped."""
        (tmp_path / "package-lock.json").write_text("{}")
        (tmp_path / "index.ts").write_text("console.log(1)")

        results = scan_repository(str(tmp_path))
        basenames = [os.path.basename(f) for f in results]
        assert "package-lock.json" not in basenames
        assert "index.ts" in basenames

    # ------------------------------------------------------------------
    # Directory filtering
    # ------------------------------------------------------------------

    def test_skips_node_modules(self, tmp_path):
        """Directories in IGNORE_DIRS (e.g. node_modules) are pruned."""
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "lodash.js").write_text("module.exports = {};")

        (tmp_path / "app.js").write_text("require('lodash')")

        results = scan_repository(str(tmp_path))
        basenames = [os.path.basename(f) for f in results]
        assert "lodash.js" not in basenames
        assert "app.js" in basenames

    def test_skips_hidden_dirs_by_default(self, tmp_path):
        """Hidden directories are skipped unless include_dotfiles is True."""
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.js").write_text("secret")

        results = scan_repository(str(tmp_path))
        assert results == []

        results = scan_repository(str(tmp_path), include_dotfiles=True)
        assert len(results) == 1

    def test_skips_hidden_files_by_default(self, tmp_path):
        """Files starting with '.' are skipped unless include_dotfiles is True."""
        (tmp_path / ".env").write_text("KEY=val")
        (tmp_path / "main.py").write_text("print('hello')")

        results = scan_repository(str(tmp_path))
        basenames = [os.path.basename(f) for f in results]
        assert ".env" not in basenames
        assert "main.py" in basenames

        results = scan_repository(str(tmp_path), include_dotfiles=True)
        basenames = [os.path.basename(f) for f in results]
        assert ".env" in basenames

    # ------------------------------------------------------------------
    # DOTDIR_ALLOWLIST
    # ------------------------------------------------------------------

    def test_dotdir_allowlist_github(self, tmp_path):
        """.github/ is indexed even when include_dotfiles=False."""
        gh = tmp_path / ".github"
        gh.mkdir()
        (gh / "workflows.yaml").write_text("name: CI")

        results = scan_repository(str(tmp_path), include_dotfiles=False)
        assert any(".github" in f for f in results)
        assert ".github/workflows.yaml" in results

    def test_dotdir_allowlist_vscode(self, tmp_path):
        """.vscode/ is indexed even when include_dotfiles=False."""
        vscode = tmp_path / ".vscode"
        vscode.mkdir()
        (vscode / "settings.json").write_text("{}")

        results = scan_repository(str(tmp_path), include_dotfiles=False)
        assert ".vscode/settings.json" in results

    def test_dotdir_allowlist_vscode_hidden_files(self, tmp_path):
        """Dotfiles inside .vscode/ are included even without include_dotfiles."""
        vscode = tmp_path / ".vscode"
        vscode.mkdir()
        (vscode / ".hidden_setting").write_text("data")

        results = scan_repository(str(tmp_path), include_dotfiles=False)
        assert ".vscode/.hidden_setting" in results

    # ------------------------------------------------------------------
    # File size filtering
    # ------------------------------------------------------------------

    def test_skips_oversized_files(self, tmp_path):
        """Files exceeding MAX_FILE_SIZE_BYTES are skipped."""
        big = tmp_path / "big.txt"
        big.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))

        small = tmp_path / "small.txt"
        small.write_text("hi")

        results = scan_repository(str(tmp_path))
        basenames = [os.path.basename(f) for f in results]
        assert "big.txt" not in basenames
        assert "small.txt" in basenames

    # ------------------------------------------------------------------
    # Extension and suffix filtering
    # ------------------------------------------------------------------

    def test_skips_minified_suffix(self, tmp_path):
        """Files ending with IGNORE_FILE_SUFFIXES like .min.js are skipped."""
        (tmp_path / "bundle.min.js").write_text("var a=1")
        (tmp_path / "app.js").write_text("const a = 1;")

        results = scan_repository(str(tmp_path))
        basenames = [os.path.basename(f) for f in results]
        assert "bundle.min.js" not in basenames
        assert "app.js" in basenames

    # ------------------------------------------------------------------
    # Return value shape
    # ------------------------------------------------------------------

    def test_returns_relative_paths(self, tmp_path):
        """scan_repository returns paths relative to repo_path."""
        (tmp_path / "main.py").write_text("print('hi')")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "utils.py").write_text("def helper(): pass")

        results = scan_repository(str(tmp_path))
        assert len(results) == 2
        # All paths should be relative
        assert all(not os.path.isabs(f) for f in results)
        assert "main.py" in results
        assert os.path.join("subdir", "utils.py") in results

    # ------------------------------------------------------------------
    # Successful scans
    # ------------------------------------------------------------------

    def test_returns_source_files(self, tmp_path):
        """Normal source files are returned."""
        (tmp_path / "main.py").write_text("print('hi')")
        (tmp_path / "utils.py").write_text("def helper(): pass")

        results = scan_repository(str(tmp_path))
        assert len(results) == 2

    def test_empty_directory(self, tmp_path):
        """Empty directory returns empty list."""
        results = scan_repository(str(tmp_path))
        assert results == []

    def test_vscode_not_in_ignore_dirs(self):
        """.vscode must not appear in IGNORE_DIRS so it can be allowlisted."""
        assert ".vscode" not in IGNORE_DIRS


# ------------------------------------------------------------------
# Unit tests for _is_generated_file
# ------------------------------------------------------------------

class TestIsGeneratedFile:
    """Tests for the _is_generated_file helper."""

    def test_exact_name_match(self):
        assert _is_generated_file("package-lock.json") is True
        assert _is_generated_file("yarn.lock") is True
        assert _is_generated_file("go.sum") is True

    def test_extension_match(self):
        assert _is_generated_file("hello.min.js") is True
        assert _is_generated_file("style.min.css") is True

    def test_not_generated(self):
        assert _is_generated_file("main.py") is False
        assert _is_generated_file("README.md") is False


# ------------------------------------------------------------------
# Constant type checks
# ------------------------------------------------------------------

class TestConstants:
    """Verify that constants match their required types and values."""

    def test_binary_extensions_is_frozenset(self):
        assert isinstance(BINARY_EXTENSIONS, frozenset)

    def test_binary_extensions_includes_wasm(self):
        assert ".wasm" in BINARY_EXTENSIONS

    def test_binary_extensions_has_at_least_30_entries(self):
        assert len(BINARY_EXTENSIONS) >= 30

    def test_generated_file_patterns_is_tuple(self):
        assert isinstance(GENERATED_FILE_PATTERNS, tuple)
        assert all(isinstance(p, str) for p in GENERATED_FILE_PATTERNS)

    def test_generated_file_patterns_has_at_least_10_entries(self):
        assert len(GENERATED_FILE_PATTERNS) >= 10

    def test_max_file_size_bytes(self):
        assert MAX_FILE_SIZE_BYTES == 1_000_000

    def test_dotdir_allowlist_includes_github_and_vscode(self):
        assert ".github" in DOTDIR_ALLOWLIST
        assert ".vscode" in DOTDIR_ALLOWLIST