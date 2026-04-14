"""Tests for utils/file_scanner — scan_repository()."""

import os
import stat
import tempfile
import textwrap
import pytest

from utils.file_scanner import (
    BINARY_EXTENSIONS,
    GENERATED_FILE_PATTERNS,
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
    # Successful scans
    # ------------------------------------------------------------------

    def test_returns_source_files(self, tmp_path):
        """Normal source files are returned."""
        (tmp_path / "main.py").write_text("print('hi')")
        (tmp_path / "utils.py").write_text("def helper(): pass")

        results = scan_repository(str(tmp_path))
        assert len(results) == 2
        # All paths should be absolute
        assert all(os.path.isabs(f) for f in results)

    def test_empty_directory(self, tmp_path):
        """Empty directory returns empty list."""
        results = scan_repository(str(tmp_path))
        assert results == []


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

    def test_generated_file_patterns_is_tuple(self):
        assert isinstance(GENERATED_FILE_PATTERNS, tuple)
        assert all(isinstance(p, str) for p in GENERATED_FILE_PATTERNS)

    def test_max_file_size_bytes(self):
        assert MAX_FILE_SIZE_BYTES == 1_000_000
