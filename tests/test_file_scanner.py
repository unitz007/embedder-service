"""Tests for utils/file_scanner.py

Covers:
- Category frozen-sets contain expected extensions
- Default skip patterns cover common tool directories
- get_extension returns correct lowercase extensions
- matches_pattern / matches_any_pattern glob matching
- is_test_file heuristic classification
- should_skip for both filename and directory component matches
- classify_file priority ordering and edge cases
- scan_directory filtering by extension, recursive / non-recursive mode,
  and skip-directory exclusion
"""

import os
import tempfile
from pathlib import Path

import pytest

from utils.file_scanner import (
    CODE_EXTENSIONS,
    CONFIG_EXTENSIONS,
    DATA_EXTENSIONS,
    DEFAULT_SKIP_DIRS,
    DOCUMENT_EXTENSIONS,
    SKIP_PATTERNS,
    TEST_EXTENSIONS,
    TEST_PATTERNS,
    classify_file,
    get_extension,
    is_test_file,
    matches_any_pattern,
    matches_pattern,
    scan_directory,
    should_skip,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Exported constants
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Verify that exported constant sets contain the expected members."""

    def test_code_extensions_include_python_and_js(self):
        assert ".py" in CODE_EXTENSIONS
        assert ".js" in CODE_EXTENSIONS
        assert ".ts" in CODE_EXTENSIONS

    def test_code_extensions_include_compiled_languages(self):
        assert ".go" in CODE_EXTENSIONS
        assert ".rs" in CODE_EXTENSIONS
        assert ".java" in CODE_EXTENSIONS
        assert ".c" in CODE_EXTENSIONS
        assert ".cpp" in CODE_EXTENSIONS

    def test_code_extensions_are_frozen(self):
        with pytest.raises(AttributeError):
            CODE_EXTENSIONS.add(".xyz")  # type: ignore[attr-defined]

    def test_test_extensions(self):
        assert ".test.ts" in TEST_EXTENSIONS
        assert ".spec.ts" in TEST_EXTENSIONS

    def test_data_extensions(self):
        assert ".json" in DATA_EXTENSIONS
        assert ".yaml" in DATA_EXTENSIONS
        assert ".csv" in DATA_EXTENSIONS

    def test_config_extensions(self):
        assert ".cfg" in CONFIG_EXTENSIONS
        assert ".toml" in CONFIG_EXTENSIONS
        assert ".yaml" in CONFIG_EXTENSIONS

    def test_document_extensions(self):
        assert ".md" in DOCUMENT_EXTENSIONS
        assert ".rst" in DOCUMENT_EXTENSIONS
        assert ".txt" in DOCUMENT_EXTENSIONS

    def test_skip_patterns_include_common_dirs(self):
        assert ".git" in SKIP_PATTERNS
        assert "__pycache__" in SKIP_PATTERNS
        assert "node_modules" in SKIP_PATTERNS
        assert ".venv" in SKIP_PATTERNS

    def test_default_skip_dirs_subset_of_skip_patterns(self):
        assert DEFAULT_SKIP_DIRS.issubset(SKIP_PATTERNS)


# ═══════════════════════════════════════════════════════════════════════════
# 2. get_extension
# ═══════════════════════════════════════════════════════════════════════════


class TestGetExtension:
    """Tests for the get_extension helper."""

    def test_standard_extension(self):
        assert get_extension("src/main.py") == ".py"

    def test_nested_path(self):
        assert get_extension("a/b/c/d.go") == ".go"

    def test_no_extension(self):
        assert get_extension("Makefile") == ""
        assert get_extension("Dockerfile") == ""

    def test_double_extension_returns_last(self):
        """Only the last suffix is returned (standard pathlib behaviour)."""
        assert get_extension("archive.tar.gz") == ".gz"
        assert get_extension("app.test.js") == ".js"

    def test_case_insensitive(self):
        assert get_extension("README.MD") == ".md"
        assert get_extension("script.PY") == ".py"

    def test_dotfile_no_extension(self):
        assert get_extension(".gitignore") == ""
        assert get_extension(".env") == ""

    def test_empty_string(self):
        assert get_extension("") == ""


# ═══════════════════════════════════════════════════════════════════════════
# 3. matches_pattern / matches_any_pattern
# ═══════════════════════════════════════════════════════════════════════════


class TestMatchesPattern:
    """Tests for glob-style pattern matching."""

    def test_star_py(self):
        assert matches_pattern("main.py", "*.py") is True
        assert matches_pattern("utils/scanner.py", "*.py") is True

    def test_wrong_extension(self):
        assert matches_pattern("main.go", "*.py") is False

    def test_prefix_pattern(self):
        assert matches_pattern("test_utils.py", "test_*.py") is True
        assert matches_pattern("utils.py", "test_*.py") is False

    def test_case_insensitive(self):
        assert matches_pattern("MAIN.PY", "*.py") is True
        assert matches_pattern("Test_Utils.PY", "test_*.py") is True

    def test_question_mark(self):
        assert matches_pattern("f1.py", "f?.py") is True
        assert matches_pattern("file.py", "f?.py") is False

    def test_only_filename_matched(self):
        """Directory components must not participate in the match."""
        assert matches_pattern("src/test_main.py", "test_*.py") is True

    def test_matches_any_pattern_positive(self):
        assert matches_any_pattern("app.test.ts", ["*.spec.ts", "*.test.ts"]) is True

    def test_matches_any_pattern_negative(self):
        assert matches_any_pattern("app.ts", ["*.spec.ts", "*.test.ts"]) is False

    def test_matches_any_pattern_empty_list(self):
        assert matches_any_pattern("anything.py", []) is False


# ═══════════════════════════════════════════════════════════════════════════
# 4. is_test_file
# ═══════════════════════════════════════════════════════════════════════════


class TestIsTestFile:
    """Tests for the test-file heuristic."""

    @pytest.mark.parametrize("path", [
        "utils/file_scanner.test.ts",
        "component.spec.tsx",
        "api.test.js",
        "widget.spec.jsx",
        "utils_test.py",
        "scanner_test.go",
        "lib_test.rs",
        "test_helpers.py",
        "test_feature_test_test.py",  # double _test_ still matches
    ])
    def test_positive(self, path: str):
        assert is_test_file(path) is True

    @pytest.mark.parametrize("path", [
        "src/main.py",
        "app.ts",
        "package.json",
        "README.md",
        "testing_utils.py",  # has "test" but not at boundary
        "attest.py",
        "contest.py",
    ])
    def test_negative(self, path: str):
        assert is_test_file(path) is False

    def test_case_insensitive(self):
        assert is_test_file("MAIN.TEST.TS") is True
        assert is_test_file("TEST_HELPERS.PY") is True


# ═══════════════════════════════════════════════════════════════════════════
# 5. should_skip
# ═══════════════════════════════════════════════════════════════════════════


class TestShouldSkip:
    """Tests for the built-in skip-pattern logic."""

    @pytest.mark.parametrize("path", [
        ".git/HEAD",
        ".git/objects/pack/abc",
        "__pycache__/module.cpython-39.pyc",
        "node_modules/lodash/index.js",
        ".venv/lib/python3.9/site.py",
        "venv/lib/python3.9/site.py",
        "build/lib/lib.a",
        "dist/package.tar.gz",
        ".pytest_cache/v/cache/step",
        ".mypy_cache/3.9/mod.json",
    ])
    def test_should_skip_directory_components(self, path: str):
        assert should_skip(path) is True

    @pytest.mark.parametrize("path", [
        "src/main.py",
        "tests/test_app.py",
        "README.md",
        "app/config.py",
        "lib/utils/scanner_test.py",
    ])
    def test_should_not_skip(self, path: str):
        assert should_skip(path) is False

    def test_skip_by_filename(self):
        """Patterns starting with '*' match against the filename."""
        assert should_skip("coverage/lcov.info") is True  # directory "coverage"
        assert should_skip("data/lcov.info") is False


# ═══════════════════════════════════════════════════════════════════════════
# 6. classify_file
# ═══════════════════════════════════════════════════════════════════════════


class TestClassifyFile:
    """Tests for file classification with priority ordering."""

    def test_test_file_has_highest_priority(self):
        """A test file must be classified as 'test' even if its extension
        is in CODE_EXTENSIONS."""
        assert classify_file("test_utils.py") == "test"
        assert classify_file("scanner_test.go") == "test"
        assert classify_file("app.test.ts") == "test"

    def test_code_classification(self):
        assert classify_file("main.py") == "code"
        assert classify_file("index.ts") == "code"
        assert classify_file("lib.rs") == "code"

    def test_config_classification(self):
        assert classify_file("setup.cfg") == "config"
        assert classify_file("pyproject.toml") == "config"

    def test_data_classification(self):
        assert classify_file("data.json") == "data"
        assert classify_file("values.yaml") == "data"

    def test_document_classification(self):
        assert classify_file("README.md") == "document"
        assert classify_file("CHANGELOG.rst") == "document"

    def test_other_classification(self):
        assert classify_file("photo.png") == "other"
        assert classify_file("song.mp3") == "other"
        assert classify_file("video.mp4") == "other"
        assert classify_file("archive.zip") == "other"

    def test_unknown_extension_is_other(self):
        assert classify_file("file.unknownext") == "other"
        assert classify_file("Makefile") == "other"

    def test_overlap_config_and_data_yaml(self):
        """YAML files appear in both CONFIG_EXTENSIONS and DATA_EXTENSIONS.
        classify_file should respect its internal priority order."""
        result = classify_file("settings.yaml")
        assert result in ("config", "data")

    def test_case_insensitive(self):
        assert classify_file("SCRIPT.PY") == "code"
        assert classify_file("README.MD") == "document"


# ═══════════════════════════════════════════════════════════════════════════
# 7. scan_directory
# ═══════════════════════════════════════════════════════════════════════════


class TestScanDirectory:
    """Integration tests for directory scanning using temporary trees."""

    @staticmethod
    def _create_tree(root: str, files: list[str]) -> None:
        """Create an arbitrary set of files and directories under *root*."""
        for rel in files:
            full = os.path.join(root, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            Path(full).write_text("placeholder", encoding="utf-8")

    def test_basic_scan(self, tmp_path: Path):
        """All files are found by default."""
        self._create_tree(str(tmp_path), ["a.py", "b.js", "c.md"])
        result = scan_directory(str(tmp_path))
        assert sorted(result) == sorted(["a.py", "b.js", "c.md"])

    def test_sorted_output(self, tmp_path: Path):
        self._create_tree(str(tmp_path), ["z.py", "a.py", "m.py"])
        result = scan_directory(str(tmp_path))
        assert result == ["a.py", "m.py", "z.py"]

    def test_filter_by_extension(self, tmp_path: Path):
        self._create_tree(str(tmp_path), ["a.py", "b.js", "c.py", "d.md"])
        result = scan_directory(str(tmp_path), extensions=[".py"])
        assert result == ["a.py", "c.py"]

    def test_filter_multiple_extensions(self, tmp_path: Path):
        self._create_tree(str(tmp_path), ["a.py", "b.js", "c.md", "d.ts"])
        result = scan_directory(str(tmp_path), extensions=[".py", ".ts"])
        assert result == ["a.py", "d.ts"]

    def test_skip_default_dirs(self, tmp_path: Path):
        """Directories in DEFAULT_SKIP_DIRS must be excluded."""
        self._create_tree(str(tmp_path), [
            "src/main.py",
            "node_modules/pkg/index.js",
            "__pycache__/mod.pyc",
            "tests/test.py",
        ])
        result = scan_directory(str(tmp_path))
        assert "node_modules/pkg/index.js" not in result
        assert "__pycache__/mod.pyc" not in result
        assert "src/main.py" in result
        assert "tests/test.py" in result

    def test_custom_skip_dirs(self, tmp_path: Path):
        self._create_tree(str(tmp_path), [
            "src/main.py",
            "vendor/lib.go",
            "tests/test.py",
        ])
        result = scan_directory(str(tmp_path), skip_dirs=["vendor"])
        assert "vendor/lib.go" not in result
        assert "src/main.py" in result

    def test_non_recursive(self, tmp_path: Path):
        self._create_tree(str(tmp_path), [
            "a.py",
            "sub/b.py",
            "sub/deep/c.py",
        ])
        result = scan_directory(str(tmp_path), recursive=False)
        assert result == ["a.py"]

    def test_recursive_by_default(self, tmp_path: Path):
        self._create_tree(str(tmp_path), [
            "a.py",
            "sub/b.py",
            "sub/deep/c.py",
        ])
        result = scan_directory(str(tmp_path))
        assert "sub/b.py" in result
        assert os.path.join("sub", "deep", "c.py") in result

    def test_empty_directory(self, tmp_path: Path):
        result = scan_directory(str(tmp_path))
        assert result == []

    def test_hidden_files_included(self):
        """Dot-files that are not skip-pattern dirs should be included."""
        with tempfile.TemporaryDirectory() as tmp:
            self._create_tree(tmp, [".env", ".gitignore", "src/main.py"])
            # .gitignore has no skip-pattern entry as a filename; .env is
            # only in DATA_EXTENSIONS, not SKIP_PATTERNS
            result = scan_directory(tmp)
            # .env and .gitignore should both be present (they are not in
            # SKIP_PATTERNS as file names)
            assert ".env" in result
            assert ".gitignore" in result

    def test_nonexistent_root_raises(self):
        with pytest.raises(OSError):
            scan_directory("/nonexistent/path/that/does/not/exist")

    def test_extension_filter_case_insensitive(self, tmp_path: Path):
        """Extensions parameter should be compared case-insensitively."""
        self._create_tree(str(tmp_path), ["a.PY", "b.py", "c.Py"])
        result = scan_directory(str(tmp_path), extensions=[".py"])
        assert len(result) == 3
