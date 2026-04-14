"""
File scanner module for classifying and filtering project files.

This module provides utilities for:
- Pattern matching: identifying files that match extension or path patterns
- Category classification: grouping files by type (code, config, data, etc.)
- File filtering: excluding files that match skip patterns
"""

import os
import re
from pathlib import Path
from typing import FrozenSet, Iterable, Optional


# ── Extension categories ────────────────────────────────────────────────────

CODE_EXTENSIONS: FrozenSet[str] = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".rb", ".java",
    ".kt", ".swift", ".c", ".cpp", ".h", ".hpp", ".cs", ".scala",
    ".php", ".lua", ".r", ".m", ".mm",
})

TEST_EXTENSIONS: FrozenSet[str] = frozenset({
    ".test.ts", ".test.tsx", ".test.js", ".test.jsx",
    ".spec.ts", ".spec.tsx", ".spec.js", ".spec.jsx",
})

# Extensions that indicate test files by the main extension + naming convention
# (actual detection handled by matches_pattern)
TEST_PATTERNS: tuple[str, ...] = (
    "*.test.ts", "*.test.tsx", "*.test.js", "*.test.jsx",
    "*.spec.ts", "*.spec.tsx", "*.spec.js", "*.spec.jsx",
    "*_test.py", "*_test.go", "*_test.rs",
    "test_*.py",
    "*_test.dart",
)

DATA_EXTENSIONS: FrozenSet[str] = frozenset({
    ".json", ".xml", ".csv", ".tsv", ".yaml", ".yml", ".toml", ".ini",
    ".env", ".properties",
})

CONFIG_EXTENSIONS: FrozenSet[str] = frozenset({
    ".cfg", ".conf", ".ini", ".toml", ".yaml", ".yml",
    ".editorconfig", ".eslintrc",
})

DOCUMENT_EXTENSIONS: FrozenSet[str] = frozenset({
    ".md", ".rst", ".txt", ".pdf", ".doc", ".docx", ".odt",
    ".rtf",
})

# ── Skip / ignore patterns ──────────────────────────────────────────────────

SKIP_PATTERNS: tuple[str, ...] = (
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".eggs", "*.egg-info",
    "dist", "build", ".next", ".nuxt", ".cache", "coverage",
    ".nyc_output", ".turbo",
)

# Default set of directory patterns to skip during recursive scans
DEFAULT_SKIP_DIRS: FrozenSet[str] = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache",
})


# ── Helper functions ────────────────────────────────────────────────────────

def get_extension(filepath: str) -> str:
    """Return the lowercase file extension including the dot.

    Examples:
        >>> get_extension("src/main.py")
        '.py'
        >>> get_extension("Makefile")
        ''
        >>> get_extension("archive.tar.gz")
        '.gz'
    """
    return Path(filepath).suffix.lower()


def matches_pattern(filepath: str, pattern: str) -> bool:
    """Check whether *filepath* matches a glob-style *pattern*.

    The match is performed **case-insensitively**.  Only the filename
    component (``Path(filepath).name``) is compared against the pattern.

    Supported glob tokens: ``*``, ``?``, ``[]`` (via :func:`fnmatch`).

    Args:
        filepath: Absolute or relative path to a file.
        pattern:  A glob pattern such as ``"*.py"`` or ``"test_*.py"``.

    Returns:
        ``True`` if the filename matches the pattern.
    """
    import fnmatch
    return fnmatch.fnmatch(Path(filepath).name.lower(), pattern.lower())


def matches_any_pattern(filepath: str, patterns: Iterable[str]) -> bool:
    """Return ``True`` if *filepath* matches **any** pattern in *patterns*."""
    return any(matches_pattern(filepath, p) for p in patterns)


def is_test_file(filepath: str) -> bool:
    """Heuristic: return ``True`` if *filepath* looks like a test file.

    Detection is based on common naming conventions (``*_test.*``,
    ``test_*.py``, ``*.test.*``, ``*.spec.*``, etc.).
    """
    return matches_any_pattern(filepath, TEST_PATTERNS)


def should_skip(filepath: str) -> bool:
    """Return ``True`` if *filepath* matches any built-in skip pattern.

    Both the filename **and** any directory component are checked.
    """
    parts = Path(filepath).parts
    name = Path(filepath).name

    for pattern in SKIP_PATTERNS:
        # Directory-level match
        if pattern.startswith("*"):
            if matches_pattern(name, pattern):
                return True
        else:
            # Exact directory name match or exact filename match
            if pattern in parts or name == pattern:
                return True
    return False


# ── Category classification ─────────────────────────────────────────────────

def classify_file(filepath: str) -> str:
    """Return a human-readable category string for *filepath*.

    Categories (in priority order):
    1. ``"test"``        – test file (see :func:`is_test_file`)
    2. ``"code"``        – source code (see :data:`CODE_EXTENSIONS`)
    3. ``"config"``      – configuration file
    4. ``"data"``        – data / serialisation format
    5. ``"document"``    – documentation / text
    6. ``"other"``       – anything else
    """
    if is_test_file(filepath):
        return "test"

    ext = get_extension(filepath)
    if ext in CODE_EXTENSIONS:
        return "code"
    if ext in CONFIG_EXTENSIONS:
        return "config"
    if ext in DATA_EXTENSIONS:
        return "data"
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    return "other"


# ── Scanning ────────────────────────────────────────────────────────────────

def scan_directory(
    root: str,
    *,
    extensions: Optional[Iterable[str]] = None,
    skip_dirs: Optional[Iterable[str]] = None,
    recursive: bool = True,
) -> list[str]:
    """Walk *root* and return a sorted list of matching file paths.

    Args:
        root:       Directory to scan.
        extensions: If given, only yield files whose extension (lowercase)
                    is in this collection.
        skip_dirs:  Directory names to skip.  Defaults to
                    :data:`DEFAULT_SKIP_DIRS`.
        recursive:  When ``False`` only the immediate children of *root*
                    are considered.

    Returns:
        Sorted list of file paths relative to *root*.
    """
    _skip = set(skip_dirs) if skip_dirs is not None else set(DEFAULT_SKIP_DIRS)
    _exts: Optional[set[str]] = set(e.lower() for e in extensions) if extensions else None

    found: list[str] = []

    def _walk(current: Path) -> None:
        try:
            entries = sorted(current.iterdir())
        except PermissionError:
            return
        for entry in entries:
            if entry.is_dir():
                if recursive and entry.name not in _skip:
                    _walk(entry)
            elif entry.is_file():
                if _exts is not None and entry.suffix.lower() not in _exts:
                    continue
                found.append(str(entry.relative_to(root)))

    _walk(Path(root))
    return found
