# utils/file_scanner.py

import os

IGNORE_DIRS = {
    # Version control
    ".git", ".svn", ".hg",
    # JS/TS
    "node_modules", "dist", "build", ".next", ".nuxt", ".turbo",
    # Python
    "__pycache__", ".venv", "venv", "env", ".env", ".tox",
    # Go
    "vendor",
    # Java/Kotlin/Scala
    "target",
    # Rust
    "target",
    # IDE / tooling
    ".idea", ".vscode", ".DS_Store",
    # Build outputs
    "out", "bin", "obj", ".cache", ".tmp", "tmp",
    # Test coverage
    "coverage", ".coverage", "htmlcov",
    # Generated / infra
    "migrations",
}

# Skip files whose names contain these suffixes (generated or minified)
IGNORE_FILE_SUFFIXES = (
    ".min.js", ".min.css",
    ".pb.go", ".pb.gw.go",   # protobuf generated
    "_gen.go",               # generated Go
    ".g.dart",               # generated Flutter
)

BINARY_EXTENSIONS = frozenset({
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico", ".webp",
    # Documents / archives
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z",
    # Media
    ".mp4", ".mp3", ".wav", ".avi", ".mov",
    # Fonts
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # Compiled / binary
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".dylib", ".o", ".a",
    # Lock files (large, no semantic value)
    ".lock",
    # Misc
    ".pkl", ".bin", ".dat",
    # WebAssembly
    ".wasm",
})

GENERATED_FILE_PATTERNS = (
    # Lock files
    "package-lock.json",
    "yarn.lock",
    "Pipfile.lock",
    "poetry.lock",
    "Cargo.lock",
    "go.sum",
    "composer.lock",
    # Compiled outputs
    ".pyc",
    # Minified bundles
    ".min.js",
    ".min.css",
    # Protobuf generated
    ".pb.go",
    ".pb.rs",
    # Generic generated
    ".generated.",
)

MAX_FILE_SIZE_BYTES = 1_000_000  # Skip files larger than 1 MB


def scan_repository(repo_path: str, include_dotfiles: bool = False):
    """
    Walk `repo_path` and return a list of source file paths to index.

    Filters out binary files, generated/lock files, oversized files,
    and (optionally) dotfiles.

    Args:
        repo_path:        Root directory to scan.
        include_dotfiles: When True, include hidden files and directories
                          (those starting with '.'), except for .git and
                          other VCS directories.  Enable this when indexing
                          dotfile repositories like ~/.dotfiles.

    Returns:
        List of absolute file paths that passed all filters.
    """
    files = []

    for root, dirs, filenames in os.walk(repo_path):
        if include_dotfiles:
            # Allow hidden dirs, but always prune VCS and known build dirs
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        else:
            # Default: prune hidden dirs AND known ignore dirs
            dirs[:] = [
                d for d in dirs
                if d not in IGNORE_DIRS and not d.startswith(".")
            ]

        for filename in filenames:
            # In normal mode, skip hidden files
            if not include_dotfiles and filename.startswith("."):
                continue

            # Skip by extension (binary files)
            _, ext = os.path.splitext(filename)
            if ext.lower() in BINARY_EXTENSIONS:
                continue

            # Skip generated / minified files by suffix
            if any(filename.endswith(suffix) for suffix in IGNORE_FILE_SUFFIXES):
                continue

            # Skip generated / lock files by exact name or pattern
            if _is_generated_file(filename):
                continue

            path = os.path.join(root, filename)

            # Skip oversized files
            try:
                if os.path.getsize(path) > MAX_FILE_SIZE_BYTES:
                    continue
            except OSError:
                continue

            files.append(path)

    return files


def _is_generated_file(filename: str) -> bool:
    """Return True if *filename* matches a known generated-file pattern."""
    for pattern in GENERATED_FILE_PATTERNS:
        if pattern.startswith(".") and pattern.endswith("."):
            # Pattern like ".generated." — match anywhere in filename
            if pattern.strip(".") in filename:
                return True
        elif filename == pattern or filename.endswith(pattern):
            return True
    return False
