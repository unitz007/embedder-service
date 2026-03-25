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

BINARY_EXTENSIONS = {
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
}

MAX_FILE_SIZE_BYTES = 512 * 1024  # Skip files larger than 512 KB


def scan_repository(repo_path: str, include_dotfiles: bool = False):
    """
    Walk `repo_path` and return a list of source file paths to index.

    Args:
        repo_path:        Root directory to scan.
        include_dotfiles: When True, include hidden files and directories
                          (those starting with '.'), except for .git and
                          other VCS directories.  Enable this when indexing
                          dotfile repositories like ~/.dotfiles.
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

            # Skip by extension
            _, ext = os.path.splitext(filename)
            if ext.lower() in BINARY_EXTENSIONS:
                continue

            # Skip generated / minified files
            if any(filename.endswith(suffix) for suffix in IGNORE_FILE_SUFFIXES):
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
