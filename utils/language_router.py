# language_router.py
"""
Maps file paths to language names and dispatches to the correct analyzer.

Detection order:
  1. Exact file name  (handles dotfiles like .zshrc, .gitconfig)
  2. File extension   (handles .py, .go, .rs, .yaml, etc.)
  3. Shebang line     (handles extensionless shell scripts)
"""

import os
from typing import Optional

from analyzers.python_analyzer import analyze_python
from analyzers.go_analyzer import analyze_go
from analyzers.js_analyzer import analyze_js
from analyzers.shell_analyzer import analyze_shell
from analyzers.config_analyzer import analyze_config
from analyzers.generic_ts_analyzer import analyze_generic
from models import FileAnalysis

# ---------------------------------------------------------------------------
# Extension → language name
# ---------------------------------------------------------------------------

EXTENSION_MAP = {
    # --- Already supported (original) ---
    ".py":   "python",
    ".go":   "go",
    ".js":   "javascript",
    ".ts":   "javascript",   # TypeScript uses the same JS analyzer
    ".jsx":  "javascript",
    ".tsx":  "javascript",
    ".mjs":  "javascript",
    ".cjs":  "javascript",

    # --- New: generic tree-sitter languages ---
    ".rs":   "rust",
    ".java": "java",
    ".rb":   "ruby",
    ".rake": "ruby",
    ".c":    "c",
    ".h":    "c",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".cxx":  "cpp",
    ".hh":   "cpp",
    ".hpp":  "cpp",
    ".hxx":  "cpp",
    ".lua":  "lua",

    # --- Shell scripts ---
    ".sh":   "shell",
    ".bash": "shell",
    ".zsh":  "shell",
    ".fish": "shell",

    # --- Configuration / data ---
    ".yaml":       "config",
    ".yml":        "config",
    ".toml":       "config",
    ".json":       "config",
    ".jsonc":      "config",
    ".ini":        "config",
    ".cfg":        "config",
    ".conf":       "config",
    ".env":        "config",
    ".properties": "config",
    ".editorconfig": "config",
    ".prettierrc": "config",
    ".eslintrc":   "config",
    ".babelrc":    "config",
    ".stylelintrc": "config",
}

# ---------------------------------------------------------------------------
# Exact file name → language name
# (handles extensionless dotfiles and special build files)
# ---------------------------------------------------------------------------

FILENAME_MAP = {
    # --- Shell dotfiles ---
    ".zshrc":          "shell",
    ".zshenv":         "shell",
    ".zprofile":       "shell",
    ".zlogin":         "shell",
    ".zlogout":        "shell",
    ".bashrc":         "shell",
    ".bash_profile":   "shell",
    ".bash_aliases":   "shell",
    ".bash_functions": "shell",
    ".bash_logout":    "shell",
    ".profile":        "shell",
    ".kshrc":          "shell",
    ".mkshrc":         "shell",
    ".fishrc":         "shell",

    # --- Config dotfiles ---
    ".gitconfig":         "config",
    ".gitattributes":     "config",
    ".gitignore":         "config",
    ".gitignore_global":  "config",
    ".gitmodules":        "config",
    ".gitmessage":        "config",
    ".hgignore":          "config",
    ".editorconfig":      "config",
    ".curlrc":            "config",
    ".wgetrc":            "config",
    ".npmrc":             "config",
    ".yarnrc":            "config",
    ".pip.conf":          "config",
    ".pydistutils.cfg":   "config",
    ".pylintrc":          "config",
    ".flake8":            "config",
    ".isort.cfg":         "config",
    ".coveragerc":        "config",
    ".tmux.conf":         "config",
    ".inputrc":           "config",
    ".dircolors":         "config",
    ".xinitrc":           "shell",
    ".Xresources":        "config",
    ".vimrc":             "config",    # Vimscript, treat as generic config
    ".gvimrc":            "config",
    ".ideavimrc":         "config",

    # --- Build / package management files ---
    "Makefile":    "config",
    "GNUmakefile": "config",
    "Dockerfile":  "config",
    "Brewfile":    "config",
    "Rakefile":    "ruby",
    "Gemfile":     "ruby",
    "Podfile":     "ruby",
    "Fastfile":    "ruby",
    "Guardfile":   "ruby",
    "Vagrantfile": "ruby",
    "Capfile":     "ruby",

    # --- SSH ---
    "config":      "config",   # matched only when in an .ssh directory
}

# Shell names recognised in shebangs
_SHELL_SHEBANGS = {
    "bash", "sh", "zsh", "fish", "ksh", "mksh", "dash", "ash",
}


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(file_path: str) -> Optional[str]:
    """
    Return the language name for `file_path`, or None if unrecognised.

    Detection priority:
      1. Exact basename match in FILENAME_MAP
      2. Extension match in EXTENSION_MAP
      3. Shebang line check (for extensionless scripts)
    """
    name = os.path.basename(file_path)
    ext = os.path.splitext(name)[1].lower()

    # 1. Exact name match
    lang = FILENAME_MAP.get(name)
    if lang:
        return lang

    # 2. Extension match
    lang = EXTENSION_MAP.get(ext)
    if lang:
        return lang

    # 3. Shebang check for extensionless files
    if not ext:
        lang = _detect_from_shebang(file_path)
        if lang:
            return lang

    return None


def _detect_from_shebang(file_path: str) -> Optional[str]:
    """Read the first line and check for a shebang that indicates the language."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline(200).strip()
    except Exception:
        return None

    if not first_line.startswith("#!"):
        return None

    shebang = first_line[2:].strip()
    # e.g. /usr/bin/env bash  →  ["bash"]
    parts = shebang.split()
    interp = parts[-1].lower() if parts else ""
    interp = os.path.basename(interp)

    if interp in _SHELL_SHEBANGS:
        return "shell"
    if interp == "python3" or interp == "python":
        return "python"
    if interp == "ruby" or interp == "rbenv":
        return "ruby"
    if interp == "lua" or interp.startswith("lua5"):
        return "lua"
    if interp == "node" or interp == "nodejs":
        return "javascript"

    return None


# ---------------------------------------------------------------------------
# Dispatch to analyzers
# ---------------------------------------------------------------------------

# Languages handled by the generic tree-sitter analyzer
_GENERIC_TS_LANGUAGES = {"rust", "java", "ruby", "c", "cpp", "lua"}


def analyze_file(file_path: str) -> Optional[FileAnalysis]:
    """
    Detect the language of `file_path` and run the appropriate analyzer.
    Returns None for unrecognised or binary files.
    """
    language = detect_language(file_path)

    if language == "python":
        return analyze_python(file_path)

    if language == "go":
        return analyze_go(file_path)

    if language == "javascript":
        return analyze_js(file_path)

    if language == "shell":
        return analyze_shell(file_path)

    if language == "config":
        return analyze_config(file_path)

    if language in _GENERIC_TS_LANGUAGES:
        return analyze_generic(file_path, language)

    return None
