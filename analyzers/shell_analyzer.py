# analyzers/shell_analyzer.py
"""
Shell script / dotfile analyzer for bash, zsh, fish, and related dotfiles.

Extracts:
  - Function definitions  (function foo() { ... }  or  foo() { ... })
  - Alias definitions     (alias foo='...')
  - Export declarations   (export FOO=...)
  - Source / import paths (source ./lib.sh  or  . ./lib.sh)

Works via regex — no tree-sitter dependency required.
"""

import os
import re
from typing import List, Optional

from models import FileAnalysis, FunctionInfo, ClassInfo

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# function foo { ... }  /  function foo() { ... }
_FUNC_KW = re.compile(
    r"^function\s+([\w][\w.-]*)\s*(?:\(\s*\))?\s*\{",
    re.MULTILINE,
)

# foo() { ... }  /  foo () { ... }
# Guard against shell keywords appearing as bare names.
_FUNC_BARE = re.compile(
    r"^([\w][\w.-]*)\s*\(\s*\)\s*\{",
    re.MULTILINE,
)
_SHELL_KEYWORDS = frozenset(
    {"if", "while", "for", "case", "until", "do", "then", "else", "elif",
     "fi", "done", "esac", "in", "select", "time", "coproc"}
)

# alias foo='bar'  /  alias foo="bar"  /  alias foo=bar
_ALIAS = re.compile(
    r"""^alias\s+([\w][\w.-]*)\s*=\s*(?:'([^']*)'|"([^"]*)"|(\S+))""",
    re.MULTILINE,
)

# export FOO=value  /  export FOO="value"
_EXPORT = re.compile(
    r"""^export\s+([\w]+)\s*=\s*(?:'([^']*)'|"([^"]*)"|(\S*))""",
    re.MULTILINE,
)

# source ./file.sh  /  . ./file.sh
_SOURCE = re.compile(
    r"^(?:source|\.)\s+(['\"]?)(.*?)\1\s*$",
    re.MULTILINE,
)

# Leading # comment block
def _preceding_comment(source: str, match_start: int) -> str:
    """Extract the # comment block immediately above match_start."""
    lines_before = source[:match_start].splitlines()
    comment_parts: List[str] = []
    for line in reversed(lines_before):
        stripped = line.strip()
        if stripped.startswith("#"):
            comment_parts.insert(0, stripped[1:].strip())
        elif stripped == "":
            continue
        else:
            break
    return " ".join(comment_parts)


# ---------------------------------------------------------------------------
# Public analyzer
# ---------------------------------------------------------------------------

def analyze_shell(file_path: str) -> Optional[FileAnalysis]:
    """
    Analyze a shell script or shell-adjacent dotfile.

    Returns a FileAnalysis whose `functions` list contains both real
    shell functions and alias entries (prefixed `alias:`).
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
    except Exception:
        return None

    functions: List[FunctionInfo] = []
    imports: List[str] = []
    seen_names: set = set()

    # --- Shell functions (keyword form) ---
    for m in _FUNC_KW.finditer(source):
        name = m.group(1)
        if name in seen_names:
            continue
        seen_names.add(name)
        line_no = source[: m.start()].count("\n")
        sig = m.group(0).rstrip("{").strip()
        functions.append(
            FunctionInfo(
                name=name,
                line=line_no,
                signature=sig,
                docstring=_preceding_comment(source, m.start()),
            )
        )

    # --- Shell functions (bare  name() { form) ---
    for m in _FUNC_BARE.finditer(source):
        name = m.group(1)
        if name in seen_names or name in _SHELL_KEYWORDS:
            continue
        seen_names.add(name)
        line_no = source[: m.start()].count("\n")
        sig = m.group(0).rstrip("{").strip()
        functions.append(
            FunctionInfo(
                name=name,
                line=line_no,
                signature=sig,
                docstring=_preceding_comment(source, m.start()),
            )
        )

    # --- Aliases ---
    for m in _ALIAS.finditer(source):
        name = m.group(1)
        value = m.group(2) or m.group(3) or m.group(4) or ""
        line_no = source[: m.start()].count("\n")
        functions.append(
            FunctionInfo(
                name=f"alias:{name}",
                line=line_no,
                signature=f"alias {name}='{value}'",
                docstring=_preceding_comment(source, m.start()),
            )
        )

    # --- Source / imports ---
    for m in _SOURCE.finditer(source):
        path = m.group(2).strip()
        if path:
            imports.append(path)

    # Detect shell variant from file name / extension
    base = os.path.basename(file_path)
    ext = os.path.splitext(base)[1].lstrip(".")
    if "zsh" in base or ext == "zsh":
        language = "zsh"
    elif "fish" in base or ext == "fish":
        language = "fish"
    else:
        language = "bash"

    return FileAnalysis(
        file_path=file_path,
        language=language,
        functions=functions,
        classes=[],
        imports=imports,
    )
