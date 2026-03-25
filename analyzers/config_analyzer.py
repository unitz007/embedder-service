# analyzers/config_analyzer.py
"""
Configuration file analyzer for YAML, TOML, JSON, INI, .env, and generic
key-value dotfiles (.gitconfig, .npmrc, SSH config, etc.).

Maps config concepts onto the FileAnalysis model:
  Sections / top-level mapping keys  → ClassInfo   (things that group config)
  Key-value pairs / entries          → FunctionInfo (searchable config entries)
  Includes / extends / imports       → imports list

No tree-sitter required — uses Python stdlib + optional PyYAML / tomli.
"""

import json
import os
import re
import sys
from typing import List, Optional

from models import FileAnalysis, FunctionInfo, ClassInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_line_of(source: str, key: str) -> int:
    """Return the 0-based line number of the first occurrence of `key`."""
    idx = source.find(key)
    if idx == -1:
        return 0
    return source[:idx].count("\n")


def _truncate(value, max_len: int = 100) -> str:
    s = str(value)
    return s[:max_len] + "…" if len(s) > max_len else s


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------

def _analyze_yaml(file_path: str, source: str) -> FileAnalysis:
    try:
        import yaml  # PyYAML
        node = yaml.compose(source)
    except Exception:
        return _analyze_text_kv(file_path, source, "yaml")

    functions: List[FunctionInfo] = []
    classes: List[ClassInfo] = []

    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            key = str(key_node.value)
            line = key_node.start_mark.line

            if isinstance(value_node, yaml.MappingNode):
                # Top-level section → ClassInfo
                classes.append(ClassInfo(name=key, line=line, kind="yaml_section"))
                for k2_node, v2_node in value_node.value:
                    k2 = str(k2_node.value)
                    val = v2_node.value if hasattr(v2_node, "value") else ""
                    functions.append(
                        FunctionInfo(
                            name=f"{key}.{k2}",
                            line=k2_node.start_mark.line,
                            signature=f"{k2}: {_truncate(val)}",
                        )
                    )
            elif isinstance(value_node, yaml.SequenceNode):
                classes.append(ClassInfo(name=key, line=line, kind="yaml_list"))
            else:
                val = value_node.value if hasattr(value_node, "value") else ""
                functions.append(
                    FunctionInfo(
                        name=key,
                        line=line,
                        signature=f"{key}: {_truncate(val)}",
                    )
                )

    return FileAnalysis(
        file_path=file_path,
        language="yaml",
        functions=functions,
        classes=classes,
        imports=[],
    )


# ---------------------------------------------------------------------------
# TOML
# ---------------------------------------------------------------------------

def _analyze_toml(file_path: str, source: str) -> FileAnalysis:
    data = None
    try:
        if sys.version_info >= (3, 11):
            import tomllib
            data = tomllib.loads(source)
        else:
            import tomli  # pip install tomli
            data = tomli.loads(source)
    except ImportError:
        pass  # fall through to text parser
    except Exception:
        pass

    if data is None:
        return _analyze_text_kv(file_path, source, "toml")

    functions: List[FunctionInfo] = []
    classes: List[ClassInfo] = []

    for key, value in data.items():
        line = _first_line_of(source, key)
        if isinstance(value, dict):
            classes.append(ClassInfo(name=key, line=line, kind="toml_table"))
            for k2, v2 in value.items():
                functions.append(
                    FunctionInfo(
                        name=f"{key}.{k2}",
                        line=_first_line_of(source, k2),
                        signature=f"{k2} = {_truncate(v2)}",
                    )
                )
        elif isinstance(value, list):
            classes.append(ClassInfo(name=key, line=line, kind="toml_array"))
        else:
            functions.append(
                FunctionInfo(
                    name=key,
                    line=line,
                    signature=f"{key} = {_truncate(value)}",
                )
            )

    return FileAnalysis(
        file_path=file_path,
        language="toml",
        functions=functions,
        classes=classes,
        imports=[],
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def _analyze_json(file_path: str, source: str) -> FileAnalysis:
    try:
        data = json.loads(source)
    except json.JSONDecodeError:
        return _analyze_text_kv(file_path, source, "json")

    functions: List[FunctionInfo] = []
    classes: List[ClassInfo] = []

    if not isinstance(data, dict):
        return FileAnalysis(
            file_path=file_path, language="json",
            functions=[], classes=[], imports=[]
        )

    for key, value in data.items():
        line = _first_line_of(source, f'"{key}"')
        if isinstance(value, dict):
            classes.append(ClassInfo(name=key, line=line, kind="json_object"))
            for k2, v2 in value.items():
                functions.append(
                    FunctionInfo(
                        name=f"{key}.{k2}",
                        line=_first_line_of(source, f'"{k2}"'),
                        signature=f'"{k2}": {json.dumps(v2)[:100]}',
                    )
                )
        elif isinstance(value, list):
            classes.append(ClassInfo(name=key, line=line, kind="json_array"))
        else:
            functions.append(
                FunctionInfo(
                    name=key,
                    line=line,
                    signature=f'"{key}": {json.dumps(value)[:100]}',
                )
            )

    return FileAnalysis(
        file_path=file_path,
        language="json",
        functions=functions,
        classes=classes,
        imports=[],
    )


# ---------------------------------------------------------------------------
# INI / CFG / CONF / .gitconfig
# ---------------------------------------------------------------------------

def _analyze_ini(file_path: str, source: str) -> FileAnalysis:
    import configparser

    config = configparser.RawConfigParser(strict=False)
    try:
        config.read_string(source)
    except Exception:
        return _analyze_text_kv(file_path, source, "ini")

    functions: List[FunctionInfo] = []
    classes: List[ClassInfo] = []

    for section in config.sections():
        line = _first_line_of(source, f"[{section}]")
        classes.append(ClassInfo(name=section, line=line, kind="ini_section"))
        for key, value in config.items(section):
            functions.append(
                FunctionInfo(
                    name=f"{section}.{key}",
                    line=_first_line_of(source, key),
                    signature=f"{key} = {_truncate(value)}",
                )
            )

    return FileAnalysis(
        file_path=file_path,
        language="ini",
        functions=functions,
        classes=classes,
        imports=[],
    )


# ---------------------------------------------------------------------------
# Generic key=value / .env / SSH config / .gitignore / unknown text
# ---------------------------------------------------------------------------

_KV_RE = re.compile(r"^([\w][\w.@/-]*)\s*[=:]\s*(.*)", re.MULTILINE)
_SECTION_RE = re.compile(r"^\[([^\]]+)\]", re.MULTILINE)
_INCLUDE_RE = re.compile(
    r"(?:^include(?:Path|Optional)?\s+|^Include\s+|^source\s+|^\.\s+)(\S+)",
    re.MULTILINE | re.IGNORECASE,
)


def _analyze_text_kv(file_path: str, source: str, language: str) -> FileAnalysis:
    """
    Fallback parser: scans for [sections], KEY=VALUE pairs, and include/source lines.
    Works for .env, .gitignore (line per pattern), SSH config, etc.
    """
    functions: List[FunctionInfo] = []
    classes: List[ClassInfo] = []
    imports: List[str] = []
    pending_comment = ""
    current_section = ""

    for i, raw_line in enumerate(source.splitlines()):
        line = raw_line.strip()

        # Comments
        if line.startswith("#") or line.startswith(";"):
            pending_comment = line.lstrip("#;").strip()
            continue

        # Section headers [foo]
        sec_m = _SECTION_RE.match(line)
        if sec_m:
            current_section = sec_m.group(1)
            classes.append(
                ClassInfo(name=current_section, line=i, kind="section")
            )
            pending_comment = ""
            continue

        # KEY = VALUE / KEY: VALUE
        kv_m = _KV_RE.match(line)
        if kv_m:
            key = kv_m.group(1)
            value = kv_m.group(2).strip().strip("'\"")
            qualified = f"{current_section}.{key}" if current_section else key
            functions.append(
                FunctionInfo(
                    name=qualified,
                    line=i,
                    signature=f"{key} = {_truncate(value)}",
                    docstring=pending_comment,
                )
            )
            pending_comment = ""
            continue

        # Non-empty, non-comment, non-kv lines (e.g. .gitignore patterns)
        if line:
            functions.append(
                FunctionInfo(name=line[:60], line=i, signature=line[:120])
            )
        pending_comment = ""

    # Include / source lines
    for m in _INCLUDE_RE.finditer(source):
        imports.append(m.group(1).strip("'\""))

    return FileAnalysis(
        file_path=file_path,
        language=language,
        functions=functions,
        classes=classes,
        imports=imports,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# File names (basename) that should be parsed with the INI parser
_INI_NAMES = {
    ".gitconfig", ".gitattributes", ".editorconfig", ".npmrc",
    ".yarnrc", ".pip.conf", ".curlrc", ".wgetrc", ".inputrc",
    ".tmux.conf", "Makefile", "Dockerfile", "Brewfile",
}


def analyze_config(file_path: str) -> Optional[FileAnalysis]:
    """
    Dispatch to the correct config parser based on extension or file name.
    Returns None only if the file cannot be read.
    """
    ext = os.path.splitext(file_path)[1].lower()
    name = os.path.basename(file_path)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
    except Exception:
        return None

    if ext in (".yaml", ".yml"):
        return _analyze_yaml(file_path, source)
    if ext == ".toml":
        return _analyze_toml(file_path, source)
    if ext == ".json":
        return _analyze_json(file_path, source)
    if ext in (".ini", ".cfg") or name in _INI_NAMES:
        return _analyze_ini(file_path, source)
    # .env, .conf, .gitignore, SSH config, Brewfile, etc.
    lang = ext.lstrip(".") or "config"
    return _analyze_text_kv(file_path, source, lang)
