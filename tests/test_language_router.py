# tests/test_language_router.py
"""Unit tests for utils/language_router.py — language detection and dispatch."""

import os
import tempfile

import pytest

from utils.language_router import detect_language, _detect_from_shebang


# ---------------------------------------------------------------------------
# detect_language — extension-based detection
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_python(self):
        assert detect_language("main.py") == "python"

    def test_go(self):
        assert detect_language("main.go") == "go"

    def test_javascript(self):
        assert detect_language("app.js") == "javascript"

    def test_typescript(self):
        assert detect_language("app.ts") == "javascript"

    def test_tsx(self):
        assert detect_language("App.tsx") == "javascript"

    def test_jsx(self):
        assert detect_language("App.jsx") == "javascript"

    def test_mjs(self):
        assert detect_language("index.mjs") == "javascript"

    def test_cjs(self):
        assert detect_language("index.cjs") == "javascript"

    def test_shell_bash(self):
        assert detect_language("run.sh") == "shell"

    def test_shell_zsh(self):
        assert detect_language(".zshrc") == "shell"

    def test_config_yaml(self):
        assert detect_language("config.yaml") == "config"

    def test_config_yml(self):
        assert detect_language("config.yml") == "config"

    def test_config_toml(self):
        assert detect_language("pyproject.toml") == "config"

    def test_config_json(self):
        assert detect_language("settings.json") == "config"

    def test_config_ini(self):
        assert detect_language("setup.ini") == "config"

    def test_rust(self):
        assert detect_language("main.rs") == "rust"

    def test_java(self):
        assert detect_language("Main.java") == "java"

    def test_ruby(self):
        assert detect_language("app.rb") == "ruby"

    def test_c(self):
        assert detect_language("main.c") == "c"

    def test_cpp(self):
        assert detect_language("main.cpp") == "cpp"

    def test_lua(self):
        assert detect_language("script.lua") == "lua"

    def test_cpp_header(self):
        assert detect_language("header.hpp") == "cpp"

    def test_unknown_extension(self):
        assert detect_language("readme.xyz") is None

    def test_case_insensitive_extension(self):
        assert detect_language("MAIN.PY") == "python"

    def test_uppercase_json(self):
        assert detect_language("CONFIG.JSON") == "config"


# ---------------------------------------------------------------------------
# detect_language — filename-based detection (dotfiles / specials)
# ---------------------------------------------------------------------------

class TestDetectLanguageByFilename:
    def test_gitconfig(self):
        assert detect_language(".gitconfig") == "config"

    def test_gitignore(self):
        assert detect_language(".gitignore") == "config"

    def test_zshrc(self):
        assert detect_language(".zshrc") == "shell"

    def test_bashrc(self):
        assert detect_language(".bashrc") == "shell"

    def test_bash_profile(self):
        assert detect_language(".bash_profile") == "shell"

    def test_vimrc(self):
        assert detect_language(".vimrc") == "config"

    def test_npmrc(self):
        assert detect_language(".npmrc") == "config"

    def test_makefile(self):
        assert detect_language("Makefile") == "config"

    def test_dockerfile(self):
        assert detect_language("Dockerfile") == "config"

    def test_gemfile(self):
        assert detect_language("Gemfile") == "ruby"

    def test_rakefile(self):
        assert detect_language("Rakefile") == "ruby"

    def test_xinitrc(self):
        assert detect_language(".xinitrc") == "shell"

    def test_tmux_conf(self):
        assert detect_language(".tmux.conf") == "config"

    def test_eslintrc(self):
        assert detect_language(".eslintrc") == "config"


# ---------------------------------------------------------------------------
# _detect_from_shebang — shebang-based detection
# ---------------------------------------------------------------------------

class TestDetectFromShebang:
    def test_bash_shebang(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix="", delete=False) as f:
            f.write("#!/bin/bash\n")
            path = f.name
        try:
            assert _detect_from_shebang(path) == "shell"
        finally:
            os.unlink(path)

    def test_python_shebang(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix="", delete=False) as f:
            f.write("#!/usr/bin/env python3\n")
            path = f.name
        try:
            assert _detect_from_shebang(path) == "python"
        finally:
            os.unlink(path)

    def test_ruby_shebang(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix="", delete=False) as f:
            f.write("#!/usr/bin/env ruby\n")
            path = f.name
        try:
            assert _detect_from_shebang(path) == "ruby"
        finally:
            os.unlink(path)

    def test_no_shebang(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix="", delete=False) as f:
            f.write("echo hello\n")
            path = f.name
        try:
            assert _detect_from_shebang(path) is None
        finally:
            os.unlink(path)

    def test_lua_shebang(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix="", delete=False) as f:
            f.write("#!/usr/bin/lua\n")
            path = f.name
        try:
            assert _detect_from_shebang(path) == "lua"
        finally:
            os.unlink(path)

    def test_node_shebang(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix="", delete=False) as f:
            f.write("#!/usr/bin/node\n")
            path = f.name
        try:
            assert _detect_from_shebang(path) == "javascript"
        finally:
            os.unlink(path)
