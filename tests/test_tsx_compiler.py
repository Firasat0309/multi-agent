"""Tests for TSXCompiler — error parsing (tsc and vue-tsc formats) and binary resolution."""

import pytest
from pathlib import Path
from unittest.mock import patch

from core.tsx_compiler import (
    TSXCompiler,
    TSXError,
    _TSC_ERROR_RE,
    _VUE_TSC_ERROR_RE,
)


class TestErrorRegex:
    """Verify both tsc and vue-tsc output formats are matched."""

    def test_tsc_format_matches(self):
        """Standard tsc --pretty false output: file(line,col): error TS...: msg"""
        line = "src/components/Button.tsx(10,5): error TS2304: Cannot find name 'x'."
        m = _TSC_ERROR_RE.match(line)
        assert m is not None
        assert m.group("file") == "src/components/Button.tsx"
        assert m.group("line") == "10"
        assert m.group("col") == "5"
        assert m.group("code") == "TS2304"
        assert "Cannot find name" in m.group("message")

    def test_vue_tsc_format_matches(self):
        """vue-tsc output: file:line:col - error TS...: msg"""
        line = "src/components/feature/LoginForm.vue:48:13 - error TS2339: Property 'ref' does not exist on type 'Ref<string, string>'."
        m = _VUE_TSC_ERROR_RE.match(line)
        assert m is not None
        assert m.group("file") == "src/components/feature/LoginForm.vue"
        assert m.group("line") == "48"
        assert m.group("col") == "13"
        assert m.group("code") == "TS2339"
        assert "Property" in m.group("message")

    def test_vue_tsc_cannot_find_module(self):
        """vue-tsc TS2307 module-not-found error."""
        line = "src/layouts/MainLayout.vue:2:23 - error TS2307: Cannot find module '../components/AppNavbar.vue' or its corresponding type declarations."
        m = _VUE_TSC_ERROR_RE.match(line)
        assert m is not None
        assert m.group("code") == "TS2307"
        assert m.group("file") == "src/layouts/MainLayout.vue"

    def test_tsc_format_not_matched_by_vue_regex(self):
        """tsc format should NOT match the vue-tsc regex (they're distinct)."""
        line = "src/Button.tsx(10,5): error TS2304: Cannot find name 'x'."
        assert _VUE_TSC_ERROR_RE.match(line) is None

    def test_vue_tsc_format_not_matched_by_tsc_regex(self):
        """vue-tsc format should NOT match the tsc regex."""
        line = "src/Button.vue:10:5 - error TS2339: Property does not exist."
        assert _TSC_ERROR_RE.match(line) is None

    def test_non_error_lines_skipped(self):
        line = "Found 7 errors in 5 files."
        assert _TSC_ERROR_RE.match(line) is None
        assert _VUE_TSC_ERROR_RE.match(line) is None


class TestParseOutput:
    """Verify _parse_output handles mixed output from vue-tsc."""

    def test_parses_vue_tsc_errors(self):
        raw = (
            "src/components/feature/LoginForm.vue:48:13 - error TS2339: Property 'ref' does not exist.\n"
            "src/layouts/MainLayout.vue:2:23 - error TS2307: Cannot find module '../components/AppNavbar.vue'.\n"
            "\n"
            "Found 2 errors in 2 files.\n"
        )
        compiler = TSXCompiler()
        errors = compiler._parse_output(raw, Path("/workspace"))
        assert len(errors) == 2
        assert errors[0].file == "src/components/feature/LoginForm.vue"
        assert errors[0].code == "TS2339"
        assert errors[1].file == "src/layouts/MainLayout.vue"
        assert errors[1].code == "TS2307"

    def test_parses_tsc_errors(self):
        raw = (
            "src/utils/api.ts(5,10): error TS2305: Module has no exported member.\n"
            "src/store/auth.ts(12,3): error TS2304: Cannot find name 'defineStore'.\n"
        )
        compiler = TSXCompiler()
        errors = compiler._parse_output(raw, Path("/workspace"))
        assert len(errors) == 2
        assert errors[0].code == "TS2305"
        assert errors[1].code == "TS2304"

    def test_parses_mixed_output(self):
        """Handle hypothetical mixed tsc + vue-tsc lines."""
        raw = (
            "src/api.ts(5,10): error TS2305: Module has no exported member.\n"
            "src/App.vue:10:3 - error TS2339: Property missing.\n"
        )
        compiler = TSXCompiler()
        errors = compiler._parse_output(raw, Path("/workspace"))
        assert len(errors) == 2

    def test_empty_output(self):
        compiler = TSXCompiler()
        errors = compiler._parse_output("", Path("/workspace"))
        assert errors == []

    def test_all_seven_user_errors_parsed(self):
        """Reproduce the exact 7 errors from the bug report."""
        raw = (
            "src/components/feature/LoginForm.vue:48:13 - error TS2339: Property 'ref' does not exist on type 'Ref<string, string>'.\n"
            "src/components/feature/RegisterForm.vue:89:33 - error TS2339: Property 'message' does not exist on type 'void'.\n"
            "src/layouts/MainLayout.vue:2:23 - error TS2307: Cannot find module '../components/AppNavbar.vue' or its corresponding type declarations.\n"
            "src/pages/DashboardPage.vue:4:24 - error TS2307: Cannot find module '../components/layout/MainLayout.vue' or its corresponding type declarations.\n"
            "src/pages/DashboardPage.vue:23:23 - error TS2339: Property 'fetchUser' does not exist on type 'Store<...>'.\n"
            "src/pages/DashboardPage.vue:49:37 - error TS2339: Property 'fetchUser' does not exist on type 'Store<...>'.\n"
            "src/pages/LoginPage.vue:2:24 - error TS2307: Cannot find module '../components/layout/AuthLayout.vue' or its corresponding type declarations.\n"
            "\n"
            "Found 7 errors in 5 files.\n"
        )
        compiler = TSXCompiler()
        errors = compiler._parse_output(raw, Path("/workspace"))
        assert len(errors) == 7
        by_file = {}
        for e in errors:
            by_file.setdefault(e.file, []).append(e)
        assert len(by_file) == 5


class TestResolveCompiler:
    """Verify _resolve_compiler prefers local node_modules binaries."""

    @staticmethod
    def _bin_name(name: str) -> str:
        import platform
        return f"{name}.cmd" if platform.system() == "Windows" else name

    @staticmethod
    def _npx_name() -> str:
        import platform
        return "npx.cmd" if platform.system() == "Windows" else "npx"

    def test_prefers_local_vue_tsc(self, tmp_path):
        """Should return local vue-tsc path when it exists."""
        bin_dir = tmp_path / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True)
        vue_tsc = bin_dir / self._bin_name("vue-tsc")
        vue_tsc.write_text("#!/bin/sh\n")
        result = TSXCompiler._resolve_compiler(tmp_path, use_vue_tsc=True)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "node_modules" in result[0]
        assert "vue-tsc" in result[0]

    def test_falls_back_to_local_tsc(self, tmp_path):
        """When vue-tsc not in node_modules but tsc is, use local tsc."""
        bin_dir = tmp_path / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True)
        tsc = bin_dir / self._bin_name("tsc")
        tsc.write_text("#!/bin/sh\n")
        result = TSXCompiler._resolve_compiler(tmp_path, use_vue_tsc=True)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "node_modules" in result[0]
        assert "tsc" in result[0]

    def test_npx_fallback_when_no_local(self, tmp_path):
        """When no local binaries exist, use npx to resolve."""
        result = TSXCompiler._resolve_compiler(tmp_path, use_vue_tsc=True)
        assert isinstance(result, list)
        assert result == [self._npx_name(), "vue-tsc"]

    def test_npx_tsc_only(self, tmp_path):
        """Non-Vue project with no local binary should use npx tsc."""
        result = TSXCompiler._resolve_compiler(tmp_path, use_vue_tsc=False)
        assert result == [self._npx_name(), "tsc"]

    def test_local_tsc_preferred_for_non_vue(self, tmp_path):
        bin_dir = tmp_path / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True)
        tsc = bin_dir / self._bin_name("tsc")
        tsc.write_text("#!/bin/sh\n")
        result = TSXCompiler._resolve_compiler(tmp_path, use_vue_tsc=False)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "node_modules" in result[0]
        assert "tsc" in result[0]
