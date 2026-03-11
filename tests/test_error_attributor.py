"""Tests for the compiler error attribution system."""

import pytest
from core.error_attributor import (
    AttributedError,
    AttributionResult,
    CompilerErrorAttributor,
)


class TestAttributionResult:
    def test_empty_result(self):
        result = AttributionResult()
        assert result.affected_files == []
        assert result.total_errors == 0

    def test_errors_for_file(self):
        result = AttributionResult(
            errors_by_file={
                "User.java": [
                    AttributedError(file_path="User.java", line=10, message="missing ;"),
                    AttributedError(file_path="User.java", line=20, message="type mismatch"),
                ],
            }
        )
        assert result.total_errors == 2
        assert result.affected_files == ["User.java"]
        assert result.errors_for_file("User.java") == ["missing ;", "type mismatch"]
        assert result.errors_for_file("Other.java") == []

    def test_summary_for_file(self):
        result = AttributionResult(
            errors_by_file={
                "Svc.java": [
                    AttributedError(file_path="Svc.java", line=5, message="cannot find symbol"),
                ],
            }
        )
        summary = result.summary_for_file("Svc.java")
        assert "Svc.java:5" in summary
        assert "cannot find symbol" in summary
        assert result.summary_for_file("Other.java") == ""


class TestCompilerErrorAttributor:
    def setup_method(self):
        self.attributor = CompilerErrorAttributor()

    # ── Java/Maven errors ──────────────────────────────────────────

    def test_java_maven_error(self):
        output = (
            "[ERROR] /workspace/src/main/java/com/example/User.java:[10,5] "
            "error: cannot find symbol\n"
            "[ERROR]   symbol: method getName()\n"
        )
        result = self.attributor.attribute(output)
        assert result.total_errors >= 1
        files = result.affected_files
        assert any("User.java" in f for f in files)

    def test_javac_error(self):
        output = "UserService.java:45: error: incompatible types\n"
        result = self.attributor.attribute(output)
        assert "UserService.java" in result.affected_files

    # ── Go errors ──────────────────────────────────────────────────

    def test_go_error(self):
        output = "./pkg/handlers/user.go:23:15: undefined: models.User\n"
        result = self.attributor.attribute(output)
        assert result.total_errors == 1
        files = result.affected_files
        assert any("user.go" in f for f in files)

    # ── TypeScript errors ──────────────────────────────────────────

    def test_typescript_error(self):
        output = "src/services/user.ts(15,3): error TS2304: Cannot find name 'User'\n"
        result = self.attributor.attribute(output)
        assert result.total_errors == 1
        files = result.affected_files
        assert any("user.ts" in f for f in files)

    # ── Rust errors ────────────────────────────────────────────────

    def test_rust_error(self):
        output = (
            "error[E0308]: mismatched types\n"
            " --> src/handlers/user.rs:10:5\n"
        )
        result = self.attributor.attribute(output)
        assert result.total_errors == 1
        files = result.affected_files
        assert any("user.rs" in f for f in files)

    # ── C# errors ──────────────────────────────────────────────────

    def test_csharp_error(self):
        output = "Services/UserService.cs(20,10): error CS1002: ; expected\n"
        result = self.attributor.attribute(output)
        assert result.total_errors == 1
        files = result.affected_files
        assert any("UserService.cs" in f for f in files)

    # ── Generic errors ─────────────────────────────────────────────

    def test_generic_file_line_error(self):
        output = "main.py:42: SyntaxError: unexpected indent\n"
        result = self.attributor.attribute(output)
        assert "main.py" in result.affected_files

    # ── Multiple files ─────────────────────────────────────────────

    def test_multiple_files(self):
        output = (
            "UserService.java:10: error: cannot find symbol\n"
            "OrderService.java:20: error: incompatible types\n"
            "UserService.java:15: error: missing return\n"
        )
        result = self.attributor.attribute(output)
        assert len(result.affected_files) == 2
        assert len(result.errors_by_file.get("UserService.java", [])) == 2
        assert len(result.errors_by_file.get("OrderService.java", [])) == 1

    # ── Known files filtering ──────────────────────────────────────

    def test_known_files_filter(self):
        output = (
            "User.java:10: error: missing ;\n"
            "Unknown.java:5: error: type mismatch\n"
        )
        result = self.attributor.attribute(
            output, known_files={"User.java"}
        )
        assert "User.java" in result.affected_files
        assert "Unknown.java" not in result.affected_files

    def test_known_files_suffix_matching(self):
        output = "User.java:10: error: missing ;\n"
        result = self.attributor.attribute(
            output,
            known_files={"src/main/java/com/example/User.java"},
        )
        assert "src/main/java/com/example/User.java" in result.affected_files

    # ── Deduplication ──────────────────────────────────────────────

    def test_deduplication(self):
        output = (
            "App.java:10: error: missing ;\n"
            "App.java:10: error: missing ;\n"
        )
        result = self.attributor.attribute(output)
        assert len(result.errors_by_file.get("App.java", [])) == 1

    # ── Empty / no errors ──────────────────────────────────────────

    def test_empty_output(self):
        result = self.attributor.attribute("")
        assert result.total_errors == 0

    def test_clean_build_output(self):
        output = "BUILD SUCCESS\nTotal time: 5.2s\n"
        result = self.attributor.attribute(output)
        assert result.total_errors == 0

    # ── Unattributed errors ────────────────────────────────────────

    def test_unattributed_error_lines(self):
        output = "FATAL ERROR: compilation failed with 3 errors\n"
        result = self.attributor.attribute(output)
        assert len(result.unattributed_errors) >= 1
