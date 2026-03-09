"""Tests for Pydantic-based LLM schema validation."""

import pytest
from pydantic import ValidationError

from core.llm_schema import (
    ArchitectureResponse,
    FileBlueprintSchema,
    ReviewFindingSchema,
    ReviewResponse,
    SecurityResponse,
    SecurityVulnerability,
    validate_architecture_response,
    validate_review_response,
    validate_security_response,
)


# ── ReviewResponse ───────────────────────────────────────────────────────────


class TestReviewResponse:
    def test_valid_review(self):
        data = {"passed": True, "summary": "All good", "findings": []}
        result = validate_review_response(data)
        assert result.passed is True
        assert result.summary == "All good"
        assert result.findings == []

    def test_missing_passed_raises(self):
        """Core invariant: missing `passed` must fail, not silently default to True."""
        with pytest.raises(ValidationError):
            validate_review_response({"summary": "ok", "findings": []})

    def test_missing_summary_raises(self):
        with pytest.raises(ValidationError):
            validate_review_response({"passed": True, "findings": []})

    def test_empty_dict_raises(self):
        with pytest.raises(ValidationError):
            validate_review_response({})

    def test_none_raises(self):
        with pytest.raises(ValidationError):
            validate_review_response(None)

    def test_findings_with_all_fields(self):
        data = {
            "passed": False,
            "summary": "Issues found",
            "findings": [
                {
                    "severity": "critical",
                    "file": "app.py",
                    "line": 42,
                    "message": "SQL injection",
                    "suggestion": "Use parameterised queries",
                },
            ],
        }
        result = validate_review_response(data)
        assert not result.passed
        assert len(result.findings) == 1
        f = result.findings[0]
        assert f.severity == "critical"
        assert f.file == "app.py"
        assert f.line == 42
        assert f.message == "SQL injection"
        assert f.suggestion == "Use parameterised queries"

    def test_finding_alias_description_to_message(self):
        data = {
            "passed": True,
            "summary": "ok",
            "findings": [{"severity": "info", "description": "style issue"}],
        }
        result = validate_review_response(data)
        assert result.findings[0].message == "style issue"

    def test_finding_alias_fix_to_suggestion(self):
        data = {
            "passed": True,
            "summary": "ok",
            "findings": [{"severity": "info", "fix": "rename variable"}],
        }
        result = validate_review_response(data)
        assert result.findings[0].suggestion == "rename variable"

    def test_top_level_alias_success_to_passed(self):
        data = {"success": True, "summary": "looks fine", "findings": []}
        result = validate_review_response(data)
        assert result.passed is True

    def test_top_level_alias_issues_to_findings(self):
        data = {
            "passed": False,
            "summary": "bad",
            "issues": [{"severity": "warning", "message": "bad style"}],
        }
        result = validate_review_response(data)
        assert len(result.findings) == 1

    def test_top_level_alias_overview_to_summary(self):
        data = {"passed": True, "overview": "everything fine"}
        result = validate_review_response(data)
        assert result.summary == "everything fine"


# ── SecurityResponse ─────────────────────────────────────────────────────────


class TestSecurityResponse:
    def test_valid_security_response(self):
        data = {"passed": True, "summary": "No vulns", "vulnerabilities": []}
        result = validate_security_response(data)
        assert result.passed is True

    def test_missing_passed_raises(self):
        with pytest.raises(ValidationError):
            validate_security_response({"summary": "ok", "vulnerabilities": []})

    def test_missing_summary_raises(self):
        with pytest.raises(ValidationError):
            validate_security_response({"passed": True, "vulnerabilities": []})

    def test_vulnerability_fields(self):
        data = {
            "passed": False,
            "summary": "Found issues",
            "vulnerabilities": [
                {
                    "severity": "high",
                    "file": "db.py",
                    "line": 10,
                    "type": "SQL_INJECTION",
                    "description": "Unparameterised query",
                    "remediation": "Use parameterised queries",
                },
            ],
        }
        result = validate_security_response(data)
        v = result.vulnerabilities[0]
        assert v.severity == "high"
        assert v.type == "SQL_INJECTION"
        assert v.description == "Unparameterised query"
        assert v.remediation == "Use parameterised queries"

    def test_vuln_alias_fix_to_remediation(self):
        data = {
            "passed": True,
            "summary": "ok",
            "vulnerabilities": [{"severity": "low", "fix": "update lib"}],
        }
        result = validate_security_response(data)
        assert result.vulnerabilities[0].remediation == "update lib"

    def test_top_level_alias_vulns(self):
        data = {
            "passed": True,
            "summary": "ok",
            "vulns": [{"severity": "medium", "description": "weak hash"}],
        }
        result = validate_security_response(data)
        assert len(result.vulnerabilities) == 1


# ── ArchitectureResponse ─────────────────────────────────────────────────────


class TestArchitectureResponse:
    def test_valid_architecture(self):
        data = {
            "name": "my-project",
            "description": "A project",
            "architecture_style": "REST",
            "tech_stack": {"language": "python"},
            "folder_structure": ["models", "services"],
            "file_blueprints": [
                {"path": "models/user.py", "purpose": "User model"},
            ],
            "architecture_doc": "# My Project",
        }
        result = validate_architecture_response(data)
        assert result.name == "my-project"
        assert len(result.file_blueprints) == 1
        assert result.file_blueprints[0].path == "models/user.py"

    def test_empty_dict_uses_defaults(self):
        result = validate_architecture_response({})
        assert result.name == "unnamed-project"
        assert result.architecture_style == "REST"
        assert result.file_blueprints == []

    def test_none_uses_defaults(self):
        result = validate_architecture_response(None)
        assert result.name == "unnamed-project"

    def test_blueprint_alias_description_to_purpose(self):
        data = {
            "file_blueprints": [
                {"path": "app.py", "description": "Main entry point"},
            ],
        }
        result = validate_architecture_response(data)
        assert result.file_blueprints[0].purpose == "Main entry point"

    def test_blueprint_alias_dependencies_to_depends_on(self):
        data = {
            "file_blueprints": [
                {"path": "svc.py", "dependencies": ["models/user.py"]},
            ],
        }
        result = validate_architecture_response(data)
        assert result.file_blueprints[0].depends_on == ["models/user.py"]

    def test_top_level_alias_files_to_file_blueprints(self):
        data = {"files": [{"path": "main.py"}]}
        result = validate_architecture_response(data)
        assert len(result.file_blueprints) == 1

    def test_blueprint_missing_path_raises(self):
        data = {"file_blueprints": [{"purpose": "no path"}]}
        with pytest.raises(ValidationError):
            validate_architecture_response(data)

    def test_top_level_alias_arch_style(self):
        data = {"arch_style": "GraphQL"}
        result = validate_architecture_response(data)
        assert result.architecture_style == "GraphQL"
