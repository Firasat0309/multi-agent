"""Pydantic schema validation for LLM JSON responses.

LLMs sometimes return valid JSON with wrong key names, missing fields, or
incorrect types.  A pre-processing step resolves common key aliases, then
Pydantic models enforce the schema strictly.

**Key principle:** A validation failure is a task failure, not a silent pass.
The retry loop in agent_manager.py will re-run the task with fresh context.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

# Re-export so agents can ``from core.llm_schema import ValidationError``
__all__ = [
    "ArchitectureResponse",
    "FileBlueprintSchema",
    "ReviewFindingSchema",
    "ReviewResponse",
    "SecurityResponse",
    "SecurityVulnerability",
    "ValidationError",
    "validate_architecture_response",
    "validate_review_response",
    "validate_security_response",
]


# ── Top-level key aliases (resolved before Pydantic validation) ──────────────

_TOP_LEVEL_ALIASES: dict[str, list[str]] = {
    "passed": ["pass", "success", "ok", "approved"],
    "findings": ["issues", "problems", "errors", "violations"],
    "vulnerabilities": ["vulns", "vulnerability", "security_issues"],
    "summary": ["overview", "conclusion"],
    "file_blueprints": ["files", "blueprints", "file_list"],
    "architecture_style": ["architecture", "arch_style", "style"],
    "tech_stack": ["stack", "technology", "technologies"],
    "folder_structure": ["folders", "structure", "directories"],
}


def _normalize_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve common LLM key variants at the top level.

    Field-level aliases inside nested objects (findings, vulnerabilities,
    file_blueprints) are handled by Pydantic ``validation_alias`` on each
    model field — no recursive normalisation needed here.
    """
    if not data:
        return data

    normalized = dict(data)
    for canonical, alts in _TOP_LEVEL_ALIASES.items():
        if canonical not in normalized:
            for alt in alts:
                if alt in normalized:
                    normalized[canonical] = normalized.pop(alt)
                    logger.debug("LLM used '%s' instead of '%s'", alt, canonical)
                    break
    return normalized


# ── Pydantic models ──────────────────────────────────────────────────────────


class ReviewFindingSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    severity: str
    file: str = ""
    line: int | None = None
    message: str = Field(
        default="",
        validation_alias=AliasChoices("message", "description", "desc", "detail"),
    )
    suggestion: str = Field(
        default="",
        validation_alias=AliasChoices("suggestion", "fix", "recommendation", "remediation"),
    )


class ReviewResponse(BaseModel):
    """Schema for code-review JSON.  ``passed`` has **no default** — the LLM
    must explicitly state whether the review passed."""

    passed: bool                                   # Required — no silent True
    summary: str                                   # Required
    findings: list[ReviewFindingSchema] = []


class SecurityVulnerability(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    severity: str
    file: str = ""
    line: int | None = None
    type: str = "unknown"
    description: str = Field(
        default="",
        validation_alias=AliasChoices("description", "message"),
    )
    remediation: str = Field(
        default="",
        validation_alias=AliasChoices("remediation", "fix", "suggestion", "recommendation"),
    )


class SecurityResponse(BaseModel):
    """Schema for security-scan JSON.  ``passed`` has **no default**."""

    passed: bool                                   # Required — no silent True
    summary: str                                   # Required
    vulnerabilities: list[SecurityVulnerability] = []


class FileBlueprintSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    path: str                                      # Required
    purpose: str = Field(
        default="",
        validation_alias=AliasChoices("purpose", "description", "desc", "role"),
    )
    depends_on: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("depends_on", "dependencies", "deps"),
    )
    exports: list[str] = []
    language: str = ""
    layer: str = ""


class ArchitectureResponse(BaseModel):
    name: str = "unnamed-project"
    description: str = ""
    architecture_style: str = "REST"
    tech_stack: dict[str, str] = {}
    folder_structure: list[str] = []
    file_blueprints: list[FileBlueprintSchema] = []
    architecture_doc: str = ""


# ── Validators (normalise → Pydantic) ────────────────────────────────────────


def validate_architecture_response(data: dict[str, Any]) -> ArchitectureResponse:
    """Validate and normalise an architecture blueprint JSON response.

    Raises ``pydantic.ValidationError`` on invalid data.
    """
    normalized = _normalize_keys(data or {})
    return ArchitectureResponse.model_validate(normalized)


def validate_review_response(data: dict[str, Any]) -> ReviewResponse:
    """Validate and normalise a code review JSON response.

    Raises ``pydantic.ValidationError`` when required fields (``passed``,
    ``summary``) are missing — a validation failure is a task failure.
    """
    normalized = _normalize_keys(data or {})
    return ReviewResponse.model_validate(normalized)


def validate_security_response(data: dict[str, Any]) -> SecurityResponse:
    """Validate and normalise a security scan JSON response.

    Raises ``pydantic.ValidationError`` when required fields (``passed``,
    ``summary``) are missing — a validation failure is a task failure.
    """
    normalized = _normalize_keys(data or {})
    return SecurityResponse.model_validate(normalized)
