"""Lightweight schema validation for LLM JSON responses.

LLMs sometimes return valid JSON with wrong key names, missing fields, or
incorrect types.  These validators catch those issues early with warnings
and provide sensible defaults instead of silently producing empty results.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── Common key-name variants that LLMs produce ──────────────────────────────

_ALIASES: dict[str, list[str]] = {
    "passed": ["pass", "success", "ok", "approved"],
    "findings": ["issues", "problems", "errors", "violations"],
    "vulnerabilities": ["vulns", "vulnerability", "security_issues"],
    "summary": ["description", "overview", "conclusion"],
    "suggestion": ["fix", "recommendation", "remediation"],
    "remediation": ["fix", "suggestion", "recommendation"],
    "file_blueprints": ["files", "blueprints", "file_list"],
    "architecture_style": ["architecture", "arch_style", "style"],
    "tech_stack": ["stack", "technology", "technologies"],
    "folder_structure": ["folders", "structure", "directories"],
    "purpose": ["description", "desc", "role"],
    "depends_on": ["dependencies", "deps"],
}


def _get(data: dict, key: str, expected_type: type, default: Any,
         context: str = "") -> Any:
    """Extract *key* from *data* with type checking and alias fallback."""
    value = data.get(key)

    # Try common LLM aliases when primary key is missing
    if value is None:
        for alt in _ALIASES.get(key, []):
            value = data.get(alt)
            if value is not None:
                logger.debug("%s: LLM used '%s' instead of '%s'", context, alt, key)
                break

    if value is None:
        return default

    if not isinstance(value, expected_type):
        logger.warning(
            "%s: key '%s' expected %s, got %s — using default",
            context, key, expected_type.__name__, type(value).__name__,
        )
        return default

    return value


# ── Validators ───────────────────────────────────────────────────────────────

def validate_architecture_response(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise an architecture blueprint JSON response."""
    ctx = "architect"
    if not data:
        logger.error("%s: LLM returned empty response", ctx)
        return _empty_architecture()

    blueprints_raw = _get(data, "file_blueprints", list, [], ctx)
    file_blueprints = _validate_file_blueprints(blueprints_raw, ctx)

    if not file_blueprints:
        logger.warning("%s: No valid file_blueprints in LLM response", ctx)

    return {
        "name": _get(data, "name", str, "unnamed-project", ctx),
        "description": _get(data, "description", str, "", ctx),
        "architecture_style": _get(data, "architecture_style", str, "REST", ctx),
        "tech_stack": _get(data, "tech_stack", dict, {}, ctx),
        "folder_structure": _get(data, "folder_structure", list, [], ctx),
        "file_blueprints": file_blueprints,
        "architecture_doc": _get(data, "architecture_doc", str, "", ctx),
    }


def validate_review_response(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise a code review JSON response."""
    ctx = "reviewer"
    if not data:
        return {"passed": True, "summary": "", "findings": []}

    findings_raw = _get(data, "findings", list, [], ctx)
    findings = _validate_findings(findings_raw, ctx)

    return {
        "passed": _get(data, "passed", bool, True, ctx),
        "summary": _get(data, "summary", str, "", ctx),
        "findings": findings,
    }


def validate_security_response(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise a security scan JSON response."""
    ctx = "security"
    if not data:
        return {"passed": True, "vulnerabilities": [], "summary": ""}

    vulns_raw = _get(data, "vulnerabilities", list, [], ctx)
    vulns = _validate_vulns(vulns_raw, ctx)

    return {
        "passed": _get(data, "passed", bool, True, ctx),
        "summary": _get(data, "summary", str, "", ctx),
        "vulnerabilities": vulns,
    }


# ── Item-level validators ────────────────────────────────────────────────────

def _validate_findings(items: list, ctx: str) -> list[dict]:
    valid = []
    for i, f in enumerate(items):
        if not isinstance(f, dict):
            logger.warning("%s: finding #%d is not a dict, skipping", ctx, i)
            continue
        valid.append({
            "severity": f.get("severity", "info"),
            "file": f.get("file", ""),
            "line": f.get("line"),
            "message": f.get("message", ""),
            "suggestion": f.get("suggestion", f.get("fix", f.get("remediation", ""))),
        })
    return valid


def _validate_vulns(items: list, ctx: str) -> list[dict]:
    valid = []
    for i, v in enumerate(items):
        if not isinstance(v, dict):
            logger.warning("%s: vulnerability #%d is not a dict, skipping", ctx, i)
            continue
        valid.append({
            "severity": v.get("severity", "medium"),
            "file": v.get("file", ""),
            "line": v.get("line"),
            "type": v.get("type", "unknown"),
            "description": v.get("description", v.get("message", "")),
            "remediation": v.get("remediation", v.get("fix", v.get("suggestion", ""))),
        })
    return valid


def _validate_file_blueprints(items: list, ctx: str) -> list[dict]:
    valid = []
    for i, fb in enumerate(items):
        if not isinstance(fb, dict):
            logger.warning("%s: blueprint #%d is not a dict, skipping", ctx, i)
            continue
        path = fb.get("path")
        if not path:
            logger.warning("%s: blueprint #%d missing 'path', skipping", ctx, i)
            continue
        valid.append({
            "path": path,
            "purpose": fb.get("purpose", fb.get("description", "")),
            "depends_on": fb.get("depends_on", fb.get("dependencies", [])),
            "exports": fb.get("exports", []),
            "language": fb.get("language", ""),
            "layer": fb.get("layer", ""),
        })
    return valid


def _empty_architecture() -> dict[str, Any]:
    return {
        "name": "unnamed-project",
        "description": "",
        "architecture_style": "REST",
        "tech_stack": {},
        "folder_structure": [],
        "file_blueprints": [],
        "architecture_doc": "",
    }
