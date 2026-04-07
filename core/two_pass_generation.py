"""Two-pass generation for complex files.

Complex files (services, controllers with many dependencies) benefit from
a two-pass approach:

  Pass 1 — SKELETON: Generate class structure with method signatures,
           field declarations, and constructor injection. Bodies are stubs.
  Pass 2 — IMPLEMENTATION: Fill in method bodies using the skeleton as
           context. The skeleton ensures all signatures are locked in before
           implementation, reducing cross-method consistency issues.

This is especially effective for files with 5+ dependencies (COMPLEX tier)
where the LLM is most likely to hallucinate method names or lose track of
types across a large file.

Gated behind the ``TWO_PASS_GENERATION`` feature flag.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import AgentContext, FileBlueprint
    from core.language import LanguageProfile

logger = logging.getLogger(__name__)


def should_use_two_pass(fb: FileBlueprint) -> bool:
    """Determine if a file should use two-pass generation.

    Criteria: COMPLEX tier heuristics from model_router —
    ≥5 dependencies, or service/controller with ≥3 deps and complex purpose.
    """
    dep_count = len(fb.depends_on)
    layer = fb.layer.lower() if fb.layer else ""
    purpose = fb.purpose.lower() if fb.purpose else ""

    _complex_signals = (
        "algorithm", "transaction", "concurren", "stream",
        "websocket", "caching", "pagination", "authentication",
        "authorization", "middleware", "interceptor", "security",
    )
    has_complex_purpose = any(s in purpose for s in _complex_signals)

    # Same logic as ModelRouter COMPLEX tier
    if dep_count >= 5:
        return True
    if has_complex_purpose and dep_count >= 3:
        return True
    if layer in ("controller", "service") and dep_count >= 4:
        return True

    return False


def build_skeleton_prompt(
    fb: FileBlueprint,
    language: str,
    dep_signatures: str,
    api_contract_section: str,
) -> str:
    """Build the Pass 1 prompt: generate skeleton with signatures only.

    The skeleton includes:
    - Package/module declaration
    - All imports
    - Class declaration with fields and constructor
    - All method signatures with return types
    - Method bodies contain only TODO comments or minimal stubs
    """
    return (
        f"Generate a SKELETON for: {fb.path}\n"
        f"Purpose: {fb.purpose}\n"
        f"Layer: {fb.layer}\n"
        f"Must export: {', '.join(fb.exports) if fb.exports else 'appropriate classes/functions'}\n\n"
        f"{api_contract_section}"
        f"{dep_signatures}"
        "SKELETON RULES (Pass 1 of 2):\n"
        "1. Include the COMPLETE class structure: package, imports, class header, fields, constructor\n"
        "2. Define ALL method signatures with correct parameter types and return types\n"
        "3. Method bodies should be MINIMAL stubs (throw new UnsupportedOperationException() "
        "or 'raise NotImplementedError()' or 'return null/None')\n"
        "4. EVERY import must be correct — copy from dependency stubs exactly\n"
        "5. EVERY field must be declared with the correct type from the dependency\n"
        "6. Constructor injection must wire all dependencies correctly\n"
        "7. Use EXACT method names from dependency AST stubs\n\n"
        f"Output the complete skeleton via write_file with path='{fb.path}'."
    )


def build_implementation_prompt(
    fb: FileBlueprint,
    skeleton_content: str,
    language: str,
    dep_signatures: str,
    api_contract_section: str,
) -> str:
    """Build the Pass 2 prompt: fill in method implementations.

    The LLM receives the skeleton as context and fills in real implementations.
    This ensures method signatures stay consistent because they're already fixed.
    """
    return (
        f"Complete the implementation of: {fb.path}\n"
        f"Purpose: {fb.purpose}\n"
        f"Layer: {fb.layer}\n\n"
        f"CURRENT SKELETON (keep ALL method signatures EXACTLY as shown):\n"
        f"```\n{skeleton_content}\n```\n\n"
        f"{api_contract_section}"
        f"{dep_signatures}"
        "IMPLEMENTATION RULES (Pass 2 of 2):\n"
        "1. Keep ALL method signatures EXACTLY as they appear in the skeleton above\n"
        "2. Replace every stub body with a COMPLETE, WORKING implementation\n"
        "3. Do NOT add, remove, rename, or change the signature of any method\n"
        "4. Do NOT change imports, fields, or constructor — they are already correct\n"
        "5. Every method must have proper error handling\n"
        "6. Every method must return the correct type\n"
        "7. Use dependency methods exactly as shown in the dependency stubs\n\n"
        f"Output the COMPLETE file (not just the changed parts) via write_file "
        f"with path='{fb.path}'."
    )
