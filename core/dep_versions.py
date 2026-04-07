"""Curated dependency version database for common frameworks.

LLMs frequently hallucinate incompatible version combinations (e.g.,
Spring Boot 3.x with Java 8 annotations, or mixing JUnit 4 and 5).
This module provides a curated version database that the code generation
agents can reference to ensure compatible version combinations.

Gated behind the ``DEP_VERSION_PINNING`` feature flag.

Usage::

    from core.dep_versions import get_version_set, inject_version_context

    versions = get_version_set("spring-boot-3.2")
    context_str = inject_version_context(tech_stack)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VersionSet:
    """A compatible set of library versions for a framework."""
    name: str
    description: str
    versions: dict[str, str]     # library → version
    constraints: list[str] = field(default_factory=list)  # Human-readable constraints

    def to_prompt_section(self) -> str:
        """Render as a prompt injection for agents."""
        lines = [
            f"PINNED DEPENDENCY VERSIONS ({self.name}):",
            f"  {self.description}",
            "",
        ]
        for lib, ver in sorted(self.versions.items()):
            lines.append(f"  {lib}: {ver}")
        if self.constraints:
            lines.append("")
            lines.append("CONSTRAINTS:")
            for c in self.constraints:
                lines.append(f"  - {c}")
        return "\n".join(lines)


# ── Curated version sets ─────────────────────────────────────────────────────

_VERSION_SETS: dict[str, VersionSet] = {
    # ── Spring Boot 3.x (Java 17+) ──────────────────────────────────────
    "spring-boot-3": VersionSet(
        name="Spring Boot 3.x",
        description="Spring Boot 3.2+ requires Java 17+. Uses Jakarta EE 10 (jakarta.* packages, NOT javax.*).",
        versions={
            "spring-boot": "3.2.5",
            "spring-boot-starter-parent": "3.2.5",
            "java": "17",
            "spring-framework": "6.1.6",
            "spring-security": "6.2.4",
            "spring-data-jpa": "3.2.5",
            "hibernate": "6.4.4.Final",
            "jakarta-ee": "10",
            "junit": "5.10.2",
            "junit-platform": "1.10.2",
            "mockito": "5.11.0",
            "lombok": "1.18.32",
            "mapstruct": "1.5.5.Final",
            "springdoc-openapi": "2.5.0",
            "h2": "2.2.224",
            "postgresql-driver": "42.7.3",
            "flyway": "10.10.0",
            "jjwt": "0.12.5",
        },
        constraints=[
            "MUST use jakarta.* packages (NOT javax.*)",
            "MUST use @SpringBootTest (NOT @RunWith(SpringRunner.class))",
            "Use @ExtendWith(MockitoExtension.class) NOT @RunWith(MockitoJUnitRunner.class)",
            "spring-boot-starter-test includes JUnit 5 — do NOT add junit-vintage-engine",
            "H2 2.x changed IDENTITY to AUTO — use GenerationType.IDENTITY or GenerationType.AUTO",
        ],
    ),

    # ── Spring Boot 2.x (Java 8+) ───────────────────────────────────────
    "spring-boot-2": VersionSet(
        name="Spring Boot 2.x",
        description="Spring Boot 2.7.x — last 2.x line. Uses javax.* packages.",
        versions={
            "spring-boot": "2.7.18",
            "spring-boot-starter-parent": "2.7.18",
            "java": "11",
            "spring-framework": "5.3.31",
            "spring-security": "5.8.9",
            "hibernate": "5.6.15.Final",
            "jakarta-ee": "8",
            "junit": "5.9.3",
            "mockito": "4.11.0",
            "lombok": "1.18.30",
            "h2": "1.4.200",
        },
        constraints=[
            "Uses javax.* packages (NOT jakarta.*)",
            "spring-boot-starter-test includes JUnit 5 by default since 2.2+",
        ],
    ),

    # ── Node.js / Express ────────────────────────────────────────────────
    "express-5": VersionSet(
        name="Express.js 5.x",
        description="Express 5 with modern Node.js (18+).",
        versions={
            "node": "20",
            "express": "5.0.0",
            "typescript": "5.4",
            "jest": "29.7.0",
            "supertest": "6.3.4",
            "helmet": "7.1.0",
            "cors": "2.8.5",
            "dotenv": "16.4.5",
            "jsonwebtoken": "9.0.2",
            "bcryptjs": "2.4.3",
            "prisma": "5.12.1",
            "zod": "3.22.5",
        },
        constraints=[
            "Express 5 uses promise-based error handling — no need for express-async-errors",
            "Use import/export syntax (ESM) with \"type\": \"module\" in package.json",
        ],
    ),

    "express-4": VersionSet(
        name="Express.js 4.x",
        description="Express 4.x — stable production version.",
        versions={
            "node": "18",
            "express": "4.19.2",
            "typescript": "5.4",
            "jest": "29.7.0",
            "supertest": "6.3.4",
            "helmet": "7.1.0",
            "cors": "2.8.5",
        },
        constraints=[
            "Async errors need express-async-errors or manual try/catch wrapping",
        ],
    ),

    # ── Python / FastAPI ─────────────────────────────────────────────────
    "fastapi": VersionSet(
        name="FastAPI",
        description="FastAPI with modern Python (3.11+).",
        versions={
            "python": "3.11",
            "fastapi": "0.111.0",
            "uvicorn": "0.29.0",
            "pydantic": "2.7.1",
            "sqlalchemy": "2.0.29",
            "alembic": "1.13.1",
            "pytest": "8.1.1",
            "pytest-asyncio": "0.23.6",
            "httpx": "0.27.0",
            "python-jose": "3.3.0",
            "passlib": "1.7.4",
            "bcrypt": "4.1.3",
        },
        constraints=[
            "Pydantic v2 uses model_validate() NOT parse_obj(), and model_dump() NOT dict()",
            "SQLAlchemy 2.0 uses select() style NOT legacy Query interface",
            "Use pytest-asyncio for async test functions",
        ],
    ),

    # ── Go ───────────────────────────────────────────────────────────────
    "go-gin": VersionSet(
        name="Go + Gin",
        description="Go 1.22+ with Gin web framework.",
        versions={
            "go": "1.22",
            "gin": "v1.9.1",
            "gorm": "v1.25.9",
            "jwt-go": "v5.2.1",
            "validator": "v10.19.0",
            "testify": "v1.9.0",
        },
        constraints=[
            "Use Go modules (go.mod required)",
            "Use golang-jwt/jwt/v5 (NOT dgrijalva/jwt-go which is deprecated)",
        ],
    ),

    # ── React / Next.js ──────────────────────────────────────────────────
    "nextjs-14": VersionSet(
        name="Next.js 14",
        description="Next.js 14 with App Router and React 18.",
        versions={
            "next": "14.2.3",
            "react": "18.3.1",
            "react-dom": "18.3.1",
            "typescript": "5.4",
            "@testing-library/react": "15.0.7",
            "@testing-library/jest-dom": "6.4.5",
            "tailwindcss": "3.4.3",
            "zustand": "4.5.2",
            "axios": "1.6.8",
        },
        constraints=[
            "App Router: use 'use client' directive for client components",
            "Server Components are the default — no useState/useEffect in server components",
        ],
    ),
}


def get_version_set(name: str) -> VersionSet | None:
    """Look up a curated version set by name."""
    return _VERSION_SETS.get(name)


def detect_version_set(tech_stack: dict[str, str]) -> VersionSet | None:
    """Auto-detect the best matching version set from the tech stack."""
    framework = tech_stack.get("framework", "").lower()
    language = tech_stack.get("language", "").lower()
    java_version = tech_stack.get("java", "")

    # Spring Boot detection
    if "spring" in framework:
        if java_version and int(java_version.split(".")[0]) >= 17:
            return _VERSION_SETS.get("spring-boot-3")
        if "3" in tech_stack.get("spring-boot", ""):
            return _VERSION_SETS.get("spring-boot-3")
        # Default to Spring Boot 3 for new projects
        return _VERSION_SETS.get("spring-boot-3")

    # Express detection
    if "express" in framework:
        return _VERSION_SETS.get("express-4")

    # FastAPI detection
    if "fastapi" in framework:
        return _VERSION_SETS.get("fastapi")

    # Go detection
    if language == "go" or "gin" in framework:
        return _VERSION_SETS.get("go-gin")

    # Next.js detection
    if "next" in framework:
        return _VERSION_SETS.get("nextjs-14")

    return None


def inject_version_context(tech_stack: dict[str, str]) -> str:
    """Generate a version-pinning context string for injection into agent prompts.

    Returns an empty string if no matching version set is found or if
    the DEP_VERSION_PINNING flag is not enabled.
    """
    from core.feature_flags import feature
    if not feature("DEP_VERSION_PINNING"):
        return ""

    version_set = detect_version_set(tech_stack)
    if not version_set:
        return ""

    logger.info("Using pinned version set: %s", version_set.name)
    return version_set.to_prompt_section() + "\n\n"
