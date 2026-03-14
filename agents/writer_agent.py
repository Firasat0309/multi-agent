"""Writer agent - generates documentation (README, changelog, API docs)."""

from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from core.models import AgentContext, AgentRole, RepositoryBlueprint, TaskResult

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    role = AgentRole.WRITER

    @property
    def system_prompt(self) -> str:
        return (
            "You are a technical documentation agent. You generate clear, accurate, "
            "project-specific documentation for software projects.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. Write clear, professional documentation in Markdown format\n"
            "2. Use ONLY information from the project details provided — do NOT invent "
            "endpoints, commands, file paths, or features that are not in the project\n"
            "3. Include code examples where helpful, using the project's actual tech stack\n"
            "4. Use the correct build/run/test commands for the project's language and framework\n"
            "5. Output only the document content — no markdown fences wrapping the whole document\n\n"
            "ACCURACY RULES:\n"
            "- Every endpoint documented must exist in the controller/router code shown to you\n"
            "- Every command must be the correct command for the project's build tool\n"
            "- Every file path must match an actual file in the project structure\n"
            "- Port numbers must match what the application configuration specifies\n"
            "- Do NOT hallucinate features, endpoints, or configuration that is not present"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Generate project documentation."""
        logger.info("Generating documentation")
        blueprint = context.blueprint
        files_written: list[str] = []

        # README
        readme = await self._generate_readme(blueprint)
        await self.repo.async_write_doc_file("README.md", readme)
        files_written.append("docs/README.md")

        # API documentation
        api_docs = await self._generate_api_docs(blueprint, context)
        await self.repo.async_write_doc_file("API.md", api_docs)
        files_written.append("docs/API.md")

        # Changelog
        changelog = self._generate_changelog(blueprint)
        await self.repo.async_write_doc_file("CHANGELOG.md", changelog)
        files_written.append("docs/CHANGELOG.md")

        return TaskResult(
            success=True,
            output=f"Generated {len(files_written)} documentation files",
            files_modified=files_written,
            metrics=self.get_metrics(),
        )

    async def _generate_readme(self, blueprint: RepositoryBlueprint) -> str:
        tech = blueprint.tech_stack
        lang = tech.get("language", "unknown")
        framework = tech.get("framework", "unknown")
        db = tech.get("db", "unknown")

        files_list = "\n".join(
            f"  - {fb.path} — {fb.purpose}" for fb in blueprint.file_blueprints
        )

        # Build tech-stack-specific command hints
        cmd_hints = self._get_command_hints(lang, framework)

        prompt = (
            f"Generate a README.md for this specific project.\n\n"
            f"Project name: {blueprint.name}\n"
            f"Description: {blueprint.description}\n"
            f"Language: {lang}\n"
            f"Framework: {framework}\n"
            f"Database: {db}\n"
            f"Architecture style: {blueprint.architecture_style}\n"
            f"Folder structure: {blueprint.folder_structure}\n\n"
            f"Actual project files:\n{files_list}\n\n"
            f"Build/run commands for this stack:\n{cmd_hints}\n\n"

            "The README MUST contain these exact sections (use ## headings):\n"
            "## Overview — 2-3 sentences about what this project does\n"
            "## Prerequisites — exact versions needed (e.g., Java 17, Node 18, Python 3.11)\n"
            "## Quick Start — numbered steps to clone, install dependencies, and run\n"
            "## API Endpoints — table with columns: Method | Path | Description\n"
            "## Project Structure — tree showing actual folders and key files from above\n"
            "## Running Tests — exact command to run the test suite\n"
            "## Configuration — list of config files and key environment variables\n"
            "## Docker — how to build and run with Docker\n\n"

            "RULES:\n"
            "- Use ONLY information from the project details above\n"
            "- Do NOT invent endpoints, commands, or files that are not listed\n"
            "- Use the EXACT build/run/test commands from the command hints above\n"
            "- Port numbers and URLs must match the project's actual configuration\n"
            "- Keep the README under 200 lines"
        )
        return await self._call_llm(prompt)

    async def _generate_api_docs(
        self, blueprint: RepositoryBlueprint, context: AgentContext
    ) -> str:
        # Collect controller/endpoint files for documentation
        controller_files = {
            path: content
            for path, content in context.related_files.items()
            if any(kw in path.lower() for kw in ("controller", "router", "handler", "endpoint", "route", "api"))
        }
        controllers_section = ""
        if controller_files:
            controllers_section = "\n\nController files:\n"
            for path, content in controller_files.items():
                # Detect fence language from file extension
                fence = "python"
                for ext, name in {".java": "java", ".go": "go", ".ts": "typescript",
                                  ".rs": "rust", ".cs": "csharp"}.items():
                    if path.endswith(ext):
                        fence = name
                        break
                controllers_section += f"\n### {path}\n```{fence}\n{content[:3000]}\n```\n"

        prompt = (
            f"Project: {blueprint.name}\n"
            f"Architecture: {blueprint.architecture_style}\n"
            f"Tech stack: {blueprint.tech_stack}\n"
            f"{controllers_section}\n\n"
            "Generate API documentation with these exact sections:\n"
            "## Base URL — the default local URL and port\n"
            "## Authentication — how authentication works (or state if none)\n"
            "## Endpoints — for EACH endpoint found in the controller code above:\n"
            "  - HTTP method and path\n"
            "  - Description of what it does\n"
            "  - Request body JSON example (if applicable)\n"
            "  - Success response JSON example with status code\n"
            "  - Error response JSON example with status code\n"
            "## Error Codes — table of status codes and their meanings\n\n"
            "RULES:\n"
            "- Document ONLY endpoints that exist in the controller code shown above\n"
            "- Use realistic JSON examples matching the actual model fields\n"
            "- Do NOT invent endpoints that are not in the code"
        )
        return await self._call_llm(prompt)

    def _generate_changelog(self, blueprint: RepositoryBlueprint) -> str:
        tech = blueprint.tech_stack
        return (
            f"# Changelog\n\n"
            f"All notable changes to {blueprint.name} will be documented in this file.\n\n"
            f"## [0.1.0] - Initial Release\n\n"
            f"### Added\n"
            f"- Initial {blueprint.name} implementation\n"
            f"- {blueprint.architecture_style} API with {tech.get('framework', 'unknown')} framework\n"
            f"- {tech.get('db', 'Database')} integration\n"
            f"- Docker and Kubernetes deployment configuration\n"
            f"- Unit and integration test suite\n"
            f"- API documentation and README\n"
        )

    @staticmethod
    def _get_command_hints(lang: str, framework: str) -> str:
        """Return tech-stack-specific build/run/test command hints."""
        lang_lower = lang.lower()
        fw_lower = framework.lower()
        if "java" in lang_lower:
            if "gradle" in fw_lower:
                return (
                    "- Build: ./gradlew build\n"
                    "- Run: ./gradlew bootRun\n"
                    "- Test: ./gradlew test"
                )
            return (
                "- Build: mvn clean package\n"
                "- Run: mvn spring-boot:run\n"
                "- Test: mvn test"
            )
        if "typescript" in lang_lower or "node" in fw_lower:
            return (
                "- Install: npm install\n"
                "- Run: npm start\n"
                "- Test: npm test"
            )
        if "go" in lang_lower:
            return (
                "- Build: go build ./...\n"
                "- Run: go run .\n"
                "- Test: go test ./..."
            )
        if "python" in lang_lower:
            if "fastapi" in fw_lower:
                return (
                    "- Install: pip install -r requirements.txt\n"
                    "- Run: uvicorn main:app --reload\n"
                    "- Test: pytest"
                )
            if "django" in fw_lower:
                return (
                    "- Install: pip install -r requirements.txt\n"
                    "- Run: python manage.py runserver\n"
                    "- Test: python manage.py test"
                )
            return (
                "- Install: pip install -r requirements.txt\n"
                "- Run: python -m uvicorn main:app\n"
                "- Test: pytest"
            )
        if "rust" in lang_lower:
            return (
                "- Build: cargo build\n"
                "- Run: cargo run\n"
                "- Test: cargo test"
            )
        if "csharp" in lang_lower or "c#" in lang_lower:
            return (
                "- Build: dotnet build\n"
                "- Run: dotnet run\n"
                "- Test: dotnet test"
            )
        return "- Refer to the project's build tool documentation"
