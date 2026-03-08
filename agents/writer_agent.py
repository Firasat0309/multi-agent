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
            "You are a technical writing agent. You generate clear, comprehensive "
            "documentation for software projects.\n\n"
            "Rules:\n"
            "- Write clear, professional documentation\n"
            "- Use proper Markdown formatting\n"
            "- Include code examples where helpful\n"
            "- Document API endpoints with request/response examples\n"
            "- Include setup and deployment instructions\n"
            "- Output only the document content"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Generate project documentation."""
        logger.info("Generating documentation")
        blueprint = context.blueprint
        files_written: list[str] = []

        # README
        readme = await self._generate_readme(blueprint)
        self.repo.write_doc_file("README.md", readme)
        files_written.append("docs/README.md")

        # API documentation
        api_docs = await self._generate_api_docs(blueprint, context)
        self.repo.write_doc_file("API.md", api_docs)
        files_written.append("docs/API.md")

        # Changelog
        changelog = self._generate_changelog(blueprint)
        self.repo.write_doc_file("CHANGELOG.md", changelog)
        files_written.append("docs/CHANGELOG.md")

        return TaskResult(
            success=True,
            output=f"Generated {len(files_written)} documentation files",
            files_modified=files_written,
            metrics=self.get_metrics(),
        )

    async def _generate_readme(self, blueprint: RepositoryBlueprint) -> str:
        prompt = (
            f"Project: {blueprint.name}\n"
            f"Description: {blueprint.description}\n"
            f"Architecture: {blueprint.architecture_style}\n"
            f"Tech stack: {blueprint.tech_stack}\n"
            f"Folder structure: {blueprint.folder_structure}\n\n"
            "Generate a comprehensive README.md including:\n"
            "- Project overview\n"
            "- Features\n"
            "- Prerequisites\n"
            "- Installation and setup\n"
            "- Running locally\n"
            "- Running tests\n"
            "- Docker deployment\n"
            "- Kubernetes deployment\n"
            "- Project structure\n"
            "- Contributing guidelines"
        )
        return await self._call_llm(prompt)

    async def _generate_api_docs(
        self, blueprint: RepositoryBlueprint, context: AgentContext
    ) -> str:
        # Collect controller files for endpoint documentation
        controller_files = {
            path: content
            for path, content in context.related_files.items()
            if "controller" in path or "router" in path
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
            f"{controllers_section}\n\n"
            "Generate comprehensive API documentation including:\n"
            "- Base URL and authentication\n"
            "- All endpoints with HTTP methods\n"
            "- Request/response examples with JSON\n"
            "- Error codes and handling\n"
            "- Rate limiting information"
        )
        return await self._call_llm(prompt)

    def _generate_changelog(self, blueprint: RepositoryBlueprint) -> str:
        return (
            f"# Changelog\n\n"
            f"## [0.1.0] - Initial Release\n\n"
            f"### Added\n"
            f"- Initial {blueprint.name} implementation\n"
            f"- {blueprint.architecture_style} API\n"
            f"- Database integration\n"
            f"- Docker and Kubernetes deployment\n"
            f"- Comprehensive test suite\n"
            f"- API documentation\n"
        )
