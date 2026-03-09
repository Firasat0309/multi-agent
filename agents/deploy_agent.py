"""Deploy agent - generates Dockerfile and Kubernetes manifests."""

from __future__ import annotations

import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.models import AgentContext, AgentRole, RepositoryBlueprint, TaskResult

logger = logging.getLogger(__name__)


class DeployAgent(BaseAgent):
    role = AgentRole.DEPLOYER

    @property
    def system_prompt(self) -> str:
        return (
            "You are a DevOps/deployment agent. You generate production-ready "
            "Docker and Kubernetes configurations.\n\n"
            "Rules:\n"
            "- Generate minimal, secure Docker images (multi-stage builds)\n"
            "- Use non-root users in containers\n"
            "- Include health checks\n"
            "- Generate Kubernetes Deployment, Service, ConfigMap, and HPA\n"
            "- Use resource limits and requests\n"
            "- Include readiness and liveness probes\n"
            "- Follow 12-factor app principles\n"
            "- Output only the file content, no markdown fences or explanations"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Generate deployment artifacts."""
        logger.info("Generating deployment artifacts")
        blueprint = context.blueprint
        files_written: list[str] = []

        # Generate Dockerfile
        dockerfile = await self._generate_dockerfile(blueprint)
        await self.repo.async_write_deploy_file("Dockerfile", dockerfile)
        files_written.append("deploy/Dockerfile")

        # Generate docker-compose.yml
        compose = await self._generate_docker_compose(blueprint)
        await self.repo.async_write_deploy_file("docker-compose.yml", compose)
        files_written.append("deploy/docker-compose.yml")

        # Generate Kubernetes manifests
        k8s_manifests = await self._generate_k8s_manifests(blueprint)
        for name, content in k8s_manifests.items():
            await self.repo.async_write_deploy_file(f"k8s/{name}", content)
            files_written.append(f"deploy/k8s/{name}")

        return TaskResult(
            success=True,
            output=f"Generated {len(files_written)} deployment artifacts",
            files_modified=files_written,
            metrics=self.get_metrics(),
        )

    async def _generate_dockerfile(self, blueprint: RepositoryBlueprint) -> str:
        prompt = (
            f"Project: {blueprint.name}\n"
            f"Tech stack: {blueprint.tech_stack}\n"
            f"Architecture: {blueprint.architecture_style}\n\n"
            "Generate a production Dockerfile with:\n"
            "- Multi-stage build\n"
            "- Non-root user\n"
            "- Health check\n"
            "- Minimal image size\n\n"
            "Output only the Dockerfile content."
        )
        return await self._call_llm(prompt)

    async def _generate_docker_compose(self, blueprint: RepositoryBlueprint) -> str:
        prompt = (
            f"Project: {blueprint.name}\n"
            f"Tech stack: {blueprint.tech_stack}\n\n"
            "Generate a docker-compose.yml for local development with:\n"
            "- Application service\n"
            "- Database service (if applicable)\n"
            "- Redis service (if applicable)\n"
            "- Proper networking and volumes\n\n"
            "Output only the YAML content."
        )
        return await self._call_llm(prompt)

    async def _generate_k8s_manifests(self, blueprint: RepositoryBlueprint) -> dict[str, str]:
        manifests: dict[str, str] = {}

        # Deployment
        prompt = (
            f"Project: {blueprint.name}\n"
            f"Tech stack: {blueprint.tech_stack}\n\n"
            "Generate a Kubernetes Deployment manifest with:\n"
            "- Resource limits/requests\n"
            "- Readiness and liveness probes\n"
            "- Rolling update strategy\n"
            "- Environment variables from ConfigMap/Secret\n\n"
            "Output only the YAML content."
        )
        manifests["deployment.yaml"] = await self._call_llm(prompt)

        # Service
        prompt = (
            f"Project: {blueprint.name}\n\n"
            "Generate a Kubernetes Service manifest (ClusterIP type).\n"
            "Output only the YAML content."
        )
        manifests["service.yaml"] = await self._call_llm(prompt)

        # ConfigMap
        prompt = (
            f"Project: {blueprint.name}\n"
            f"Tech stack: {blueprint.tech_stack}\n\n"
            "Generate a Kubernetes ConfigMap with common application settings.\n"
            "Output only the YAML content."
        )
        manifests["configmap.yaml"] = await self._call_llm(prompt)

        # HPA
        prompt = (
            f"Project: {blueprint.name}\n\n"
            "Generate a Kubernetes HorizontalPodAutoscaler (min 2, max 10 replicas, "
            "target 70% CPU).\nOutput only the YAML content."
        )
        manifests["hpa.yaml"] = await self._call_llm(prompt)

        return manifests
