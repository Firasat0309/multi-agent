"""Deploy agent - generates Dockerfile and Kubernetes manifests."""

from __future__ import annotations

import logging
import re
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
            "RULES:\n"
            "1. Generate minimal, secure Docker images using multi-stage builds\n"
            "2. Use non-root users in all containers (never run as root)\n"
            "3. Include health checks (HEALTHCHECK in Docker, probes in K8s)\n"
            "4. Set resource limits AND requests on all Kubernetes containers\n"
            "5. Include readiness and liveness probes with appropriate thresholds\n"
            "6. Follow 12-factor app principles (config via env vars, stateless processes)\n"
            "7. Output only the file content — no markdown fences or explanations\n\n"
            "CONSISTENCY RULES:\n"
            "- Use the SAME port number across Dockerfile EXPOSE, docker-compose ports, "
            "K8s containerPort, Service targetPort, and health check probe port\n"
            "- Use the SAME application name for all labels, container names, and service names\n"
            "- Health check paths must match what the application actually exposes\n"
            "- Environment variables must be consistent across docker-compose and K8s ConfigMap"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Generate deployment artifacts."""
        logger.info("Generating deployment artifacts")
        blueprint = context.blueprint
        files_written: list[str] = []

        # Determine app port and health endpoint from tech stack
        app_port, health_path = self._detect_app_settings(blueprint)

        # Generate Dockerfile
        dockerfile = await self._generate_dockerfile(blueprint, app_port, health_path)
        await self.repo.async_write_deploy_file("Dockerfile", dockerfile)
        files_written.append("deploy/Dockerfile")

        # Generate docker-compose.yml
        compose = await self._generate_docker_compose(blueprint, app_port)
        await self.repo.async_write_deploy_file("docker-compose.yml", compose)
        files_written.append("deploy/docker-compose.yml")

        # Generate Kubernetes manifests (all together for consistency)
        k8s_manifests = await self._generate_k8s_manifests(blueprint, app_port, health_path)
        for name, content in k8s_manifests.items():
            await self.repo.async_write_deploy_file(f"k8s/{name}", content)
            files_written.append(f"deploy/k8s/{name}")

        return TaskResult(
            success=True,
            output=f"Generated {len(files_written)} deployment artifacts",
            files_modified=files_written,
            metrics=self.get_metrics(),
        )

    @staticmethod
    def _detect_app_settings(blueprint: RepositoryBlueprint) -> tuple[int, str]:
        """Detect the application port and health endpoint from the tech stack."""
        tech = blueprint.tech_stack
        lang = tech.get("language", "").lower()
        framework = tech.get("framework", "").lower()

        if "java" in lang or "spring" in framework:
            return 8080, "/actuator/health"
        if "go" in lang:
            return 8080, "/health"
        if "typescript" in lang or "node" in lang or "express" in framework or "next" in framework or "react" in framework or "vue" in framework or "angular" in framework:
            return 3000, "/health"
        if "python" in lang:
            if "django" in framework:
                return 8000, "/health/"
            return 8000, "/health"
        if "rust" in lang:
            return 8080, "/health"
        if "csharp" in lang or "dotnet" in framework:
            return 5000, "/health"
        return 8080, "/health"

    async def _generate_dockerfile(
        self, blueprint: RepositoryBlueprint, app_port: int, health_path: str,
    ) -> str:
        tech = blueprint.tech_stack
        lang = tech.get("language", "unknown").lower()
        framework = tech.get("framework", "unknown")
        build_tool = tech.get("build_tool", "")

        prompt = (
            f"Project: {blueprint.name}\n"
            f"Language: {lang}\n"
            f"Framework: {framework}\n"
            f"Build tool: {build_tool}\n"
            f"Application port: {app_port}\n"
            f"Health endpoint: {health_path}\n\n"
            "Generate a production Dockerfile with these EXACT requirements:\n\n"
            "STAGE 1 — Build:\n"
            f"- Use the official {lang} SDK/build image as the builder stage\n"
            "- Copy dependency manifests first (for layer caching), then install dependencies\n"
            "- Copy source code and build the application\n"
            f"- Build output dir: {'dist/' if 'vue' in framework or 'vite' in build_tool else '.next/' if 'next' in framework else 'build/'}\n\n"
            "STAGE 2 — Runtime:\n"
            f"- Use a minimal runtime image (e.g., nginx:alpine for static SPAs like Vue/React, node:alpine for Next.js SSR)\n"
            "- Copy ONLY the built artifact from the builder stage\n"
            "- Create a non-root user and switch to it: RUN addgroup -S app && adduser -S app -G app\n"
            "- USER app\n"
            f"- EXPOSE {app_port}\n"
            f"- HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD wget -qO- http://localhost:{app_port}{health_path} || exit 1\n"
            "- Set the proper entrypoint/CMD for the application\n\n"
            "SECURITY:\n"
            "- No secrets or credentials in the image\n"
            "- Pin base image versions (not :latest)\n"
            "- Use .dockerignore patterns if relevant\n\n"
            "Output ONLY the Dockerfile content — no explanations."
        )
        return await self._call_llm(prompt)

    async def _generate_docker_compose(
        self, blueprint: RepositoryBlueprint, app_port: int,
    ) -> str:
        tech = blueprint.tech_stack
        db = tech.get("db", "").lower()

        # Determine which services are needed based on actual tech stack
        db_service = ""
        if db:
            if "postgres" in db:
                db_service = "PostgreSQL (image: postgres:16-alpine, port 5432)"
            elif "mysql" in db or "mariadb" in db:
                db_service = "MySQL (image: mysql:8, port 3306)"
            elif "mongo" in db:
                db_service = "MongoDB (image: mongo:7, port 27017)"
            elif "redis" in db:
                db_service = "Redis (image: redis:7-alpine, port 6379)"
            elif "h2" in db or "sqlite" in db:
                db_service = ""  # Embedded, no service needed

        prompt = (
            f"Project: {blueprint.name}\n"
            f"Tech stack: {tech}\n"
            f"Application port: {app_port}\n\n"
            "Generate a docker-compose.yml for local development with:\n\n"
            f"1. Application service ('{blueprint.name}'):\n"
            f"   - Build from ./Dockerfile\n"
            f"   - Port mapping: {app_port}:{app_port}\n"
            "   - Environment variables for database connection and app config\n"
            "   - depends_on the database service (if any)\n"
            "   - Restart policy: unless-stopped\n"
        )
        if db_service:
            prompt += (
                f"\n2. Database service ('{db}'):\n"
                f"   - {db_service}\n"
                "   - Named volume for data persistence\n"
                "   - Environment variables for root/default credentials\n"
            )
        prompt += (
            "\nNETWORKING:\n"
            "- Use a custom bridge network\n"
            "- Service names must be usable as hostnames by the app\n\n"
            "RULES:\n"
            "- Use version '3.8' or later\n"
            "- Include a volumes section for database persistence\n"
            f"- The application's DB host env var must reference the database service name\n"
            "- Output ONLY the YAML content — no explanations"
        )
        return await self._call_llm(prompt)

    async def _generate_k8s_manifests(
        self, blueprint: RepositoryBlueprint, app_port: int, health_path: str,
    ) -> dict[str, str]:
        """Generate ALL Kubernetes manifests in a single LLM call for consistency."""
        tech = blueprint.tech_stack
        name = blueprint.name

        prompt = (
            f"Project: {name}\n"
            f"Tech stack: {tech}\n"
            f"Architecture: {blueprint.architecture_style}\n"
            f"Application port: {app_port}\n"
            f"Health endpoint: {health_path}\n\n"
            "Generate ALL Kubernetes manifests as a single YAML stream separated by '---'.\n"
            "Include exactly these resources in this order:\n\n"
            f"1. ConfigMap — named '{name}-config' with application settings as data entries\n\n"
            f"2. Deployment — named '{name}' with:\n"
            "   - 2 replicas\n"
            f"   - Label: app={name} on ALL resources (metadata.labels and selector)\n"
            "   - Resource limits: 256Mi memory, 500m CPU\n"
            "   - Resource requests: 128Mi memory, 250m CPU\n"
            f"   - Readiness probe: HTTP GET {health_path} port {app_port}, "
            "initialDelaySeconds: 10, periodSeconds: 5\n"
            f"   - Liveness probe: HTTP GET {health_path} port {app_port}, "
            "initialDelaySeconds: 30, periodSeconds: 10, failureThreshold: 3\n"
            f"   - envFrom referencing '{name}-config' ConfigMap\n"
            f"   - containerPort: {app_port}\n"
            "   - Rolling update strategy: maxSurge=1, maxUnavailable=0\n\n"
            f"3. Service — named '{name}-service', type ClusterIP:\n"
            f"   - Selector: app={name}\n"
            f"   - Port 80 → targetPort {app_port}\n\n"
            f"4. HPA — named '{name}-hpa':\n"
            f"   - scaleTargetRef: Deployment/{name}\n"
            "   - minReplicas: 2, maxReplicas: 10\n"
            "   - targetCPUUtilizationPercentage: 70\n\n"
            "CONSISTENCY RULES:\n"
            f"- Use app label: app={name} on ALL resources\n"
            f"- Use port {app_port} consistently across Deployment containerPort, "
            "Service targetPort, and probe ports\n"
            f"- The HPA must reference the exact Deployment name: {name}\n"
            "- Output ONLY the YAML — no markdown fences, no explanations"
        )

        combined = await self._call_llm(prompt)
        return self._split_yaml_stream(combined, name)

    @staticmethod
    def _split_yaml_stream(combined: str, project_name: str) -> dict[str, str]:
        """Split a combined YAML stream into named manifest files."""
        # Clean any markdown fences
        combined = combined.strip()
        if combined.startswith("```"):
            first_nl = combined.find("\n")
            if first_nl != -1:
                combined = combined[first_nl + 1:]
        if combined.endswith("```"):
            combined = combined[:-3].strip()

        documents = re.split(r"\n---\s*\n", combined)
        manifest_map: dict[str, str] = {}

        kind_to_file = {
            "configmap": "configmap.yaml",
            "deployment": "deployment.yaml",
            "service": "service.yaml",
            "horizontalpodautoscaler": "hpa.yaml",
        }

        for doc in documents:
            doc = doc.strip()
            if not doc:
                continue
            # Detect the kind from the YAML
            kind_match = re.search(r"kind:\s*(\w+)", doc, re.IGNORECASE)
            if kind_match:
                kind = kind_match.group(1).lower()
                filename = kind_to_file.get(kind, f"{kind}.yaml")
            else:
                # Fallback: assign incrementing names
                filename = f"resource-{len(manifest_map) + 1}.yaml"
            manifest_map[filename] = doc + "\n"

        # Ensure all expected files exist (fallback if split failed)
        if not manifest_map:
            manifest_map["deployment.yaml"] = combined

        return manifest_map
