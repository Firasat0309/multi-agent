"""API Contract Agent — generates an OpenAPI contract shared by FE and BE."""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.models import (
    AgentContext,
    AgentRole,
    APIContract,
    APIEndpoint,
    ProductRequirements,
    RepositoryBlueprint,
    TaskResult,
)

logger = logging.getLogger(__name__)


class APIContractAgent(BaseAgent):
    """Produces a formal API contract (OpenAPI 3.x) from the product requirements
    and the backend blueprint.

    The contract is the handshake between the frontend and backend pipelines:
    the frontend uses it to generate typed API calls, the backend uses it to
    validate route implementations.
    """

    role = AgentRole.API_CONTRACT_GENERATOR

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior API design agent. Your job is to produce an OpenAPI 3.x\n"
            "contract from product requirements and a backend architecture blueprint.\n\n"
            "Respond with a JSON object matching this schema:\n"
            "{\n"
            '  "title": "Project API",\n'
            '  "version": "1.0.0",\n'
            '  "base_url": "/api/v1",\n'
            '  "contract_format": "openapi",\n'
            '  "endpoints": [\n'
            "    {\n"
            '      "path": "/users",\n'
            '      "method": "GET",\n'
            '      "description": "List all users",\n'
            '      "request_schema": {},\n'
            '      "response_schema": {"type": "array", "items": {"$ref": "#/components/schemas/User"}},\n'
            '      "auth_required": true,\n'
            '      "tags": ["users"]\n'
            "    }\n"
            "  ],\n"
            '  "schemas": {\n'
            '    "User": {\n'
            '      "type": "object",\n'
            '      "properties": {"id": {"type": "integer"}, "email": {"type": "string"}}\n'
            "    }\n"
            "  },\n"
            '  "openapi_spec": "openapi: 3.0.3\\ninfo:\\n  title: ..."\n'
            "}\n\n"
            "Rules:\n"
            "- Cover every CRUD operation implied by the backend blueprint.\n"
            "- Use RESTful path conventions (/resources, /resources/{id}).\n"
            "- Mark auth_required=true for all non-public endpoints.\n"
            "- Populate 'schemas' with reusable request/response models.\n"
            "- The 'openapi_spec' field must contain a valid OpenAPI 3.x YAML string.\n"
            "- Output ONLY the JSON object — no markdown code fences."
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        requirements: ProductRequirements | None = context.task.metadata.get("requirements")
        blueprint: RepositoryBlueprint = context.blueprint
        try:
            contract = await self.generate_contract(requirements, blueprint)
            return TaskResult(
                success=True,
                output=f"API contract generated: {contract.title} "
                       f"({len(contract.endpoints)} endpoints)",
                metrics={
                    **self.get_metrics(),
                    "endpoints": len(contract.endpoints),
                    "schemas": len(contract.schemas),
                },
            )
        except Exception as exc:
            logger.exception("APIContractAgent.execute failed")
            return TaskResult(success=False, errors=[str(exc)])

    async def generate_contract(
        self,
        requirements: ProductRequirements | None,
        blueprint: RepositoryBlueprint,
    ) -> APIContract:
        """Generate the API contract from requirements + backend blueprint."""
        logger.info("APIContractAgent: generating API contract")

        req_text = ""
        if requirements:
            req_text = (
                f"Product: {requirements.title}\n"
                f"Description: {requirements.description}\n"
                f"Features: {', '.join(requirements.features)}\n"
            )
            if requirements.user_stories:
                req_text += "User stories:\n"
                for story in requirements.user_stories:
                    req_text += f"  - {story}\n"

        # Include file purposes, exports, and layers so the LLM can infer
        # accurate endpoints rather than guessing from file names alone.
        file_details = "\n".join(
            f"  - {fb.path} [{fb.layer}]: {fb.purpose}"
            + (f"  exports: {', '.join(fb.exports[:6])}" if fb.exports else "")
            for fb in blueprint.file_blueprints
        )
        bp_text = (
            f"Backend: {blueprint.name}\n"
            f"Style: {blueprint.architecture_style}\n"
            f"Tech stack: {json.dumps(blueprint.tech_stack)}\n"
            f"Files:\n{file_details}\n"
        )

        raw = await self._call_llm_json(
            f"{req_text}\n{bp_text}\n\nGenerate the complete API contract JSON now."
        )
        self._metrics["llm_calls"] += 1
        return self._parse_contract(raw)

    # ─────────────────────────────────────────────────────────────────────────

    # NOTE: _parse_contract is intentionally a @staticmethod so that
    # load_from_file (a classmethod) can call it without instantiating
    # the agent (which requires llm_client and repo_manager).

    @classmethod
    def load_from_file(cls, path: "str | Path") -> APIContract:
        """Load and parse an api_contract.json file into an APIContract object.

        Accepts the same JSON format that the agent writes to disk, so a
        user-supplied contract file works as a direct pipeline input.

        Raises:
            FileNotFoundError: if the path does not exist.
            json.JSONDecodeError: if the file is not valid JSON.
            ValueError: if the file has no endpoints.
        """
        from pathlib import Path as _Path
        p = _Path(path)
        if not p.exists():
            raise FileNotFoundError(f"api_contract.json not found: {p}")
        raw = json.loads(p.read_text(encoding="utf-8"))
        parsed = cls._parse_contract(raw)
        if not parsed.endpoints:
            raise ValueError(
                f"api_contract.json at {p} has 0 endpoints — "
                "check that 'endpoints' array is populated."
            )
        logger.info(
            "Loaded api_contract.json: %s (%d endpoints, %d schemas)",
            parsed.title, len(parsed.endpoints), len(parsed.schemas),
        )
        return parsed

    @staticmethod
    def _parse_contract(raw: dict[str, Any]) -> APIContract:
        endpoints = [
            APIEndpoint(
                path=str(ep.get("path", "/")),
                method=str(ep.get("method", "GET")).upper(),
                description=str(ep.get("description", "")),
                request_schema=ep.get("request_schema", {}),
                response_schema=ep.get("response_schema", {}),
                auth_required=bool(ep.get("auth_required", False)),
                tags=[str(t) for t in ep.get("tags", [])],
            )
            for ep in raw.get("endpoints", [])
            if isinstance(ep, dict)
        ]
        return APIContract(
            title=str(raw.get("title", "API")),
            version=str(raw.get("version", "1.0.0")),
            base_url=str(raw.get("base_url", "/api/v1")),
            endpoints=endpoints,
            schemas=raw.get("schemas", {}),
            openapi_spec=str(raw.get("openapi_spec", "")),
            contract_format=str(raw.get("contract_format", "openapi")),
        )
