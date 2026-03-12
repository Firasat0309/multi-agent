"""Agent implementations for the multi-agent code generation platform."""

from agents.repository_analyzer_agent import RepositoryAnalyzerAgent
from agents.change_planner_agent import ChangePlannerAgent
# ── Fullstack / Frontend agents ───────────────────────────────────────────────
from agents.product_planner_agent import ProductPlannerAgent
from agents.api_contract_agent import APIContractAgent
from agents.design_parser_agent import DesignParserAgent
from agents.component_planner_agent import ComponentPlannerAgent
from agents.component_dag_agent import ComponentDAGAgent
from agents.component_generator_agent import ComponentGeneratorAgent
from agents.api_integration_agent import APIIntegrationAgent
from agents.state_management_agent import StateManagementAgent

__all__ = [
    "RepositoryAnalyzerAgent",
    "ChangePlannerAgent",
    # Fullstack
    "ProductPlannerAgent",
    "APIContractAgent",
    "DesignParserAgent",
    "ComponentPlannerAgent",
    "ComponentDAGAgent",
    "ComponentGeneratorAgent",
    "APIIntegrationAgent",
    "StateManagementAgent",
]
