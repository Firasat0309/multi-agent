import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.change_planner_agent import ChangePlannerAgent
from core.models import ModuleInfo, RepoAnalysis


def _make_repo_analysis() -> RepoAnalysis:
    return RepoAnalysis(
        modules=[
            ModuleInfo(
                name="users",
                file="src/users.py",
                classes=["UserService"],
                functions=["create_user"],
                imports=["typing"],
                layer="service",
            ),
        ],
        tech_stack={"language": "python"},
        architecture_style="layered",
        entry_points=["src/main.py"],
        summary="User service application",
    )


@pytest.mark.anyio
async def test_plan_changes_uses_cache_on_second_call(tmp_path):
    repo = MagicMock()
    repo.workspace = tmp_path
    repo.read_source_files.return_value = {}

    agent = ChangePlannerAgent(llm_client=MagicMock(), repo_manager=repo)
    agent._call_llm = AsyncMock(return_value=json.dumps({
        "summary": "Add user email validation",
        "changes": [
            {
                "type": "modify_function",
                "file": "src/users.py",
                "description": "Validate email before save",
                "function": "create_user",
                "class_name": "UserService",
                "depends_on": [],
            },
        ],
        "new_files": [],
        "affected_tests": ["tests/test_users.py"],
        "risk_notes": [],
    }))

    first = await agent.plan_changes("Add email validation", _make_repo_analysis())
    second = await agent.plan_changes("Add email validation", _make_repo_analysis())

    assert first.summary == "Add user email validation"
    assert second.summary == first.summary
    assert agent._call_llm.await_count == 1
    assert repo.read_source_files.call_count == 1


@pytest.mark.anyio
async def test_revise_changes_uses_distinct_cache_key(tmp_path):
    repo = MagicMock()
    repo.workspace = tmp_path
    repo.read_source_files.return_value = {}

    agent = ChangePlannerAgent(llm_client=MagicMock(), repo_manager=repo)
    agent._call_llm = AsyncMock(side_effect=[
        json.dumps({
            "summary": "Initial plan",
            "changes": [],
            "new_files": [],
            "affected_tests": [],
            "risk_notes": [],
        }),
        json.dumps({
            "summary": "Revised plan",
            "changes": [],
            "new_files": [],
            "affected_tests": [],
            "risk_notes": ["Needs migration"],
        }),
    ])

    current = await agent.plan_changes("Add email validation", _make_repo_analysis())
    revised = await agent.revise_changes(
        "Add email validation",
        _make_repo_analysis(),
        current,
        "Also add a migration",
    )
    cached = await agent.revise_changes(
        "Add email validation",
        _make_repo_analysis(),
        current,
        "Also add a migration",
    )

    assert revised.summary == "Revised plan"
    assert cached.summary == revised.summary
    assert agent._call_llm.await_count == 2
