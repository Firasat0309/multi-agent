"""Tests for Phase 2 improvements.

Covers:
  - PromptTemplates.compose() and individual fragments
  - Feature flag guards (STREAMING_LLM, TOKEN_BUDGETS, etc.)
  - New error types (ContextOverflowError, TokenBudgetExceededError, BlueprintValidationError)
  - LLM-based compaction (compact_messages_with_summary)
  - Language profile YAML loader (load_language_profiles_yaml)
  - LiveConsole.update_cost
  - LLMClient.on_cost_update callback
  - Blueprint validation (_validate_blueprint)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import ExecutionConfig
from core.errors import (
    BlueprintValidationError,
    ContextOverflowError,
    TokenBudgetExceededError,
)
from core.feature_flags import feature, set_feature
from core.prompt_templates import PromptTemplates


def _run(coro):
    return asyncio.run(coro)


# ── PromptTemplates ──────────────────────────────────────────────────────────


class TestPromptTemplates:
    def test_role_substitution(self):
        result = PromptTemplates.role("code reviewer")
        assert "code reviewer" in result
        assert "expert" in result.lower()

    def test_code_constraints_language(self):
        result = PromptTemplates.code_constraints(language="Java")
        assert "Java" in result

    def test_fix_constraints_remaining(self):
        result = PromptTemplates.fix_constraints(remaining=2)
        assert "2" in result

    def test_output_json(self):
        result = PromptTemplates.output_json()
        assert "JSON" in result
        assert "markdown" in result.lower()

    def test_output_code(self):
        result = PromptTemplates.output_code(language="Python")
        assert "Python" in result

    def test_output_diff(self):
        result = PromptTemplates.output_diff()
        assert "diff" in result.lower()

    def test_security_rules(self):
        result = PromptTemplates.security_rules()
        assert "OWASP" in result

    def test_test_rules_framework(self):
        result = PromptTemplates.test_rules(framework="pytest")
        assert "pytest" in result

    def test_review_rules(self):
        result = PromptTemplates.review_rules()
        assert "critical" in result

    def test_no_hallucination(self):
        result = PromptTemplates.no_hallucination()
        assert "hallucinate" in result.lower()

    def test_compose_joins_fragments(self):
        result = PromptTemplates.compose("part1", "part2", "part3")
        assert result == "part1\n\npart2\n\npart3"

    def test_compose_skips_empty(self):
        result = PromptTemplates.compose("part1", "", "part3")
        assert result == "part1\n\npart3"

    def test_compose_empty(self):
        result = PromptTemplates.compose()
        assert result == ""


# ── Feature Flags ────────────────────────────────────────────────────────────


class TestFeatureFlags:
    def test_default_feature_is_false(self):
        assert feature("NON_EXISTENT_FLAG") is False

    def test_set_and_get(self):
        set_feature("TEST_FLAG_ABC", True)
        assert feature("TEST_FLAG_ABC") is True
        set_feature("TEST_FLAG_ABC", False)
        assert feature("TEST_FLAG_ABC") is False

    def test_known_flags_exist(self):
        """Verify the Phase 2 flag constants are importable."""
        from core.feature_flags import (
            STREAMING_LLM,
            REACTIVE_COMPACTION,
            TOKEN_BUDGETS,
            BLUEPRINT_RETRY,
            OTEL_SPANS,
            COST_WARNING,
        )
        assert STREAMING_LLM == "STREAMING_LLM"
        assert REACTIVE_COMPACTION == "REACTIVE_COMPACTION"
        assert TOKEN_BUDGETS == "TOKEN_BUDGETS"
        assert BLUEPRINT_RETRY == "BLUEPRINT_RETRY"


# ── Error Types ──────────────────────────────────────────────────────────────


class TestErrorTypes:
    def test_context_overflow_error(self):
        err = ContextOverflowError("too big", estimated_tokens=120000)
        assert err.estimated_tokens == 120000
        assert "too big" in str(err)

    def test_token_budget_exceeded_error(self):
        err = TokenBudgetExceededError(
            "over budget", file_path="src/main.py", tokens_used=60000, budget=50000
        )
        assert err.file_path == "src/main.py"
        assert err.tokens_used == 60000
        assert err.budget == 50000

    def test_blueprint_validation_error(self):
        err = BlueprintValidationError(
            "invalid", validation_errors=["no name", "no tech"], raw_output="{}"
        )
        assert len(err.validation_errors) == 2
        assert err.raw_output == "{}"


# ── ExecutionConfig New Fields ───────────────────────────────────────────────


class TestExecutionConfigPhase2:
    def test_new_fields_defaults(self):
        cfg = ExecutionConfig()
        assert cfg.file_token_budget == 50_000
        assert cfg.agent_token_budget == 80_000
        assert cfg.compaction_threshold_tokens == 80_000
        assert cfg.blueprint_max_retries == 3
        assert cfg.cost_warning_pct == 0.8

    def test_new_fields_override(self):
        cfg = replace(
            ExecutionConfig(),
            file_token_budget=30_000,
            blueprint_max_retries=5,
        )
        assert cfg.file_token_budget == 30_000
        assert cfg.blueprint_max_retries == 5


# ── Context Compaction ───────────────────────────────────────────────────────


class TestCompactMessages:
    def test_no_compaction_when_under_budget(self):
        from core.context_compaction import compact_messages

        msgs = [{"role": "user", "content": "hi"}]
        result = compact_messages(msgs, char_budget=1_000_000)
        assert result is msgs  # same reference — no copy

    def test_compaction_preserves_head_and_tail(self):
        from core.context_compaction import compact_messages

        msgs = [
            {"role": "user", "content": "task prompt " * 100},
        ]
        for i in range(20):
            msgs.append({"role": "assistant", "content": f"response {i} " * 50})
            msgs.append({"role": "user", "content": f"follow up {i} " * 50})

        result = compact_messages(msgs, char_budget=2000, keep_tail=4)
        # First message preserved
        assert result[0] == msgs[0]
        # Last 4 preserved
        assert result[-1] == msgs[-1]
        assert result[-4] == msgs[-4]
        # Summary marker in the middle
        assert "[" in result[1]["content"] and "removed" in result[1]["content"]

    def test_llm_compaction_success(self):
        from core.context_compaction import compact_messages_with_summary

        msgs = [{"role": "user", "content": "task " * 200}]
        for i in range(20):
            msgs.append({"role": "assistant", "content": f"reply {i} " * 100})
            msgs.append({"role": "user", "content": f"question {i} " * 100})

        mock_response = MagicMock()
        mock_response.content = "- File A was created\n- Build error fixed\n- Tests added"

        async def mock_generate(**kwargs):
            return mock_response

        result = _run(compact_messages_with_summary(
            msgs, mock_generate, char_budget=1000, keep_tail=4,
        ))
        # Should contain the LLM summary
        assert any("LLM-generated summary" in m.get("content", "") for m in result)

    def test_llm_compaction_fallback_on_error(self):
        from core.context_compaction import compact_messages_with_summary

        msgs = [{"role": "user", "content": "task " * 200}]
        for i in range(20):
            msgs.append({"role": "assistant", "content": f"reply {i} " * 100})

        async def mock_generate_fail(**kwargs):
            raise RuntimeError("LLM unavailable")

        result = _run(compact_messages_with_summary(
            msgs, mock_generate_fail, char_budget=1000, keep_tail=4,
        ))
        # Falls back to simple compaction — still has summary marker
        assert any("removed" in m.get("content", "") for m in result)


# ── Language Profile YAML Loader ─────────────────────────────────────────────


class TestLanguageProfilesYaml:
    def test_load_nonexistent_returns_zero(self):
        from core.language import load_language_profiles_yaml

        count = load_language_profiles_yaml(path="/nonexistent/path.yaml")
        assert count == 0

    def test_load_patch_existing_profile(self):
        from core.language import LANGUAGE_PROFILES, load_language_profiles_yaml

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("python:\n  build_command: 'python -m build'\n")
            f.flush()
            tmp_path = f.name

        try:
            original = LANGUAGE_PROFILES["python"]
            count = load_language_profiles_yaml(path=tmp_path)
            assert count == 1
            assert LANGUAGE_PROFILES["python"].build_command == "python -m build"
        finally:
            # Restore
            LANGUAGE_PROFILES["python"] = original
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_new_profile(self):
        from core.language import LANGUAGE_PROFILES, load_language_profiles_yaml

        yaml_content = (
            "swift:\n"
            "  name: swift\n"
            "  display_name: Swift\n"
            "  file_extensions: ['.swift']\n"
            "  glob_pattern: '**/*.swift'\n"
            "  docker_image: 'swift:5.10'\n"
            "  test_command: 'swift test'\n"
            "  lint_command: 'swiftlint'\n"
            "  type_check_command: ''\n"
            "  security_scan_command: ''\n"
            "  build_command: 'swift build'\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            f.flush()
            tmp_path = f.name

        try:
            count = load_language_profiles_yaml(path=tmp_path)
            assert count == 1
            assert "swift" in LANGUAGE_PROFILES
            assert LANGUAGE_PROFILES["swift"].display_name == "Swift"
        finally:
            LANGUAGE_PROFILES.pop("swift", None)
            Path(tmp_path).unlink(missing_ok=True)


# ── LiveConsole Cost ─────────────────────────────────────────────────────────


class TestLiveConsoleCost:
    def test_update_cost_logs_warning(self):
        from core.live_console import LiveConsole

        lc = LiveConsole()
        lc._start_time = 1.0  # fake start so timestamps work

        with patch.object(lc, "_refresh"):
            lc.update_cost(8.5, 10.0)
            assert lc._cost_spent == 8.5
            assert lc._cost_limit == 10.0
            # 85% → should have logged a warning
            assert any("Cost warning" in entry or "COST" in entry for entry in lc._agent_log)

    def test_update_cost_no_warning_below_threshold(self):
        from core.live_console import LiveConsole

        lc = LiveConsole()
        lc._start_time = 1.0

        with patch.object(lc, "_refresh"):
            lc.update_cost(2.0, 10.0)
            # 20% → no warning
            assert not any("Cost" in entry for entry in lc._agent_log)


# ── LLMClient Cost Callback ─────────────────────────────────────────────────


class TestLLMClientCostCallback:
    def test_on_cost_update_called(self):
        """Verify _check_cost_cap invokes on_cost_update callback."""
        from core.llm_client import LLMClient, LLMConfig, LLMProvider

        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-sonnet-4-20250514",
            api_key="test-key",
        )
        client = LLMClient(config)
        client.max_cost_usd = 10.0
        client.total_input_tokens = 1000
        client.total_output_tokens = 500

        callback = MagicMock()
        client.on_cost_update = callback
        client._check_cost_cap()

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[1] == 10.0  # limit
        assert args[0] >= 0     # spent >= 0


# ── Blueprint Validation ─────────────────────────────────────────────────────


class TestBlueprintValidation:
    def test_empty_blueprint_fails(self):
        from core.models import RepositoryBlueprint
        from core.pipeline_run import RunPipeline

        bp = RepositoryBlueprint(name="", description="", architecture_style="")
        errors = RunPipeline._validate_blueprint(bp)
        assert len(errors) > 0

    def test_valid_blueprint_passes(self):
        from core.models import FileBlueprint, RepositoryBlueprint
        from core.pipeline_run import RunPipeline

        bp = RepositoryBlueprint(
            name="test-app",
            description="test",
            architecture_style="REST",
            tech_stack={"language": "python"},
            file_blueprints=[
                FileBlueprint(path="main.py", purpose="entry point"),
                FileBlueprint(path="utils.py", purpose="helpers", depends_on=["main.py"]),
            ],
        )
        errors = RunPipeline._validate_blueprint(bp)
        assert errors == []

    def test_duplicate_paths_detected(self):
        from core.models import FileBlueprint, RepositoryBlueprint
        from core.pipeline_run import RunPipeline

        bp = RepositoryBlueprint(
            name="test-app",
            description="test",
            architecture_style="REST",
            tech_stack={"language": "python"},
            file_blueprints=[
                FileBlueprint(path="main.py", purpose="entry"),
                FileBlueprint(path="main.py", purpose="duplicate"),
            ],
        )
        errors = RunPipeline._validate_blueprint(bp)
        assert any("duplicate" in e.lower() or "Duplicate" in e for e in errors)

    def test_dangling_dependency_detected(self):
        from core.models import FileBlueprint, RepositoryBlueprint
        from core.pipeline_run import RunPipeline

        bp = RepositoryBlueprint(
            name="test-app",
            description="test",
            architecture_style="REST",
            tech_stack={"language": "python"},
            file_blueprints=[
                FileBlueprint(path="main.py", purpose="entry", depends_on=["missing.py"]),
            ],
        )
        errors = RunPipeline._validate_blueprint(bp)
        assert any("missing.py" in e for e in errors)
