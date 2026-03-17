"""Tests for language detection and config-file routing in coder/architect agents."""

from unittest.mock import MagicMock

from agents.architect_agent import ArchitectAgent
from agents.coder_agent import CoderAgent, _config_format
from core.models import FileBlueprint, RepositoryBlueprint, Task, TaskType
from core.models import AgentContext


# ── _config_format detection ──────────────────────────────────────────────────

class TestConfigFormatDetection:
    def test_properties_detected(self):
        assert _config_format("src/main/resources/application.properties") == \
            "Java Spring Boot application.properties format (key=value pairs, no code)"

    def test_yaml_detected(self):
        assert _config_format("config/application.yaml") == "YAML format"
        assert _config_format("docker-compose.yml") == "YAML format"

    def test_xml_detected(self):
        # pom.xml has a specific format name via _BUILD_CONFIG_NAMES
        fmt = _config_format("pom.xml")
        assert fmt is not None and "pom" in fmt.lower() or "xml" in fmt.lower()
        assert _config_format("src/main/resources/beans.xml") == "XML format"

    def test_dockerfile_detected(self):
        assert _config_format("Dockerfile") == "Dockerfile"
        assert _config_format("Dockerfile.dev") == "Dockerfile"

    def test_json_detected(self):
        # package.json has a specific format name via _BUILD_CONFIG_NAMES
        fmt = _config_format("package.json")
        assert fmt is not None and ("package.json" in fmt.lower() or "json" in fmt.lower())

    def test_sql_detected(self):
        assert _config_format("schema.sql") == "SQL"

    def test_source_files_return_none(self):
        assert _config_format("src/UserService.java") is None
        assert _config_format("handlers/user.go") is None
        assert _config_format("services/user.ts") is None
        assert _config_format("models/user.py") is None
        assert _config_format("src/main.rs") is None


# ── _resolve_file_language ─────────────────────────────────────────────────────

class TestResolveFileLanguage:
    def make_agent(self):
        return ArchitectAgent(llm_client=MagicMock(), repo_manager=MagicMock())

    def test_java_extension_always_java(self):
        agent = self.make_agent()
        # Even if LLM said "python", extension wins
        assert agent._resolve_file_language("python", "src/User.java", "java") == "java"

    def test_go_extension_always_go(self):
        agent = self.make_agent()
        assert agent._resolve_file_language("", "handlers/user.go", "python") == "go"

    def test_ts_extension_always_typescript(self):
        agent = self.make_agent()
        assert agent._resolve_file_language("python", "services/user.ts", "java") == "typescript"

    def test_properties_file_gets_project_lang(self):
        agent = self.make_agent()
        # No extension override → falls through to project lang
        result = agent._resolve_file_language("", "application.properties", "java")
        assert result == "java"

    def test_yaml_file_gets_project_lang(self):
        agent = self.make_agent()
        result = agent._resolve_file_language("", "application.yaml", "go")
        assert result == "go"

    def test_config_file_uses_project_lang(self):
        agent = self.make_agent()
        # Non-source files always use project language regardless of LLM tag
        result = agent._resolve_file_language("java", "pom.xml", "java")
        assert result == "java"
        # Even if LLM says something else, project lang wins for config files
        result = agent._resolve_file_language("xml", "pom.xml", "java")
        assert result == "java"

    def test_project_lang_fallback_when_llm_says_python(self):
        agent = self.make_agent()
        # LLM incorrectly tagged a Java config file as "python"
        result = agent._resolve_file_language("python", "application.properties", "java")
        assert result == "java"


# ── CoderAgent routing ─────────────────────────────────────────────────────────

def make_context(path: str, language: str = "java") -> AgentContext:
    fb = FileBlueprint(path=path, purpose="test", language=language)
    bp = RepositoryBlueprint(
        name="test", description="", architecture_style="REST",
        tech_stack={"language": language},
        file_blueprints=[fb],
    )
    task = Task(task_id=1, task_type=TaskType.GENERATE_FILE, file=path, description="gen")
    return AgentContext(task=task, blueprint=bp, file_blueprint=fb)


class TestCoderAgentRouting:
    def _make_coder(self, written: list) -> CoderAgent:
        repo = MagicMock()
        repo.write_file.side_effect = lambda path, content: written.append((path, content))
        llm = MagicMock()
        return CoderAgent(llm_client=llm, repo_manager=repo)

    def test_java_source_uses_java_system_prompt(self):
        agent = self._make_coder([])
        prompt = agent._get_source_system_prompt("java")
        assert "Java" in prompt
        assert "Python" not in prompt

    def test_config_system_prompt_mentions_format(self):
        agent = self._make_coder([])
        prompt = agent._get_config_system_prompt("Java Spring Boot application.properties format (key=value pairs, no code)")
        assert "no code" in prompt or "configuration" in prompt.lower()

    def test_clean_fences_strips_java_fence(self):
        agent = self._make_coder([])
        raw = "```java\npublic class Foo {}\n```"
        assert agent._clean_fences(raw, "java") == "public class Foo {}\n"

    def test_clean_fences_strips_generic_fence(self):
        agent = self._make_coder([])
        raw = "```\nkey=value\n```"
        assert agent._clean_fences(raw, "") == "key=value\n"

    def test_clean_fences_passthrough_plain_content(self):
        agent = self._make_coder([])
        raw = "server.port=8080\nspring.datasource.url=jdbc:postgresql://localhost/db"
        assert agent._clean_fences(raw, "") == raw + "\n"
