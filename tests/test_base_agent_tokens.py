"""Tests for dynamic max_tokens and truncation detection in BaseAgent."""

import pytest
from unittest.mock import MagicMock

from agents.base_agent import BaseAgent
from core.models import AgentContext, FileBlueprint, RepositoryBlueprint, Task, TaskType


# ── _detect_truncated_code ───────────────────────────────────────────────────

class TestDetectTruncatedCode:
    def test_balanced_java_not_truncated(self):
        code = """\
package com.example;

public class User {
    private String name;

    public String getName() {
        return this.name;
    }
}
"""
        assert BaseAgent._detect_truncated_code(code, "User.java") is None

    def test_unbalanced_java_detected(self):
        code = """\
package com.example;

public class User {
    private String name;

    public String getName() {
        return this.name;
"""
        result = BaseAgent._detect_truncated_code(code, "User.java")
        assert result is not None
        assert "TRUNCATED CODE DETECTED" in result

    def test_typescript_unbalanced(self):
        code = """\
export class UserService {
    async findById(id: number): Promise<User> {
        const user = await this.repo.findOne(id);
"""
        result = BaseAgent._detect_truncated_code(code, "user.service.ts")
        assert result is not None
        assert "TRUNCATED" in result

    def test_typescript_balanced(self):
        code = """\
export class UserService {
    async findById(id: number): Promise<User> {
        return this.repo.findOne(id);
    }
}
"""
        assert BaseAgent._detect_truncated_code(code, "user.service.ts") is None

    def test_go_unbalanced(self):
        code = """\
func (h *Handler) Get(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    if id, ok := vars["id"]; ok {
        user := h.svc.Find(id)
"""
        result = BaseAgent._detect_truncated_code(code, "handler.go")
        assert result is not None

    def test_python_ignored(self):
        # Python doesn't use braces — should return None regardless
        code = "def foo():\n    pass\n"
        assert BaseAgent._detect_truncated_code(code, "foo.py") is None

    def test_single_unclosed_brace_not_flagged(self):
        # One unclosed brace can happen in valid patterns (like lambdas)
        # We only flag 2+ unclosed braces to reduce false positives
        code = """\
public class User {
    Runnable r = () -> {
        System.out.println("hello");
    };
"""
        result = BaseAgent._detect_truncated_code(code, "User.java")
        # Only 1 unclosed brace — should not flag
        assert result is None

    def test_severely_truncated(self):
        code = """\
public class UserService {
    private final UserRepository repo;

    public UserService(UserRepository repo) {
        this.repo = repo;
    }

    public Optional<User> findById(Long id) {
        return repo.findById(id);
    }

    public User save(User user) {
"""
        result = BaseAgent._detect_truncated_code(code, "UserService.java")
        assert result is not None
        assert "unclosed braces" in result

    def test_non_code_file_ignored(self):
        assert BaseAgent._detect_truncated_code("{{{", "data.json") is None
        assert BaseAgent._detect_truncated_code("{{{", "readme.md") is None


# ── _estimate_max_tokens ─────────────────────────────────────────────────────

def _make_context(
    file_path: str = "User.java",
    existing_content: str = "",
    deps: list[str] | None = None,
) -> AgentContext:
    fb = FileBlueprint(
        path=file_path,
        purpose="Test file",
        depends_on=deps or [],
    )
    related = {}
    if existing_content:
        related[file_path] = existing_content
    return AgentContext(
        task=Task(task_id=1, task_type=TaskType.GENERATE_FILE, file=file_path, description="test"),
        blueprint=RepositoryBlueprint(
            name="test", description="test", architecture_style="REST",
            tech_stack={"language": "java"},
            file_blueprints=[fb],
        ),
        file_blueprint=fb,
        related_files=related,
    )


class TestEstimateMaxTokens:
    def test_base_minimum(self):
        ctx = _make_context()
        tokens = BaseAgent._estimate_max_tokens(ctx)
        assert tokens >= 16384

    def test_scales_with_existing_file(self):
        # A large existing file should get more tokens
        large_content = "x" * 50_000  # ~14k tokens
        ctx = _make_context(existing_content=large_content)
        tokens = BaseAgent._estimate_max_tokens(ctx)
        assert tokens > 16384  # Should scale up

    def test_scales_with_deps(self):
        deps = [f"Dep{i}.java" for i in range(6)]
        ctx = _make_context(deps=deps)
        tokens = BaseAgent._estimate_max_tokens(ctx)
        assert tokens >= 24576  # High dep count → larger budget

    def test_small_file_uses_base(self):
        ctx = _make_context(existing_content="class Foo {}")
        tokens = BaseAgent._estimate_max_tokens(ctx)
        assert tokens == 16384  # Small file stays at base

    def test_medium_deps(self):
        deps = ["A.java", "B.java", "C.java"]
        ctx = _make_context(deps=deps)
        tokens = BaseAgent._estimate_max_tokens(ctx)
        assert tokens >= 20480
