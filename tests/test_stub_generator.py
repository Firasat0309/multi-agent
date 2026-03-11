"""Tests for the stub generator."""

import tempfile
from pathlib import Path

import pytest
from core.stub_generator import StubGenerator


class TestStubGenerator:
    def _make_generator(self, lang: str, workspace: Path) -> StubGenerator:
        return StubGenerator(lang, workspace)

    def test_java_stub_with_package(self, tmp_path):
        gen = self._make_generator("java", tmp_path)
        created = gen.generate_stubs(["src/main/java/com/example/services/UserService.java"])
        assert len(created) == 1
        content = (tmp_path / "src/main/java/com/example/services/UserService.java").read_text()
        assert "package com.example.services;" in content
        assert "public class UserService {" in content

    def test_java_interface_stub(self, tmp_path):
        gen = self._make_generator("java", tmp_path)
        gen.generate_stubs(["src/main/java/com/example/UserRepository.java"])
        content = (tmp_path / "src/main/java/com/example/UserRepository.java").read_text()
        assert "public interface UserRepository {" in content

    def test_go_stub(self, tmp_path):
        gen = self._make_generator("go", tmp_path)
        created = gen.generate_stubs(["pkg/handlers/user.go"])
        assert len(created) == 1
        content = (tmp_path / "pkg/handlers/user.go").read_text()
        assert "package handlers" in content

    def test_typescript_stub(self, tmp_path):
        gen = self._make_generator("typescript", tmp_path)
        gen.generate_stubs(["src/services/UserService.ts"])
        content = (tmp_path / "src/services/UserService.ts").read_text()
        assert "export class UserService {}" in content

    def test_rust_stub(self, tmp_path):
        gen = self._make_generator("rust", tmp_path)
        gen.generate_stubs(["src/user_service.rs"])
        content = (tmp_path / "src/user_service.rs").read_text()
        assert "pub struct UserService;" in content

    def test_csharp_stub(self, tmp_path):
        gen = self._make_generator("csharp", tmp_path)
        gen.generate_stubs(["Services/UserService.cs"])
        content = (tmp_path / "Services/UserService.cs").read_text()
        assert "namespace Services" in content
        assert "public class UserService" in content

    def test_skip_existing_file(self, tmp_path):
        # Create a file that already exists
        target = tmp_path / "src" / "Model.java"
        target.parent.mkdir(parents=True)
        target.write_text("existing content")

        gen = self._make_generator("java", tmp_path)
        created = gen.generate_stubs(["src/Model.java"])
        assert created == []
        assert target.read_text() == "existing content"

    def test_cleanup_stubs(self, tmp_path):
        gen = self._make_generator("java", tmp_path)
        created = gen.generate_stubs(["src/Foo.java"])
        assert len(created) == 1
        assert (tmp_path / "src/Foo.java").exists()

        gen.cleanup_stubs(created)
        assert not (tmp_path / "src/Foo.java").exists()

    def test_unsupported_language_returns_nothing(self, tmp_path):
        gen = self._make_generator("haskell", tmp_path)
        created = gen.generate_stubs(["src/Main.hs"])
        assert created == []

    def test_multiple_stubs(self, tmp_path):
        gen = self._make_generator("java", tmp_path)
        files = [
            "src/main/java/com/example/User.java",
            "src/main/java/com/example/Order.java",
            "src/main/java/com/example/IPayment.java",
        ]
        created = gen.generate_stubs(files)
        assert len(created) == 3
        # IPayment should be an interface
        content = (tmp_path / "src/main/java/com/example/IPayment.java").read_text()
        assert "public interface IPayment {" in content


class TestScopeCommand:
    """Tests for BuildCheckpoint._scope_command (module-scoped builds)."""

    def test_maven_module_scoping(self):
        from core.checkpoint import BuildCheckpoint
        result = BuildCheckpoint._scope_command("mvn compile", "user-service")
        assert result == "mvn compile -pl user-service"

    def test_gradle_module_scoping(self):
        from core.checkpoint import BuildCheckpoint
        result = BuildCheckpoint._scope_command("gradle build", "user-module")
        assert result == "gradle build -p user-module"

    def test_cargo_module_scoping(self):
        from core.checkpoint import BuildCheckpoint
        result = BuildCheckpoint._scope_command("cargo build", "my_crate")
        assert result == "cargo build -p my_crate"

    def test_go_module_scoping(self):
        from core.checkpoint import BuildCheckpoint
        result = BuildCheckpoint._scope_command("go build ./...", "pkg/handlers")
        assert result == "go build ./pkg/handlers/..."

    def test_dotnet_module_scoping(self):
        from core.checkpoint import BuildCheckpoint
        result = BuildCheckpoint._scope_command("dotnet build", "MyProject/")
        assert result == "dotnet build --project MyProject/"

    def test_no_module_returns_original(self):
        from core.checkpoint import BuildCheckpoint
        result = BuildCheckpoint._scope_command("mvn compile", None)
        assert result == "mvn compile"

    def test_unknown_tool_appends_path(self):
        from core.checkpoint import BuildCheckpoint
        result = BuildCheckpoint._scope_command("bazel build", "//pkg:target")
        assert result == "bazel build //pkg:target"
