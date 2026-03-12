"""Tests for language profile system."""

import pytest

from core.language import (
    PYTHON, JAVA, GO, TYPESCRIPT, RUST, CSHARP,
    LanguageProfile,
    get_language_profile,
    detect_language_from_blueprint,
)


class TestLanguageProfiles:
    def test_all_profiles_have_required_fields(self):
        for profile in [PYTHON, JAVA, GO, TYPESCRIPT, RUST, CSHARP]:
            assert profile.name
            assert profile.display_name
            assert profile.file_extensions
            assert profile.glob_pattern
            assert profile.docker_image
            assert profile.test_command
            assert profile.code_fence_name
            assert profile.allowed_commands

    def test_python_profile(self):
        assert PYTHON.file_extensions == [".py"]
        assert PYTHON.package_init_file == "__init__.py"
        assert PYTHON.module_separator == "."
        assert "pytest" in PYTHON.test_command

    def test_java_profile(self):
        assert JAVA.file_extensions == [".java"]
        assert JAVA.package_init_file == ""
        assert "mvn" in JAVA.test_command
        assert "temurin" in JAVA.docker_image

    def test_go_profile(self):
        assert GO.file_extensions == [".go"]
        assert GO.module_separator == "/"
        assert "go test" in GO.test_command

    def test_typescript_profile(self):
        assert TYPESCRIPT.file_extensions == [".ts"]
        assert "jest" in TYPESCRIPT.test_command
        assert "node" in TYPESCRIPT.docker_image


class TestGetLanguageProfile:
    def test_exact_match(self):
        assert get_language_profile("python") is PYTHON
        assert get_language_profile("java") is JAVA
        assert get_language_profile("go") is GO
        assert get_language_profile("typescript") is TYPESCRIPT
        assert get_language_profile("rust") is RUST
        assert get_language_profile("csharp") is CSHARP

    def test_aliases(self):
        assert get_language_profile("golang") is GO
        assert get_language_profile("ts") is TYPESCRIPT
        assert get_language_profile("rs") is RUST
        assert get_language_profile("c#") is CSHARP

    def test_case_insensitive(self):
        assert get_language_profile("Python") is PYTHON
        assert get_language_profile("JAVA") is JAVA

    def test_framework_detection(self):
        assert get_language_profile("spring") is JAVA
        assert get_language_profile("express") is TYPESCRIPT
        assert get_language_profile("gin") is GO
        assert get_language_profile("actix") is RUST
        assert get_language_profile("asp.net") is CSHARP

    def test_unknown_defaults_to_python(self):
        assert get_language_profile("unknown") is PYTHON
        assert get_language_profile("") is PYTHON


class TestDetectLanguageFromBlueprint:
    def test_detect_from_language_key(self):
        assert detect_language_from_blueprint({"language": "java"}) is JAVA
        assert detect_language_from_blueprint({"language": "go"}) is GO

    def test_detect_from_framework_key(self):
        assert detect_language_from_blueprint({"framework": "spring"}) is JAVA
        assert detect_language_from_blueprint({"framework": "express"}) is TYPESCRIPT

    def test_defaults_to_python(self):
        assert detect_language_from_blueprint({}) is PYTHON
        assert detect_language_from_blueprint({"db": "postgresql"}) is PYTHON


class TestLanguageProfileMethods:
    def test_matches_extension(self):
        assert PYTHON.matches_extension("models/user.py")
        assert not PYTHON.matches_extension("models/User.java")
        assert JAVA.matches_extension("models/User.java")
        assert GO.matches_extension("handlers/user.go")
        assert TYPESCRIPT.matches_extension("components/Button.ts")
        assert TYPESCRIPT.matches_extension("components/Button.tsx")

    def test_to_module_path(self):
        assert PYTHON.to_module_path("models/user.py") == "models.user"
        assert JAVA.to_module_path("com/example/User.java") == "com.example.User"
        assert GO.to_module_path("handlers/user.go") == "handlers/user"
        assert TYPESCRIPT.to_module_path("services/user.ts") == "services/user"
