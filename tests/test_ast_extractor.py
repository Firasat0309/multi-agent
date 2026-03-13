"""Tests for the AST-based signature extractor."""

import pytest
from core.ast_extractor import ASTExtractor


@pytest.fixture
def extractor():
    return ASTExtractor()


# ── Basic Java parsing ──────────────────────────────────────────────────

JAVA_SERVICE = """\
package com.example.service;

import com.example.model.User;
import com.example.repository.UserRepository;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.Optional;

@Service
public class UserService {

    private final UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public User findById(Long id) {
        return userRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("User not found"));
    }

    public List<User> findAll() {
        return userRepository.findAll();
    }

    protected void internalMethod() {
        System.out.println("internal");
    }

    private void validateUser(User user) {
        if (user == null) throw new IllegalArgumentException();
    }
}
"""


class TestJavaClassExtraction:
    def test_extracts_package(self, extractor):
        sig = extractor.extract("UserService.java", JAVA_SERVICE, "java")
        assert sig is not None
        assert sig.package == "com.example.service"

    def test_extracts_imports(self, extractor):
        sig = extractor.extract("UserService.java", JAVA_SERVICE, "java")
        assert len(sig.imports) == 5
        assert any("UserRepository" in imp for imp in sig.imports)

    def test_extracts_class(self, extractor):
        sig = extractor.extract("UserService.java", JAVA_SERVICE, "java")
        assert len(sig.types) == 1
        cls = sig.types[0]
        assert cls.name == "UserService"
        assert cls.kind == "class"
        assert "public" in cls.modifiers

    def test_extracts_annotations(self, extractor):
        sig = extractor.extract("UserService.java", JAVA_SERVICE, "java")
        cls = sig.types[0]
        assert any("@Service" in a for a in cls.annotations)

    def test_extracts_fields(self, extractor):
        sig = extractor.extract("UserService.java", JAVA_SERVICE, "java")
        cls = sig.types[0]
        assert len(cls.fields) == 1
        assert cls.fields[0].name == "userRepository"
        assert cls.fields[0].type_name == "UserRepository"

    def test_extracts_constructor(self, extractor):
        sig = extractor.extract("UserService.java", JAVA_SERVICE, "java")
        cls = sig.types[0]
        constructors = [m for m in cls.methods if m.is_constructor]
        assert len(constructors) == 1
        assert "UserRepository" in constructors[0].parameters

    def test_extracts_all_methods(self, extractor):
        sig = extractor.extract("UserService.java", JAVA_SERVICE, "java")
        cls = sig.types[0]
        method_names = [m.name for m in cls.methods if not m.is_constructor]
        assert "findById" in method_names
        assert "findAll" in method_names
        assert "internalMethod" in method_names
        assert "validateUser" in method_names

    def test_method_return_types(self, extractor):
        sig = extractor.extract("UserService.java", JAVA_SERVICE, "java")
        cls = sig.types[0]
        methods = {m.name: m for m in cls.methods}
        assert methods["findById"].return_type == "User"
        assert methods["findAll"].return_type == "List<User>"
        assert methods["internalMethod"].return_type == "void"


# ── Java interface extraction ───────────────────────────────────────────

JAVA_INTERFACE = """\
package com.example.repository;

import com.example.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;
import java.util.List;

public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByEmail(String email);
    List<User> findByStatus(String status);
}
"""


class TestJavaInterfaceExtraction:
    def test_extracts_interface(self, extractor):
        sig = extractor.extract("UserRepository.java", JAVA_INTERFACE, "java")
        assert len(sig.types) == 1
        iface = sig.types[0]
        assert iface.kind == "interface"
        assert iface.name == "UserRepository"

    def test_extracts_extends(self, extractor):
        sig = extractor.extract("UserRepository.java", JAVA_INTERFACE, "java")
        iface = sig.types[0]
        assert "JpaRepository<User, Long>" in iface.extends

    def test_extracts_interface_methods(self, extractor):
        sig = extractor.extract("UserRepository.java", JAVA_INTERFACE, "java")
        iface = sig.types[0]
        assert len(iface.methods) == 2
        names = [m.name for m in iface.methods]
        assert "findByEmail" in names
        assert "findByStatus" in names


# ── Java enum extraction ────────────────────────────────────────────────

JAVA_ENUM = """\
package com.example.model;

public enum UserStatus {
    ACTIVE,
    INACTIVE,
    SUSPENDED;

    public boolean isActive() {
        return this == ACTIVE;
    }
}
"""


class TestJavaEnumExtraction:
    def test_extracts_enum(self, extractor):
        sig = extractor.extract("UserStatus.java", JAVA_ENUM, "java")
        assert len(sig.types) == 1
        enum = sig.types[0]
        assert enum.kind == "enum"
        assert enum.name == "UserStatus"

    def test_extracts_enum_constants(self, extractor):
        sig = extractor.extract("UserStatus.java", JAVA_ENUM, "java")
        enum = sig.types[0]
        assert "ACTIVE" in enum.enum_constants
        assert "INACTIVE" in enum.enum_constants
        assert "SUSPENDED" in enum.enum_constants

    def test_extracts_enum_methods(self, extractor):
        sig = extractor.extract("UserStatus.java", JAVA_ENUM, "java")
        enum = sig.types[0]
        assert len(enum.methods) == 1
        assert enum.methods[0].name == "isActive"


# ── Java record extraction ──────────────────────────────────────────────

JAVA_RECORD = """\
package com.example.dto;

public record CreateUserRequest(
    String name,
    String email
) {}
"""


class TestJavaRecordExtraction:
    def test_extracts_record(self, extractor):
        sig = extractor.extract("CreateUserRequest.java", JAVA_RECORD, "java")
        assert len(sig.types) == 1
        rec = sig.types[0]
        assert rec.kind == "record"
        assert rec.name == "CreateUserRequest"

    def test_extracts_record_components(self, extractor):
        sig = extractor.extract("CreateUserRequest.java", JAVA_RECORD, "java")
        rec = sig.types[0]
        assert "String name" in rec.record_components
        assert "String email" in rec.record_components


# ── Stub rendering ──────────────────────────────────────────────────────

class TestStubRendering:
    def test_stub_includes_package(self, extractor):
        stub = extractor.extract_stub("UserService.java", JAVA_SERVICE, "java")
        assert stub is not None
        assert "package com.example.service;" in stub

    def test_stub_includes_imports(self, extractor):
        stub = extractor.extract_stub("UserService.java", JAVA_SERVICE, "java")
        assert "import com.example.model.User" in stub

    def test_stub_includes_class_header(self, extractor):
        stub = extractor.extract_stub("UserService.java", JAVA_SERVICE, "java")
        assert "class UserService" in stub

    def test_stub_includes_method_signatures(self, extractor):
        stub = extractor.extract_stub("UserService.java", JAVA_SERVICE, "java")
        assert "findById(Long id)" in stub
        assert "findAll()" in stub

    def test_stub_omits_method_bodies(self, extractor):
        stub = extractor.extract_stub("UserService.java", JAVA_SERVICE, "java")
        assert "orElseThrow" not in stub
        assert "RuntimeException" not in stub

    def test_stub_includes_annotations(self, extractor):
        stub = extractor.extract_stub("UserService.java", JAVA_SERVICE, "java")
        assert "@Service" in stub

    def test_interface_stub_includes_extends(self, extractor):
        stub = extractor.extract_stub("UserRepository.java", JAVA_INTERFACE, "java")
        assert "extends JpaRepository<User, Long>" in stub

    def test_enum_stub_includes_constants(self, extractor):
        stub = extractor.extract_stub("UserStatus.java", JAVA_ENUM, "java")
        assert "ACTIVE" in stub
        assert "INACTIVE" in stub

    def test_stub_is_compact(self, extractor):
        """Stub should be significantly smaller than the original source."""
        stub = extractor.extract_stub("UserService.java", JAVA_SERVICE, "java")
        assert len(stub) < len(JAVA_SERVICE) * 0.8  # at least 20% reduction


# ── Multi-type file ─────────────────────────────────────────────────────

JAVA_MULTI_TYPE = """\
package com.example;

public class MainApp {
    public void run() {}
}

interface Configurable {
    void configure();
}

enum AppMode {
    DEV, PROD;
}
"""


class TestMultiTypeFile:
    def test_extracts_all_types(self, extractor):
        sig = extractor.extract("MainApp.java", JAVA_MULTI_TYPE, "java")
        assert len(sig.types) == 3
        kinds = {t.kind for t in sig.types}
        assert kinds == {"class", "interface", "enum"}


# ── Edge cases ──────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_unsupported_language_returns_none(self, extractor):
        assert extractor.extract("foo.rb", "class Foo; end", "ruby") is None

    def test_empty_source(self, extractor):
        sig = extractor.extract("Empty.java", "", "java")
        assert sig is not None
        assert sig.types == []
        assert sig.package == ""

    def test_is_supported_java(self, extractor):
        assert ASTExtractor.is_supported("java") is True

    def test_is_supported_unknown(self, extractor):
        assert ASTExtractor.is_supported("ruby") is False

    def test_caching_returns_same_object(self, extractor):
        sig1 = extractor.extract("A.java", JAVA_ENUM, "java", checksum="abc123")
        sig2 = extractor.extract("A.java", JAVA_ENUM, "java", checksum="abc123")
        assert sig1 is sig2

    def test_extract_stub_unsupported_returns_none(self, extractor):
        assert extractor.extract_stub("foo.py", "x = 1", "python") is None


# ── Implements clause ───────────────────────────────────────────────────

JAVA_IMPLEMENTS = """\
package com.example;

public class UserServiceImpl implements UserService, Auditable {
    public User findById(Long id) { return null; }
    public void audit(String event) {}
}
"""


class TestImplementsClause:
    def test_extracts_implements(self, extractor):
        sig = extractor.extract("UserServiceImpl.java", JAVA_IMPLEMENTS, "java")
        cls = sig.types[0]
        assert "UserService" in cls.implements
        assert "Auditable" in cls.implements
