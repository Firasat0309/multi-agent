"""Tests for BaseAgent._strip_leading_prose — stripping LLM chain-of-thought from code."""

import pytest

from agents.base_agent import BaseAgent


class TestStripLeadingProse:
    """Verify that LLM deliberation text is stripped before actual code."""

    def test_java_no_prose(self):
        """Clean Java file should be returned unchanged."""
        code = (
            "package com.example.auth.controller;\n\n"
            "import org.springframework.web.bind.annotation.RestController;\n\n"
            "@RestController\n"
            "public class AuthController {\n}\n"
        )
        assert BaseAgent._strip_leading_prose(code, "AuthController.java") == code

    def test_java_with_chain_of_thought(self):
        """Chain-of-thought before Java package statement should be stripped."""
        prose = (
            'lacks a method body in a non-abstract class."\n'
            "        It doesn't say *which* class. But since I'm writing `AuthController`, I'll put it there.\n\n"
            "    *   Wait, I'll check the `serialVersionUID` rule again.\n"
            '        "serialVersionUID must be initialised"\n'
            "    *   Okay, I'm ready.\n\n"
        )
        code = (
            "package com.example.auth.controller;\n\n"
            "import org.springframework.web.bind.annotation.RestController;\n\n"
            "@RestController\n"
            "public class AuthController implements Serializable {\n}\n"
        )
        result = BaseAgent._strip_leading_prose(prose + code, "AuthController.java")
        assert result == code

    def test_java_with_comment_header(self):
        """A Java file starting with a comment should NOT be stripped."""
        code = (
            "// Copyright 2024 Example Corp.\n"
            "package com.example;\n\n"
            "public class Foo {}\n"
        )
        assert BaseAgent._strip_leading_prose(code, "Foo.java") == code

    def test_java_with_javadoc_header(self):
        """A Java file starting with Javadoc should NOT be stripped."""
        code = (
            "/**\n * Main application class.\n */\n"
            "package com.example;\n\n"
            "public class Main {}\n"
        )
        assert BaseAgent._strip_leading_prose(code, "Main.java") == code

    def test_python_with_prose(self):
        """Chain-of-thought before Python import should be stripped."""
        prose = "Let me check the requirements again.\n\nOkay, I'm ready.\n\n"
        code = "from flask import Flask\n\napp = Flask(__name__)\n"
        result = BaseAgent._strip_leading_prose(prose + code, "app.py")
        assert result == code

    def test_python_clean(self):
        """Clean Python file should be unchanged."""
        code = "import os\n\ndef main():\n    pass\n"
        assert BaseAgent._strip_leading_prose(code, "main.py") == code

    def test_typescript_with_prose(self):
        """Chain-of-thought before TypeScript import should be stripped."""
        prose = "I'll implement the component now. Wait, let me check the types.\n\n"
        code = "import React from 'react';\n\nexport const App = () => <div />;\n"
        result = BaseAgent._strip_leading_prose(prose + code, "App.tsx")
        assert result == code

    def test_go_with_prose(self):
        """Chain-of-thought before Go package statement should be stripped."""
        prose = "I need to check the interface. I'll use `Handler` from net/http.\n\n"
        code = "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"hello\")\n}\n"
        result = BaseAgent._strip_leading_prose(prose + code, "main.go")
        assert result == code

    def test_csharp_with_prose(self):
        """Chain-of-thought before C# using statement should be stripped."""
        prose = "Wait, I should check the namespace. Okay, ready.\n\n"
        code = "using System;\n\nnamespace Example\n{\n    class Program {}\n}\n"
        result = BaseAgent._strip_leading_prose(prose + code, "Program.cs")
        assert result == code

    def test_unknown_extension_unchanged(self):
        """Files with unknown extensions should be returned unchanged."""
        content = "some random text\nmore text\n"
        assert BaseAgent._strip_leading_prose(content, "data.dat") == content

    def test_empty_content(self):
        """Empty or very short content should be returned as-is."""
        assert BaseAgent._strip_leading_prose("", "Foo.java") == ""
        assert BaseAgent._strip_leading_prose("short", "Foo.java") == "short"

    def test_prose_without_indicators_not_stripped(self):
        """Leading text without LLM deliberation markers should NOT be stripped."""
        # This has text before the package statement but no LLM markers
        content = (
            "Apache License 2.0 - see LICENSE file.\n\n"
            "package com.example;\n\npublic class Foo {}\n"
        )
        # No prose indicators → left intact
        assert BaseAgent._strip_leading_prose(content, "Foo.java") == content

    def test_xml_with_prose(self):
        """Chain-of-thought before XML declaration should be stripped."""
        prose = "I'll create the pom.xml now. Let me check the dependencies.\n\n"
        code = '<?xml version="1.0" encoding="UTF-8"?>\n<project>\n</project>\n'
        result = BaseAgent._strip_leading_prose(prose + code, "pom.xml")
        assert result == code

    def test_kotlin_with_prose(self):
        """Chain-of-thought before Kotlin package should be stripped."""
        prose = "Wait, I need to check the data class fields.\n\n"
        code = "package com.example\n\ndata class User(val name: String)\n"
        result = BaseAgent._strip_leading_prose(prose + code, "User.kt")
        assert result == code

    def test_rust_with_prose(self):
        """Chain-of-thought before Rust use statement should be stripped."""
        prose = "I'll implement the struct. Let me verify the trait bounds.\n\n"
        code = "use std::io;\n\nfn main() {\n    println!(\"hello\");\n}\n"
        result = BaseAgent._strip_leading_prose(prose + code, "main.rs")
        assert result == code

    def test_real_world_authcontroller_contamination(self):
        """Reproduce the exact AuthController contamination from the bug report."""
        contaminated = (
            'lacks a method body in a non-abstract class."\n'
            "        It doesn't say *which* class. But since I'm writing `AuthController`, I'll put it there.\n\n"
            "    *   Wait, I'll check the `serialVersionUID` rule again.\n"
            '        "serialVersionUID must be initialised: `private static final long serialVersionUID = 1L;`"\n'
            "        I'll put it in the controller.\n\n"
            '    *   Wait, I\'ll check the "Every field/variable declaration MUST end with exactly ONE semicolon" rule.\n'
            "        I'll make sure I use `;`.\n\n"
            "    *   Okay, I'm ready.\n\n"
            "package com.example.auth.controller;\n\n"
            "import org.springframework.web.bind.annotation.RestController;\n"
            "import org.springframework.web.bind.annotation.RequestMapping;\n"
            "import org.springframework.web.bind.annotation.PostMapping;\n"
            "import org.springframework.web.bind.annotation.RequestBody;\n"
            "import org.springframework.http.ResponseEntity;\n"
            "import org.springframework.http.HttpStatus;\n"
            "import com.example.auth.service.AuthService;\n"
            "import com.example.auth.dto.SignupRequest;\n"
            "import com.example.auth.dto.LoginRequest;\n"
            "import com.example.auth.dto.AuthResponse;\n"
            "import java.io.Serializable;\n\n"
            "@RestController\n"
            '@RequestMapping("/api/auth")\n'
            "public class AuthController implements Serializable {\n\n"
            "    private static final long serialVersionUID = 1L;\n"
            "}\n"
        )
        result = BaseAgent._strip_leading_prose(
            contaminated,
            "src/main/java/com/example/auth/controllers/AuthController.java",
        )
        assert result.startswith("package com.example.auth.controller;")
        assert "I'll" not in result
        assert "Wait," not in result
