"""Edge-case tests for _extract_type_definitions and dep/store signature extraction."""
import pytest
from agents.component_generator_agent import (
    _extract_type_definitions,
    ComponentGeneratorAgent,
)


class TestExtractTypeDefinitionsEdgeCases:

    def test_type_alias_no_braces(self):
        """A simple `type X = 'a' | 'b';` should still be captured as a header."""
        src = 'export type ButtonVariant = "primary" | "secondary";'
        result = _extract_type_definitions(src)
        text = "\n".join(result)
        # The _TYPE_DEF regex matches `type ButtonVariant`, brace_depth stays 0
        # after the header line, so it exits immediately. The header should appear.
        assert "ButtonVariant" in text

    def test_nested_braces_do_not_terminate_early(self):
        """Nested objects inside an interface should NOT close the type early."""
        src = (
            "export interface Config {\n"
            "  nested: {\n"
            "    innerProp: string;\n"
            "  };\n"
            "  topLevel: boolean;\n"
            "}\n"
        )
        result = _extract_type_definitions(src)
        text = "\n".join(result)
        assert "Config" in text
        assert "topLevel" in text  # field after the nested block

    def test_extends_clause(self):
        """'extends' on the interface line should be preserved in the header."""
        src = (
            "export interface ExtendedProps extends BaseProps {\n"
            "  extra: string;\n"
            "}\n"
        )
        result = _extract_type_definitions(src)
        text = "\n".join(result)
        assert "ExtendedProps extends BaseProps" in text
        assert "extra" in text

    def test_type_alias_object(self):
        """type Foo = { bar: string; } should be extracted like an interface."""
        src = (
            "export type Foo = {\n"
            "  bar: string;\n"
            "  baz: number;\n"
            "}\n"
        )
        result = _extract_type_definitions(src)
        text = "\n".join(result)
        assert "Foo" in text
        assert "bar" in text

    def test_class_methods_not_captured_as_fields(self):
        """Class method bodies should not be included."""
        src = (
            "export class MyService {\n"
            "  async getData(id: string): Promise<Data> {\n"
            "    const result = await fetch('/api');\n"
            "    return result.json();\n"
            "  }\n"
            "  name: string;\n"
            "}\n"
        )
        result = _extract_type_definitions(src)
        text = "\n".join(result)
        assert "MyService" in text
        assert "name: string" in text
        # Method body internals like 'fetch' and 'result' should NOT appear
        assert "fetch" not in text
        assert "result.json" not in text

    def test_empty_interface(self):
        """Empty interface {} on a single line."""
        src = "export interface Empty {}"
        result = _extract_type_definitions(src)
        text = "\n".join(result)
        assert "Empty" in text

    def test_brace_on_next_line(self):
        """Opening brace on the next line should still track the type."""
        src = (
            "export interface SplitBrace\n"
            "{\n"
            "  field: string;\n"
            "}\n"
        )
        result = _extract_type_definitions(src)
        text = "\n".join(result)
        assert "SplitBrace" in text
        assert "field" in text

    def test_function_field_types(self):
        """Fields that are arrow function types should be captured."""
        src = (
            "export interface Actions {\n"
            "  login: (email: string, password: string) => Promise<void>;\n"
            "  logout: () => void;\n"
            "}\n"
        )
        result = _extract_type_definitions(src)
        text = "\n".join(result)
        assert "login" in text
        assert "logout" in text

    def test_large_interface_performance(self):
        """500-field interface should still be extracted without issues."""
        fields = "\n".join(f"  field{i}: string;" for i in range(500))
        src = f"export interface Large {{\n{fields}\n}}\n"
        result = _extract_type_definitions(src)
        assert len(result) == 501  # 1 header + 500 fields

    def test_multiple_consecutive_types(self):
        """Two consecutive interfaces should both be extracted."""
        src = (
            "export interface A {\n"
            "  x: number;\n"
            "}\n"
            "\n"
            "export interface B {\n"
            "  y: string;\n"
            "}\n"
        )
        result = _extract_type_definitions(src)
        text = "\n".join(result)
        assert "interface A" in text
        assert "x: number" in text
        assert "interface B" in text
        assert "y: string" in text

    def test_empty_string(self):
        """Empty input returns empty list."""
        assert _extract_type_definitions("") == []

    def test_declare_keyword(self):
        """'declare interface' should be recognized."""
        src = (
            "declare interface GlobalState {\n"
            "  theme: string;\n"
            "}\n"
        )
        result = _extract_type_definitions(src)
        text = "\n".join(result)
        assert "GlobalState" in text

    def test_generic_interface(self):
        """Interface with generics: `interface Response<T>` should capture the name."""
        src = (
            "export interface Response<T> {\n"
            "  data: T;\n"
            "  error: string | null;\n"
            "}\n"
        )
        result = _extract_type_definitions(src)
        text = "\n".join(result)
        # _TYPE_DEF captures (\w+) which will match "Response" before the "<"
        assert "Response" in text
        assert "data" in text


class TestExtractDepSignaturesEdgeCases:

    def test_empty_related_files(self):
        result = ComponentGeneratorAgent._extract_dep_signatures({})
        assert result == ""

    def test_only_store_files_returns_empty(self):
        files = {
            "src/store/useAuth.ts": "export const useAuth = () => {};",
            "src/lib/api.ts": "export const api = {};",
            "src/hooks/useData.ts": "export function useData() {}",
        }
        result = ComponentGeneratorAgent._extract_dep_signatures(files)
        assert result == ""

    def test_mixed_files_only_processes_components(self):
        files = {
            "src/store/useAuth.ts": "export const useAuth = () => {};",
            "src/components/Button.tsx": (
                "export interface ButtonProps {\n"
                "  variant: string;\n"
                "}\n"
            ),
        }
        result = ComponentGeneratorAgent._extract_dep_signatures(files)
        assert "ButtonProps" in result
        assert "useAuth" not in result

    def test_component_with_no_types_or_exports(self):
        """A component file with only internal code should return empty."""
        files = {
            "src/components/Internal.tsx": (
                "function internal() {\n"
                "  return null;\n"
                "}\n"
            ),
        }
        result = ComponentGeneratorAgent._extract_dep_signatures(files)
        assert result == ""

    def test_dep_signatures_captures_export_default_function(self):
        files = {
            "src/components/Card.tsx": (
                "export default function Card({ children }: CardProps) {\n"
                "  return <div>{children}</div>;\n"
                "}\n"
            ),
        }
        result = ComponentGeneratorAgent._extract_dep_signatures(files)
        assert "Card" in result


class TestExtractStoreSignaturesEdgeCases:

    def test_empty_related_files(self):
        result = ComponentGeneratorAgent._extract_store_signatures({})
        assert result == ""

    def test_non_store_files_ignored(self):
        files = {
            "src/components/Button.tsx": "export const Button = () => {};",
        }
        result = ComponentGeneratorAgent._extract_store_signatures(files)
        assert result == ""

    def test_store_with_types_and_actions(self):
        files = {
            "src/store/useCart.ts": (
                "export interface CartItem {\n"
                "  id: string;\n"
                "  quantity: number;\n"
                "}\n\n"
                "export function addItem(item: CartItem): void {\n"
                "}\n"
            ),
        }
        result = ComponentGeneratorAgent._extract_store_signatures(files)
        assert "CartItem" in result
        assert "id: string" in result
        assert "addItem" in result
