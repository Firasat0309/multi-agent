"""Standard tool definitions available to all agents via the tool-use loop."""

from __future__ import annotations

from typing import Any

from core.llm_client import ToolDefinition

try:
    from pydantic import BaseModel, ValidationError, field_validator
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False

# ── Standard agent tools ─────────────────────────────────────────────────────
# All tools follow the JSON Schema format required by the Claude tool_use API.

READ_FILE_TOOL = ToolDefinition(
    name="read_file",
    description=(
        "Read the content of a file in the workspace. "
        "Files larger than 150 lines are returned in chunks — the response header "
        "shows the total line count so you can request subsequent ranges with "
        "start_line / end_line.  Omit both to read from line 1."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path to the file (e.g. 'src/models/user.py').",
            },
            "start_line": {
                "type": "integer",
                "description": "1-based line to start reading from (inclusive). Defaults to 1.",
            },
            "end_line": {
                "type": "integer",
                "description": "1-based line to stop reading at (inclusive). Defaults to start_line + 149.",
            },
        },
        "required": ["path"],
    },
)

SEARCH_CODE_TOOL = ToolDefinition(
    name="search_code",
    description=(
        "Search for a symbol, pattern, or string across all source files in the workspace. "
        "Returns matching lines with file path and line number."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Regex pattern or literal string to search for.",
            },
            "file_pattern": {
                "type": "string",
                "description": "Optional glob pattern to restrict search (e.g. '**/*.py').",
            },
        },
        "required": ["query"],
    },
)

FIND_DEFINITION_TOOL = ToolDefinition(
    name="find_definition",
    description=(
        "Find where a class, function, method, or variable is defined in the workspace. "
        "Returns the file and line where the definition appears."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Name of the symbol to find (e.g. 'UserService', 'calculate_total').",
            }
        },
        "required": ["symbol"],
    },
)

WRITE_FILE_TOOL = ToolDefinition(
    name="write_file",
    description=(
        "Write or overwrite a file in the workspace with the given content. "
        "Parent directories are created automatically."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path to write (e.g. 'src/services/order_service.py').",
            },
            "content": {
                "type": "string",
                "description": "Full file content to write.",
            },
        },
        "required": ["path", "content"],
    },
    is_concurrency_safe=False,
)

APPLY_PATCH_TOOL = ToolDefinition(
    name="apply_patch",
    description=(
        "Apply a unified diff patch to an existing file in the workspace. "
        "The patch must be in standard unified diff format (--- / +++ headers, @@ hunks)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path of the file to patch.",
            },
            "patch": {
                "type": "string",
                "description": "Unified diff patch content.",
            },
        },
        "required": ["path", "patch"],
    },
    is_concurrency_safe=False,
)

LIST_FILES_TOOL = ToolDefinition(
    name="list_files",
    description="List files in a workspace directory, optionally filtered by glob pattern.",
    input_schema={
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "Workspace-relative directory path.  Defaults to workspace root.",
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern to filter results (e.g. '**/*.py').  Defaults to '*'.",
            },
        },
        "required": [],
    },
)

# ── Convenience groupings ────────────────────────────────────────────────────

#: Tools available to every agent by default.
STANDARD_TOOLS: list[ToolDefinition] = [
    READ_FILE_TOOL,
    SEARCH_CODE_TOOL,
    FIND_DEFINITION_TOOL,
    LIST_FILES_TOOL,
]

#: Full tool set for agents that also write files (CoderAgent, PatchAgent, …).
CODER_TOOLS: list[ToolDefinition] = [
    READ_FILE_TOOL,
    SEARCH_CODE_TOOL,
    FIND_DEFINITION_TOOL,
    WRITE_FILE_TOOL,
    APPLY_PATCH_TOOL,
    LIST_FILES_TOOL,
]

# ── Tool input schema registry (name → schema) ──────────────────────────────

TOOL_SCHEMAS: dict[str, dict] = {t.name: t.input_schema for t in CODER_TOOLS}

# JSON Schema type → Python type(s)
_JSON_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def validate_tool_input(tool_name: str, inp: dict) -> str | None:
    """Validate *inp* against the JSON Schema for *tool_name*.

    Returns an error message string if validation fails, or ``None`` if
    the input is valid.  Only checks ``required`` fields and top-level
    ``type`` constraints — intentionally lightweight to avoid a heavy
    dependency on jsonschema.
    """
    schema = TOOL_SCHEMAS.get(tool_name)
    if schema is None:
        return None  # unknown tool — skip validation

    # Check required keys
    for key in schema.get("required", []):
        if key not in inp:
            return f"Missing required field '{key}' for tool '{tool_name}'"

    # Check types of provided fields
    props = schema.get("properties", {})
    for key, value in inp.items():
        prop_schema = props.get(key)
        if prop_schema is None:
            continue  # extra keys are tolerated
        expected_type = prop_schema.get("type")
        if expected_type is None:
            continue
        allowed = _JSON_TYPE_MAP.get(expected_type)
        if allowed and not isinstance(value, allowed):
            return (
                f"Field '{key}' for tool '{tool_name}' expects {expected_type}, "
                f"got {type(value).__name__}"
            )
    return None


# ── Pydantic models for strict tool-input validation ─────────────────────────
# These provide deeper validation (path traversal blocking, length limits)
# while the lightweight ``validate_tool_input`` above remains available as a
# fast fall-through for unknown/MCP tools.

if _HAS_PYDANTIC:
    class _ReadFileInput(BaseModel):
        path: str
        start_line: int | None = None
        end_line: int | None = None

        @field_validator("path")
        @classmethod
        def _no_traversal(cls, v: str) -> str:
            if ".." in v.replace("\\", "/").split("/"):
                raise ValueError("path must not contain '..' traversal segments")
            return v

    class _WriteFileInput(BaseModel):
        path: str
        content: str

        @field_validator("path")
        @classmethod
        def _no_traversal(cls, v: str) -> str:
            if ".." in v.replace("\\", "/").split("/"):
                raise ValueError("path must not contain '..' traversal segments")
            return v

    class _SearchCodeInput(BaseModel):
        query: str
        file_pattern: str | None = None

    class _FindDefinitionInput(BaseModel):
        symbol: str

    class _ListFilesInput(BaseModel):
        directory: str | None = None
        pattern: str | None = None

    class _ApplyPatchInput(BaseModel):
        path: str
        patch: str

        @field_validator("path")
        @classmethod
        def _no_traversal(cls, v: str) -> str:
            if ".." in v.replace("\\", "/").split("/"):
                raise ValueError("path must not contain '..' traversal segments")
            return v

    _PYDANTIC_SCHEMAS: dict[str, type[BaseModel]] = {
        "read_file": _ReadFileInput,
        "write_file": _WriteFileInput,
        "search_code": _SearchCodeInput,
        "find_definition": _FindDefinitionInput,
        "list_files": _ListFilesInput,
        "apply_patch": _ApplyPatchInput,
    }
else:
    _PYDANTIC_SCHEMAS = {}  # type: ignore[assignment]


def validate_tool_input_strict(tool_name: str, inp: dict) -> str | None:
    """Strict Pydantic-based validation for tool inputs.

    Falls back to :func:`validate_tool_input` when Pydantic is unavailable
    or the tool has no Pydantic schema defined.
    """
    model_cls = _PYDANTIC_SCHEMAS.get(tool_name) if _HAS_PYDANTIC else None
    if model_cls is None:
        return validate_tool_input(tool_name, inp)
    try:
        model_cls.model_validate(inp)
    except ValidationError as exc:
        return "; ".join(e["msg"] for e in exc.errors())
    return None
