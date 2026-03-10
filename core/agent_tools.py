"""Standard tool definitions available to all agents via the tool-use loop."""

from __future__ import annotations

from core.llm_client import ToolDefinition

# ── Standard agent tools ─────────────────────────────────────────────────────
# All tools follow the JSON Schema format required by the Claude tool_use API.

READ_FILE_TOOL = ToolDefinition(
    name="read_file",
    description="Read the current content of a file in the workspace.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path to the file (e.g. 'src/models/user.py').",
            }
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
