"""Model Context Protocol (MCP) client integration."""

import logging
from typing import Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core.agent_tools import ToolDefinition

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for connecting to a local embedded MCP Server via stdio."""

    def __init__(self, server_command: list[str]):
        """
        Initialize the embedded MCP client.
        
        Args:
            server_command: The command array to execute, e.g. ["npx", "-y", "@figma/mcp-server"]
        """
        self.server_command = server_command
        if not self.server_command:
            raise ValueError("server_command must not be empty")

        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        
        # Keep track of the context managers so we can exit them cleanly
        self._stdio_ctx: Any | None = None
        self._session_ctx: Any | None = None

    async def initialize(self) -> None:
        """Boot the local subprocess and establish the MCP session."""
        if self._session is not None:
            return  # Already initialized
            
        logger.info(f"Starting embedded MCP server with command: {' '.join(self.server_command)}")
        
        try:
            self._exit_stack = AsyncExitStack()
            
            # Use tuple slicing workaround for typed list slicing errors in pyre
            command_args = []
            if len(self.server_command) > 1:
                command_args = [arg for arg in self.server_command[1:]]
                
            server_parameters = StdioServerParameters(
                command=self.server_command[0],
                args=command_args,
            )
            
            # 1. Start the stdio process
            self._stdio_ctx = stdio_client(server_parameters)
            if self._exit_stack and self._stdio_ctx:
                read_stream, write_stream = await self._exit_stack.enter_async_context(self._stdio_ctx)
                
                # 2. Establish the session
                self._session_ctx = ClientSession(read_stream, write_stream)
                self._session = await self._exit_stack.enter_async_context(self._session_ctx)
                
                # 3. Initialize the protocol handshake
                await self._session.initialize()
                
                logger.info("Embedded MCP server initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedded MCP server: {e}")
            if self._exit_stack:
                await self._exit_stack.aclose()
            raise

    async def list_tools(self) -> list[ToolDefinition]:
        """Fetch tools from the local MCP Server."""
        if not self._session:
            await self.initialize()
            
        try:
            # The python SDK returns a list of generic Tools
            result = await self._session.list_tools()
            
            tool_definitions = []
            for tool in result.tools:
                tool_definitions.append(
                    ToolDefinition(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema or {}
                    )
                )
            logger.info(f"Loaded {len(tool_definitions)} MCP tools locally")
            return tool_definitions
        except Exception as e:
            logger.error(f"Failed to fetch MCP tools locally: {e}")
            return []

    async def execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Forward a tool execution request to the local MCP Server."""
        if not self._session:
            await self.initialize()
            
        try:
            result = await self._session.call_tool(name, arguments=args)
            
            if not result.content:
                return f"MCP tool {name} succeeded but returned no content."
                
            result_texts = []
            for item in result.content:
                if item.type == "text":
                    result_texts.append(item.text)
                else:
                    # e.g., images or resources - currently just stringified
                    result_texts.append(str(item))
            
            return "\n".join(result_texts)
        except Exception as e:
            logger.error(f"MCP tool {name} execution failed locally: {e}")
            return f"Error executing MCP tool '{name}': {e}"

    async def close(self):
        """Cleanly terminate the local MCP subprocess."""
        logger.info("Closing embedded MCP server connection")
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            self._stdio_ctx = None
            self._session_ctx = None
