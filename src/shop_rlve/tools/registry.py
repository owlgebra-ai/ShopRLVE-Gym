"""Tool registry and dispatch system for ShopRLVE-GYM.

Spec Section 2.2:
    - LLM actions include tool_calls: [{"name": "tool.name", "args": {"k": "v"}}]
    - Each tool call must validate against a known schema.
    - Invalid tool calls result in reward = -1 (hard fail).

This module provides:
    - ToolCall / ToolResult pydantic models for structured tool invocation.
    - TOOL_SCHEMA dict mapping tool names to their arg schemas.
    - ToolRegistry class with register/validate/execute/list_tools.
    - Debug lever: ToolRegistry.trace_mode logs every call + result + timing.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool call / result models
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A single tool invocation from the LLM.

    Spec Section 2.1:
        "tool_calls": [{"name": "tool.name", "args": {"k": "v"}}]
    """

    name: str = Field(..., description="Fully qualified tool name (e.g. 'catalog.search')")
    args: dict[str, Any] = Field(default_factory=dict, description="Tool arguments as key-value pairs")


class ToolResult(BaseModel):
    """Result of a tool execution.

    Returned by ToolRegistry.execute() after dispatching a ToolCall.
    """

    name: str = Field(..., description="Tool name that was called")
    result: Any = Field(default=None, description="Tool return value (serializable)")
    error: str | None = Field(default=None, description="Error message if execution failed")
    duration_ms: float = Field(default=0.0, description="Execution wall-clock time in ms")


# ---------------------------------------------------------------------------
# Global tool schema registry (populated by register calls)
# ---------------------------------------------------------------------------

# Maps tool name -> pydantic arg schema class
TOOL_SCHEMA: dict[str, type[BaseModel]] = {}


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Central registry for tool handlers with validation and dispatch.

    Supports:
        - Registering tools with name, handler, and pydantic arg schema.
        - Validating tool calls before execution (name exists + args validate).
        - Executing tool calls against an episode state.
        - Listing all registered tools with schemas (for LLM system prompt).

    Debug lever:
        ToolRegistry.trace_mode = True
        When enabled, every tool call is logged with its arguments, result,
        and execution time. Useful for debugging tool interactions.

    Example:
        >>> registry = ToolRegistry()
        >>> from pydantic import BaseModel
        >>> class EchoArgs(BaseModel):
        ...     message: str
        >>> def echo_handler(message: str, state: Any = None) -> str:
        ...     return f"Echo: {message}"
        >>> registry.register("echo", echo_handler, EchoArgs)
        >>> call = ToolCall(name="echo", args={"message": "hello"})
        >>> result = registry.execute(call, state=None)
        >>> result.result
        'Echo: hello'
    """

    trace_mode: bool = False

    def __init__(self) -> None:
        self._handlers: dict[str, Callable] = {}
        self._schemas: dict[str, type[BaseModel]] = {}

    def register(
        self,
        name: str,
        handler: Callable,
        arg_schema: type[BaseModel],
    ) -> None:
        """Register a tool handler with its name and argument schema.

        Args:
            name:       Fully qualified tool name (e.g. "catalog.search").
            handler:    Callable that accepts validated keyword args + optional
                        'state' kwarg. Must return a serializable value.
            arg_schema: Pydantic BaseModel subclass defining the expected arguments.

        Raises:
            ValueError: If a tool with this name is already registered.
        """
        if name in self._handlers:
            raise ValueError(f"Tool '{name}' is already registered")

        self._handlers[name] = handler
        self._schemas[name] = arg_schema

        # Also update global schema registry
        TOOL_SCHEMA[name] = arg_schema

        if self.trace_mode:
            logger.info(
                "ToolRegistry: registered tool '%s' (schema=%s)",
                name,
                arg_schema.__name__,
            )

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: Tool name to remove.

        Raises:
            KeyError: If tool is not registered.
        """
        if name not in self._handlers:
            raise KeyError(f"Tool '{name}' is not registered")

        del self._handlers[name]
        del self._schemas[name]
        TOOL_SCHEMA.pop(name, None)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._handlers

    def validate_call(self, tool_call: ToolCall) -> tuple[bool, str | None]:
        """Validate a tool call: check name exists and args match schema.

        Args:
            tool_call: The ToolCall to validate.

        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        # Check tool name exists
        if tool_call.name not in self._handlers:
            return False, f"Unknown tool: '{tool_call.name}'"

        # Validate args against schema
        schema_cls = self._schemas[tool_call.name]
        try:
            schema_cls.model_validate(tool_call.args)
            return True, None
        except Exception as exc:
            return False, f"Invalid args for '{tool_call.name}': {exc}"

    def execute(self, tool_call: ToolCall, state: Any = None) -> ToolResult:
        """Dispatch a tool call to its handler and return the result.

        The handler is called with validated keyword args from the tool call,
        plus a 'state' keyword argument if the handler accepts it.

        Args:
            tool_call: The ToolCall to execute.
            state:     Episode state object passed to the handler.

        Returns:
            ToolResult with the handler's return value or error information.
        """
        start_time = time.monotonic()

        # Validate first
        is_valid, error_msg = self.validate_call(tool_call)
        if not is_valid:
            duration_ms = (time.monotonic() - start_time) * 1000
            result = ToolResult(
                name=tool_call.name,
                result=None,
                error=error_msg,
                duration_ms=duration_ms,
            )
            if self.trace_mode:
                logger.warning(
                    "ToolRegistry TRACE [%s] VALIDATION FAILED: %s (%.1fms)",
                    tool_call.name,
                    error_msg,
                    duration_ms,
                )
            return result

        # Parse and validate args
        schema_cls = self._schemas[tool_call.name]
        validated_args = schema_cls.model_validate(tool_call.args)
        handler = self._handlers[tool_call.name]

        # Execute handler
        try:
            # Convert pydantic model to dict for keyword unpacking
            kwargs = validated_args.model_dump()
            kwargs["state"] = state
            handler_result = handler(**kwargs)

            duration_ms = (time.monotonic() - start_time) * 1000

            result = ToolResult(
                name=tool_call.name,
                result=handler_result,
                error=None,
                duration_ms=duration_ms,
            )

            if self.trace_mode:
                # Truncate large results for logging
                result_repr = repr(handler_result)
                if len(result_repr) > 500:
                    result_repr = result_repr[:500] + "..."
                logger.info(
                    "ToolRegistry TRACE [%s] args=%s -> result=%s (%.1fms)",
                    tool_call.name,
                    tool_call.args,
                    result_repr,
                    duration_ms,
                )

            return result

        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            error_msg = f"Handler error for '{tool_call.name}': {type(exc).__name__}: {exc}"

            if self.trace_mode:
                logger.error(
                    "ToolRegistry TRACE [%s] HANDLER ERROR: %s (%.1fms)",
                    tool_call.name,
                    error_msg,
                    duration_ms,
                )

            return ToolResult(
                name=tool_call.name,
                result=None,
                error=error_msg,
                duration_ms=duration_ms,
            )

    def list_tools(self) -> list[dict[str, Any]]:
        """Return tool names + schemas for LLM prompt generation.

        Returns a list of dicts, each containing:
            - name: tool name string
            - description: from schema docstring
            - parameters: JSON Schema of the arg model

        Returns:
            List of tool description dicts, suitable for inclusion in
            an LLM system prompt.
        """
        tools: list[dict[str, Any]] = []
        for name in sorted(self._handlers):
            schema_cls = self._schemas[name]
            json_schema = schema_cls.model_json_schema()

            tool_info: dict[str, Any] = {
                "name": name,
                "description": schema_cls.__doc__ or f"Tool: {name}",
                "parameters": json_schema,
            }
            tools.append(tool_info)

        return tools

    def get_tool_names(self) -> list[str]:
        """Return sorted list of all registered tool names."""
        return sorted(self._handlers.keys())

    def get_schema(self, name: str) -> type[BaseModel] | None:
        """Get the arg schema class for a tool name.

        Args:
            name: Tool name.

        Returns:
            Pydantic BaseModel subclass, or None if not found.
        """
        return self._schemas.get(name)

    def execute_batch(
        self,
        tool_calls: list[ToolCall],
        state: Any = None,
        budget: int | None = None,
    ) -> list[ToolResult]:
        """Execute a batch of tool calls sequentially.

        Spec Section 8.3 step 2:
            - Execute tool calls sequentially; append results.
            - Enforce budget: if len(tool_calls) > B_tool(d) -> penalty/hard fail.

        Args:
            tool_calls: List of ToolCall instances.
            state:      Episode state object.
            budget:     Maximum allowed tool calls (B_tool(d)). If exceeded,
                        remaining calls are rejected with an error.

        Returns:
            List of ToolResult instances, one per tool call.
        """
        results: list[ToolResult] = []

        for i, call in enumerate(tool_calls):
            if budget is not None and i >= budget:
                results.append(
                    ToolResult(
                        name=call.name,
                        result=None,
                        error=(
                            f"Tool call budget exceeded: {i + 1} > {budget}. "
                            f"B_tool(d) = {budget} calls allowed per step."
                        ),
                        duration_ms=0.0,
                    )
                )
                continue

            result = self.execute(call, state=state)
            results.append(result)

        return results
