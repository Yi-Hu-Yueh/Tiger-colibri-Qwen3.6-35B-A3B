from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


class ToolCompatibilityError(RuntimeError):
    def __init__(self, message: str, code: str, *, client_error: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.client_error = client_error


@dataclass(frozen=True)
class PreparedToolRequest:
    messages: list[dict[str, str]]
    expect_tool_action: bool
    selected_tool: str | None


def _function_map(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    functions: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if tool.get("type") != "function":
            raise ToolCompatibilityError(
                "Only function tools are supported.",
                "unsupported_tool_type",
                client_error=True,
            )
        function = tool.get("function") or {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ToolCompatibilityError("Each function tool requires a name.", "invalid_tool", client_error=True)
        if name in functions:
            raise ToolCompatibilityError(f"Duplicate function tool: {name}", "invalid_tool", client_error=True)
        parameters = function.get("parameters") or {"type": "object", "properties": {}}
        _validate_parameter_schema(parameters)
        functions[name] = function
    return functions


def _validate_parameter_schema(schema: dict[str, Any]) -> None:
    if not isinstance(schema, dict) or schema.get("type", "object") != "object":
        raise ToolCompatibilityError(
            "Function parameters must use an object JSON schema.",
            "invalid_tool_schema",
            client_error=True,
        )
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise ToolCompatibilityError("Invalid function parameter schema.", "invalid_tool_schema", client_error=True)
    if any(name not in properties for name in required):
        raise ToolCompatibilityError(
            "Every required parameter must be declared in properties.",
            "invalid_tool_schema",
            client_error=True,
        )
    supported_types = {"string", "number", "integer", "boolean"}
    for name, definition in properties.items():
        if not isinstance(definition, dict) or definition.get("type") not in supported_types:
            raise ToolCompatibilityError(
                f"Unsupported schema for function parameter '{name}'.",
                "invalid_tool_schema",
                client_error=True,
            )


def _validate_arguments(arguments: dict[str, Any], function: dict[str, Any]) -> None:
    schema = function.get("parameters") or {"type": "object", "properties": {}}
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ToolCompatibilityError(
            f"Tool arguments are missing required fields: {', '.join(missing)}.",
            "tool_arguments_invalid",
        )
    if schema.get("additionalProperties") is False:
        extra = [name for name in arguments if name not in properties]
        if extra:
            raise ToolCompatibilityError(
                f"Tool arguments contain unsupported fields: {', '.join(extra)}.",
                "tool_arguments_invalid",
            )
    checks = {
        "string": lambda value: isinstance(value, str),
        "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "boolean": lambda value: isinstance(value, bool),
    }
    for name, value in arguments.items():
        definition = properties.get(name)
        if definition is None:
            continue
        expected = definition["type"]
        if not checks[expected](value):
            raise ToolCompatibilityError(
                f"Tool argument '{name}' must be of type {expected}.",
                "tool_arguments_invalid",
            )
        if "enum" in definition and value not in definition["enum"]:
            raise ToolCompatibilityError(
                f"Tool argument '{name}' is not an allowed value.",
                "tool_arguments_invalid",
            )


def _tool_choice_name(tool_choice: str | dict[str, Any], functions: dict[str, dict[str, Any]]) -> str | None:
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none"}:
            return None
        raise ToolCompatibilityError(
            "tool_choice must be 'auto', 'none', or an explicit function selection.",
            "invalid_tool_choice",
            client_error=True,
        )
    name = ((tool_choice.get("function") or {}).get("name")) if isinstance(tool_choice, dict) else None
    if tool_choice.get("type") != "function" or not isinstance(name, str):
        raise ToolCompatibilityError("Invalid explicit tool_choice.", "invalid_tool_choice", client_error=True)
    if name not in functions:
        raise ToolCompatibilityError(
            f"Explicit tool_choice references unknown function '{name}'.",
            "invalid_tool_choice",
            client_error=True,
        )
    return name


def prepare_tool_request(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any],
) -> PreparedToolRequest:
    functions = _function_map(tools)
    selected_tool = _tool_choice_name(tool_choice, functions)
    translated: list[dict[str, str]] = []
    calls_by_id: dict[str, dict[str, Any]] = {}
    has_tool_result = False

    for message in messages:
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                if call.get("type") != "function":
                    raise ToolCompatibilityError(
                        "Only function tool calls are supported.",
                        "invalid_messages",
                        client_error=True,
                    )
                function_call = call.get("function") or {}
                name = function_call.get("name")
                if name not in functions:
                    raise ToolCompatibilityError(
                        f"Assistant message references unknown function '{name}'.",
                        "invalid_messages",
                        client_error=True,
                    )
                try:
                    arguments = json.loads(function_call.get("arguments", ""))
                except json.JSONDecodeError as error:
                    raise ToolCompatibilityError(
                        "Assistant tool-call arguments must be valid JSON.",
                        "invalid_messages",
                        client_error=True,
                    ) from error
                if not isinstance(arguments, dict):
                    raise ToolCompatibilityError(
                        "Assistant tool-call arguments must be a JSON object.",
                        "invalid_messages",
                        client_error=True,
                    )
                _validate_arguments(arguments, functions[name])
                calls_by_id[call["id"]] = {"name": name, "arguments": arguments}
            summary = json.dumps(list(calls_by_id.values()), ensure_ascii=False, separators=(",", ":"))
            translated.append({"role": "assistant", "content": f"Requested tool call: {summary}"})
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            call = calls_by_id.get(call_id)
            if call is None:
                raise ToolCompatibilityError(
                    f"Tool result references unknown tool_call_id '{call_id}'.",
                    "invalid_messages",
                    client_error=True,
                )
            has_tool_result = True
            translated.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool result for {call['name']} with arguments "
                        f"{json.dumps(call['arguments'], ensure_ascii=False, separators=(',', ':'))}: "
                        f"{message['content']}"
                    ),
                }
            )
            continue
        content = message.get("content")
        if content is not None:
            translated.append({"role": role, "content": content})

    if has_tool_result:
        translated.insert(
            0,
            {
                "role": "system",
                "content": (
                    "A trusted client has supplied a tool result in the conversation. "
                    "Use that result to answer the user's task directly. Do not request another tool and do not emit JSON."
                ),
            },
        )
        return PreparedToolRequest(translated, False, selected_tool)

    if tool_choice == "none":
        return PreparedToolRequest(translated, False, None)

    compact_tools = [
        {
            "name": name,
            "description": function.get("description", ""),
            "parameters": function.get("parameters") or {"type": "object", "properties": {}},
        }
        for name, function in functions.items()
    ]
    if selected_tool:
        choice_instruction = f"You must call the function named '{selected_tool}'."
    else:
        choice_instruction = (
            "When an available function can perform the requested operation, you must call it instead of "
            "performing the operation or answering from your own reasoning. Arithmetic must use an available "
            "calculator function. Answer normally only when no available function applies."
        )
    instruction = (
        "You can request one function tool. "
        + choice_instruction
        + " If calling a function, respond with ONLY one compact JSON object in the exact shape "
        + '{"name":"function_name","arguments":{...}}. '
        + "Do not include prose, Markdown, code fences, or chain-of-thought. Available functions: "
        + json.dumps(compact_tools, ensure_ascii=False, separators=(",", ":"))
    )
    translated.insert(0, {"role": "system", "content": instruction})
    return PreparedToolRequest(translated, True, selected_tool)


def parse_structured_action(
    content: str,
    tools: list[dict[str, Any]],
    *,
    selected_tool: str | None = None,
    allow_normal_response: bool = True,
) -> tuple[str, dict[str, Any]] | None:
    candidate = content.strip()
    if not candidate.startswith("{"):
        if allow_normal_response:
            return None
        raise ToolCompatibilityError("The model did not return the required tool action.", "tool_protocol_error")
    try:
        action = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ToolCompatibilityError("The model returned malformed tool-action JSON.", "tool_protocol_error") from error
    if not isinstance(action, dict) or set(action) != {"name", "arguments"}:
        raise ToolCompatibilityError(
            "The model tool action must contain exactly name and arguments.",
            "tool_protocol_error",
        )
    name = action["name"]
    arguments = action["arguments"]
    functions = _function_map(tools)
    if name not in functions:
        raise ToolCompatibilityError(f"The model requested unknown function '{name}'.", "unknown_tool")
    if selected_tool is not None and name != selected_tool:
        raise ToolCompatibilityError(
            f"The model requested '{name}' instead of explicitly selected function '{selected_tool}'.",
            "wrong_tool_selected",
        )
    if not isinstance(arguments, dict):
        raise ToolCompatibilityError("Tool arguments must be a JSON object.", "tool_arguments_invalid")
    _validate_arguments(arguments, functions[name])
    return name, arguments


def translate_tool_completion(
    completion: dict[str, Any],
    tools: list[dict[str, Any]],
    *,
    selected_tool: str | None = None,
) -> dict[str, Any]:
    choices = completion.get("choices") or []
    if not choices:
        raise ToolCompatibilityError("The backend returned no completion choice.", "tool_protocol_error")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ToolCompatibilityError("The backend returned no tool-action content.", "tool_protocol_error")
    parsed = parse_structured_action(
        content,
        tools,
        selected_tool=selected_tool,
        allow_normal_response=selected_tool is None,
    )
    if parsed is None:
        return completion
    name, arguments = parsed
    arguments_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(f"{name}\0{arguments_json}".encode("utf-8")).hexdigest()[:24]
    choices[0]["message"] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{digest}",
                "type": "function",
                "function": {"name": name, "arguments": arguments_json},
            }
        ],
    }
    choices[0]["finish_reason"] = "tool_calls"
    return completion
