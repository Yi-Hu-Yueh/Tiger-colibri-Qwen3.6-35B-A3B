from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

OPENAI_BASE_URL = "http://127.0.0.1:18081/v1"
MODEL_ID = "qwen36"
MAX_LLM_TURNS = 3

SYSTEM_PROMPT = """You are a tool-using local assistant with one available tool: calculator.
When arithmetic is needed, do not calculate it yourself. Respond with ONLY one JSON object in this exact shape:
{"action":"calculator","arguments":{"operation":"multiply","a":6,"b":7}}
Supported operations are add, subtract, multiply, and divide.
After a calculator result is supplied, answer the original task in one concise natural-language sentence using that result. Do not request another tool.
Do not include hidden reasoning or chain-of-thought."""


class AgentError(RuntimeError):
    pass


class AgentLoopError(AgentError):
    pass


class ToolRequestNotFound(AgentError):
    pass


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


@dataclass(frozen=True)
class AgentEvidence:
    initial_user_task: str
    first_llm_response: str
    parsed_tool_request: dict[str, Any]
    calculator_arguments: dict[str, Any]
    calculator_result: int | float
    follow_up_llm_request: str
    final_model_response: str
    llm_turns: int
    total_wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenAIChatClient:
    def __init__(self, base_url: str = OPENAI_BASE_URL, model: str = MODEL_ID) -> None:
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="local", max_retries=0, timeout=900.0)

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=128,
            temperature=0,
            stream=False,
        )
        content = response.choices[0].message.content
        if not content:
            raise AgentError("qwen36 returned an empty response.")
        return content.strip()


def _validate_calculator_arguments(operation: str, a: int | float, b: int | float) -> None:
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise AgentError("Calculator arguments 'a' and 'b' must be numbers.")
    if operation not in {"add", "subtract", "multiply", "divide"}:
        raise AgentError(f"Unsupported calculator operation: {operation}")
    if operation == "divide" and b == 0:
        raise AgentError("Division by zero is not allowed.")


def calculator(operation: str, a: int | float, b: int | float) -> int | float:
    _validate_calculator_arguments(operation, a, b)
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        return a / b
    raise AssertionError("Validated calculator operation was not handled.")


def parse_tool_request(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise ToolRequestNotFound("The model did not return a structured tool request.")
    try:
        request = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as error:
        raise AgentError("The model returned invalid tool-request JSON.") from error
    if not isinstance(request, dict):
        raise AgentError("The tool request must be a JSON object.")
    action = request.get("action")
    if action != "calculator":
        raise AgentError(f"Unknown action requested: {action!r}")
    arguments = request.get("arguments")
    if not isinstance(arguments, dict):
        raise AgentError("The calculator action requires structured arguments.")
    if set(arguments) != {"operation", "a", "b"}:
        raise AgentError("Calculator arguments must contain only operation, a, and b.")
    _validate_calculator_arguments(arguments["operation"], arguments["a"], arguments["b"])
    return request


def run_agent(task: str, llm: LLMClient, max_turns: int = MAX_LLM_TURNS) -> AgentEvidence:
    if not task.strip():
        raise AgentError("A user task is required.")
    if not 1 <= max_turns <= MAX_LLM_TURNS:
        raise AgentError(f"max_turns must be between 1 and {MAX_LLM_TURNS}.")

    started = time.perf_counter()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    first_response = ""
    parsed_request: dict[str, Any] | None = None
    calculator_arguments: dict[str, Any] | None = None
    calculator_result: int | float | None = None
    follow_up_request = ""

    for turn in range(1, max_turns + 1):
        response = llm.complete(messages)
        if turn == 1:
            first_response = response

        try:
            tool_request = parse_tool_request(response)
        except ToolRequestNotFound:
            if calculator_result is None:
                raise AgentError("The model answered without initiating the required calculator tool path.")
            return AgentEvidence(
                initial_user_task=task,
                first_llm_response=first_response,
                parsed_tool_request=parsed_request or {},
                calculator_arguments=calculator_arguments or {},
                calculator_result=calculator_result,
                follow_up_llm_request=follow_up_request,
                final_model_response=response,
                llm_turns=turn,
                total_wall_seconds=round(time.perf_counter() - started, 3),
            )

        arguments = tool_request["arguments"]
        result = calculator(arguments["operation"], arguments["a"], arguments["b"])
        if parsed_request is None:
            parsed_request = tool_request
            calculator_arguments = dict(arguments)
            calculator_result = result
        follow_up_request = (
            "Calculator tool result: "
            + json.dumps(
                {"operation": arguments["operation"], "a": arguments["a"], "b": arguments["b"], "result": result},
                ensure_ascii=False,
            )
            + ". Now answer the original user task in one concise natural-language sentence using this tool result."
        )
        messages.extend(
            [
                {"role": "assistant", "content": response},
                {"role": "user", "content": follow_up_request},
            ]
        )

    raise AgentLoopError(f"Agent exceeded the maximum of {max_turns} LLM turns.")

