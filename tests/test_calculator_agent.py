from __future__ import annotations

import json

import pytest

from app.agent.calculator_agent import AgentError, AgentLoopError, calculator, parse_tool_request, run_agent


@pytest.mark.parametrize(
    ("operation", "a", "b", "expected"),
    [
        ("add", 8, 3, 11),
        ("subtract", 8, 3, 5),
        ("multiply", 8, 3, 24),
        ("divide", 8, 4, 2),
    ],
)
def test_calculator_operations(operation: str, a: int, b: int, expected: int):
    assert calculator(operation, a, b) == expected


def test_calculator_rejects_division_by_zero():
    with pytest.raises(AgentError, match="Division by zero"):
        calculator("divide", 8, 0)


def test_tool_request_parsing():
    request = parse_tool_request(
        '{"action":"calculator","arguments":{"operation":"multiply","a":12,"b":4}}'
    )
    assert request["action"] == "calculator"
    assert request["arguments"] == {"operation": "multiply", "a": 12, "b": 4}


def test_unknown_action_is_rejected():
    with pytest.raises(AgentError, match="Unknown action"):
        parse_tool_request('{"action":"shell","arguments":{}}')


def test_maximum_loop_protection():
    class RepeatingLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, _messages):
            self.calls += 1
            return json.dumps(
                {"action": "calculator", "arguments": {"operation": "add", "a": 1, "b": 1}}
            )

    llm = RepeatingLLM()
    with pytest.raises(AgentLoopError, match="maximum of 3"):
        run_agent("Add two numbers using the calculator.", llm)
    assert llm.calls == 3
