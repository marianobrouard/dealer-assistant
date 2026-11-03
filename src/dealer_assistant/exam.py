"""Exam cases for the sales agent and the automatic checks applied to each conversation."""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from dealer_assistant.grounding import ungrounded_numbers

CASES_PATH = Path(__file__).resolve().parents[2] / "evals" / "cases.json"


class ExpectedCall(BaseModel):
    tool: str
    args: dict[str, object] = Field(default_factory=dict)


class Case(BaseModel):
    """One scripted conversation. The customer sends `turns` in order."""

    id: str
    type: str
    turns: list[str] = Field(min_length=1)
    expect_tools: list[str] = Field(default_factory=list)
    expect_calls: list[ExpectedCall] = Field(default_factory=list)
    forbid_tools: list[str] = Field(default_factory=list)
    reply_must_contain_any: list[str] = Field(default_factory=list)
    reply_must_not_contain: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class CaseRun:
    """What happened when one case was played against the agent."""

    case_id: str
    case_type: str
    attempt: int
    customer_messages: list[str]
    replies: list[str]
    tool_calls: list[tuple[str, dict[str, object]]]
    tool_outputs: list[str]
    error: str | None = None  # infrastructure failure: the case is not scored


@dataclass(frozen=True)
class Failure:
    check: str
    detail: str


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    cases = [
        Case.model_validate(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    return cases


def _args_match(actual: dict[str, object], expected: dict[str, object]) -> bool:
    return all(
        key in actual and actual[key] == value for key, value in expected.items()
    )


def check_run(case: Case, run: CaseRun) -> list[Failure]:
    """Failures of a run. A run that hit an infrastructure error has none: it is not scored."""
    if run.error is not None:
        return []
    failures: list[Failure] = []
    reply_text = "\n".join(run.replies)
    lowered = reply_text.casefold()
    used = {name for name, _ in run.tool_calls}

    invented = ungrounded_numbers(
        reply_text, [*run.tool_outputs, *run.customer_messages]
    )
    if invented:
        listed = ", ".join(str(number) for number in invented)
        failures.append(
            Failure("grounding", f"numbers not in any tool result: {listed}")
        )

    missing = [tool for tool in case.expect_tools if tool not in used]
    if missing:
        failures.append(Failure("expected_tools", f"not called: {', '.join(missing)}"))

    for expected in case.expect_calls:
        if not any(
            name == expected.tool and _args_match(args, expected.args)
            for name, args in run.tool_calls
        ):
            failures.append(
                Failure(
                    "expected_call", f"no call to {expected.tool} with {expected.args}"
                )
            )

    forbidden = [tool for tool in case.forbid_tools if tool in used]
    if forbidden:
        failures.append(Failure("forbidden_tools", f"called: {', '.join(forbidden)}"))

    wanted = case.reply_must_contain_any
    if wanted and not any(text.casefold() in lowered for text in wanted):
        failures.append(
            Failure("reply_content", f"none of {wanted} appears in the reply")
        )

    leaked = [
        text for text in case.reply_must_not_contain if text.casefold() in lowered
    ]
    if leaked:
        failures.append(Failure("reply_forbidden_text", f"reply contains: {leaked}"))
    return failures
