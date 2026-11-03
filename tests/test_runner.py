from collections.abc import Iterator
from pathlib import Path

import httpx
from langchain_core.messages import AIMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from scripted import ScriptedModel, tool_call

from dealer_assistant.evaluate import progress_mark, run_case
from dealer_assistant.exam import Case, CaseRun, Failure
from dealer_assistant.inventory import Inventory

STOCK_PATH = Path(__file__).resolve().parents[1] / "data" / "stock_sample.csv"


def play(case: Case, script: Iterator[AIMessage]):  # type: ignore[no-untyped-def]  # returns CaseRun
    return run_case(
        case,
        1,
        model_factory=lambda: ScriptedModel(messages=script),
        inventory=Inventory.from_csv(STOCK_PATH),
        pause_s=0,
    )


def test_run_case_collects_replies_tool_calls_and_tool_outputs() -> None:
    case = Case.model_validate({"id": "x", "type": "t", "turns": ["¿Tienen Toyota?"]})
    script = iter(
        [
            tool_call("search_cars", {"make": "toyota", "limit": 1}),
            AIMessage(content="Tengo un Corolla Cross."),
        ]
    )
    result = play(case, script)
    assert result.error is None
    assert result.replies == ["Tengo un Corolla Cross."]
    assert result.tool_calls == [("search_cars", {"make": "toyota", "limit": 1})]
    assert "C022" in result.tool_outputs[0]
    assert result.customer_messages == ["¿Tienen Toyota?"]


def test_run_case_with_several_turns_keeps_one_reply_per_turn() -> None:
    case = Case.model_validate({"id": "x", "type": "t", "turns": ["hola", "gracias"]})
    result = play(case, iter([AIMessage(content="Hola"), AIMessage(content="De nada")]))
    assert result.replies == ["Hola", "De nada"]


def test_run_case_with_a_model_service_failure_records_the_error_instead_of_raising() -> (
    None
):
    def failing() -> Iterator[AIMessage]:
        raise ChatGoogleGenerativeAIError("service down")
        yield  # pragma: no cover - makes this a generator

    case = Case.model_validate({"id": "x", "type": "t", "turns": ["hola"]})
    result = play(case, failing())
    assert result.error is not None and "service down" in result.error
    assert result.replies == []


def test_run_case_with_a_network_timeout_records_the_error_instead_of_crashing() -> (
    None
):
    def timing_out() -> Iterator[AIMessage]:
        raise httpx.ReadTimeout("slow")
        yield  # pragma: no cover - makes this a generator

    case = Case.model_validate({"id": "x", "type": "t", "turns": ["hola"]})
    result = play(case, timing_out())
    assert result.error is not None and "ReadTimeout" in result.error


def test_progress_mark_distinguishes_pass_failure_and_service_error() -> None:
    ok = CaseRun("c", "t", 1, ["hola"], ["ok"], [], [])
    erroring = CaseRun("c", "t", 1, ["hola"], [], [], [], error="ReadTimeout: slow")
    assert progress_mark(ok, []) == "✅"
    assert progress_mark(ok, [Failure("grounding", "x")]) == "❌ grounding"
    assert progress_mark(erroring, []).startswith("⚠ error (ReadTimeout")
