from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError
from scripted import ScriptedModel, tool_call

from dealer_assistant.agent import (
    SYSTEM_PROMPT,
    build_agent,
    chat_turn,
    parse_turn,
    tool_error_message,
)
from dealer_assistant.config import Settings
from dealer_assistant.inventory import Inventory
from dealer_assistant.tools import SearchCarsInput, make_tools

STOCK_PATH = Path(__file__).resolve().parents[1] / "data" / "stock_sample.csv"


def make_agent(tmp_path: Path, script: list[AIMessage]) -> Any:
    tools = make_tools(
        Inventory.from_csv(STOCK_PATH),
        leads_path=tmp_path / "leads.jsonl",
        handoffs_path=tmp_path / "handoffs.jsonl",
    )
    return build_agent(ScriptedModel(messages=iter(script)), tools)


def state_messages(agent: Any, thread_id: str) -> list[Any]:
    snapshot = agent.get_state({"configurable": {"thread_id": thread_id}})
    return list(snapshot.values["messages"])


def test_chat_turn_with_a_stock_question_calls_the_tool_and_returns_the_final_reply(
    tmp_path: Path,
) -> None:
    agent = make_agent(
        tmp_path,
        [
            tool_call("search_cars", {"make": "toyota", "limit": 2}),
            AIMessage(content="Tengo dos Toyota disponibles."),
        ],
    )
    turn = chat_turn(agent, "t1", "¿Tienen Toyota?")
    assert turn.reply == "Tengo dos Toyota disponibles."
    assert turn.tool_calls == [("search_cars", {"make": "toyota", "limit": 2})]


def test_chat_turn_with_missing_currency_tells_the_model_what_is_missing_and_recovers(
    tmp_path: Path,
) -> None:
    agent = make_agent(
        tmp_path,
        [
            tool_call("search_cars", {"max_price": 20_000_000}),  # currency missing
            AIMessage(content="¿Tu presupuesto es en pesos o en dólares?"),
        ],
    )
    turn = chat_turn(agent, "t2", "Busco algo de hasta 20 millones")
    assert turn.reply == "¿Tu presupuesto es en pesos o en dólares?"
    results = [m for m in state_messages(agent, "t2") if isinstance(m, ToolMessage)]
    assert len(results) == 1
    assert "invalid_filters" in str(results[0].content)
    assert "currency" in str(results[0].content)


def test_chat_turn_registering_a_lead_writes_the_record_and_reports_it(
    tmp_path: Path,
) -> None:
    agent = make_agent(
        tmp_path,
        [
            tool_call(
                "register_lead",
                {
                    "name": "Ana Pérez",
                    "phone": "3515550100",
                    "source": "instagram",
                    "car_id": "C004",
                },
            ),
            AIMessage(content="Listo, te vamos a contactar."),
        ],
    )
    chat_turn(agent, "t3", "Soy Ana, mi cel es 3515550100, los vi en Instagram")
    assert (tmp_path / "leads.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_chat_turn_keeps_history_within_a_thread_and_separates_threads(
    tmp_path: Path,
) -> None:
    agent = make_agent(
        tmp_path,
        [AIMessage(content="uno"), AIMessage(content="dos"), AIMessage(content="tres")],
    )
    chat_turn(agent, "a", "primer mensaje")
    chat_turn(agent, "a", "segundo mensaje")
    chat_turn(agent, "b", "otro cliente")
    humans_a = [m for m in state_messages(agent, "a") if isinstance(m, HumanMessage)]
    humans_b = [m for m in state_messages(agent, "b") if isinstance(m, HumanMessage)]
    assert [m.content for m in humans_a] == ["primer mensaje", "segundo mensaje"]
    assert [m.content for m in humans_b] == ["otro cliente"]


def test_parse_turn_only_reports_tool_calls_after_the_last_customer_message() -> None:
    result = {
        "messages": [
            HumanMessage(content="hola"),
            tool_call("search_cars", {"make": "ford"}, "old"),
            ToolMessage(content="[]", tool_call_id="old"),
            AIMessage(content="No tengo Ford."),
            HumanMessage(content="¿y Fiat?"),
            tool_call("search_cars", {"make": "fiat"}, "new"),
            ToolMessage(content="[...]", tool_call_id="new"),
            AIMessage(content="Tengo dos Fiat."),
        ]
    }
    turn = parse_turn(result)
    assert turn.reply == "Tengo dos Fiat."
    assert turn.tool_calls == [("search_cars", {"make": "fiat"})]


def request_for(name: str) -> ToolCallRequest:
    return cast(ToolCallRequest, SimpleNamespace(tool_call={"name": name}))


def test_tool_error_message_with_validation_error_names_the_rule_not_the_values() -> (
    None
):
    with pytest.raises(ValidationError) as info:
        SearchCarsInput(limit=987654)  # type: ignore[call-arg]  # out of range on purpose; mypy cannot see Field defaults
    message = tool_error_message(info.value, request_for("search_cars"))
    assert message is not None
    assert "limit" in message and "987654" not in message


def test_tool_error_message_with_value_error_explains_the_failure() -> None:
    message = tool_error_message(ValueError("bad filter"), request_for("search_cars"))
    assert message is not None and "bad filter" in message


def test_tool_error_message_with_unexpected_error_lets_it_propagate() -> None:
    assert tool_error_message(RuntimeError("boom"), request_for("search_cars")) is None


def test_system_prompt_mentions_only_tools_that_really_exist(tmp_path: Path) -> None:
    names = {
        tool.name
        for tool in make_tools(
            Inventory.from_csv(STOCK_PATH),
            leads_path=tmp_path / "leads.jsonl",
            handoffs_path=tmp_path / "handoffs.jsonl",
        )
    }
    for mentioned in ("search_cars", "register_lead", "handoff_to_human"):
        assert mentioned in SYSTEM_PROMPT and mentioned in names


def test_settings_with_api_key_in_environment_uses_the_sample_stock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-not-a-real-key")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]  # key comes from env
    assert settings.google_api_key.get_secret_value() == "dummy-not-a-real-key"
    assert settings.stock_path.name == "stock_sample.csv"


def test_settings_without_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]  # key missing on purpose
