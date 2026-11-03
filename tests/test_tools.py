import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from dealer_assistant.inventory import Inventory
from dealer_assistant.tools import make_tools

STOCK_PATH = Path(__file__).resolve().parents[1] / "data" / "stock_sample.csv"
FIXED_NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)


@pytest.fixture
def tools(tmp_path: Path) -> dict[str, BaseTool]:
    built = make_tools(
        Inventory.from_csv(STOCK_PATH),
        leads_path=tmp_path / "leads.jsonl",
        handoffs_path=tmp_path / "handoffs.jsonl",
        clock=lambda: FIXED_NOW,
    )
    return {tool.name: tool for tool in built}


def read_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_make_tools_exposes_exactly_the_four_agent_tools(
    tools: dict[str, BaseTool],
) -> None:
    assert set(tools) == {
        "search_cars",
        "get_car_details",
        "register_lead",
        "handoff_to_human",
    }


def test_search_cars_with_make_returns_matching_cars_as_plain_dicts(
    tools: dict[str, BaseTool],
) -> None:
    result = tools["search_cars"].invoke({"make": "toyota", "limit": 2})
    assert [car["id"] for car in result] == ["C022", "C021"]
    assert result[0]["currency"] == "ARS"


def test_search_cars_with_no_match_returns_empty_list(
    tools: dict[str, BaseTool],
) -> None:
    assert tools["search_cars"].invoke({"make": "ferrari"}) == []


def test_search_cars_with_price_limit_but_no_currency_explains_what_is_missing(
    tools: dict[str, BaseTool],
) -> None:
    result = tools["search_cars"].invoke({"max_price": 20_000_000})
    assert result["error"] == "invalid_filters"
    assert "currency" in result["message"]


def test_search_cars_with_unknown_category_is_rejected(
    tools: dict[str, BaseTool],
) -> None:
    with pytest.raises(ValidationError):
        tools["search_cars"].invoke({"category": "spaceship"})


def test_get_car_details_with_existing_id_returns_the_full_record(
    tools: dict[str, BaseTool],
) -> None:
    car = tools["get_car_details"].invoke({"car_id": "C004"})
    assert car["model"] == "Hilux" and car["price"] == 38_000


def test_get_car_details_with_unknown_id_returns_an_error_not_a_guess(
    tools: dict[str, BaseTool],
) -> None:
    assert tools["get_car_details"].invoke({"car_id": "C999"})["error"] == "not_found"


def test_register_lead_with_valid_data_appends_one_record(
    tools: dict[str, BaseTool], tmp_path: Path
) -> None:
    result = tools["register_lead"].invoke(
        {
            "name": "Ana Pérez",
            "phone": "+54 351 555 0100",
            "source": "instagram",
            "car_id": "C004",
        }
    )
    assert result["status"] == "registered"
    (record,) = read_lines(tmp_path / "leads.jsonl")
    assert record["lead_id"] == result["lead_id"]
    assert record["created_at"] == FIXED_NOW.isoformat()
    assert record["source"] == "instagram" and record["car_id"] == "C004"


def test_register_lead_with_malformed_phone_is_rejected_and_writes_nothing(
    tools: dict[str, BaseTool], tmp_path: Path
) -> None:
    with pytest.raises(ValidationError):
        tools["register_lead"].invoke(
            {"name": "Ana", "phone": "hola", "source": "whatsapp"}
        )
    assert not (tmp_path / "leads.jsonl").exists()


def test_register_lead_with_unknown_source_is_rejected(
    tools: dict[str, BaseTool],
) -> None:
    with pytest.raises(ValidationError):
        tools["register_lead"].invoke(
            {"name": "Ana", "phone": "3515550100", "source": "billboard"}
        )


def test_register_lead_with_unknown_car_returns_error_and_writes_nothing(
    tools: dict[str, BaseTool], tmp_path: Path
) -> None:
    result = tools["register_lead"].invoke(
        {
            "name": "Ana",
            "phone": "3515550100",
            "source": "website",
            "car_id": "C999",
        }
    )
    assert result["error"] == "unknown_car_id"
    assert not (tmp_path / "leads.jsonl").exists()


def test_handoff_to_human_with_valid_reason_queues_one_record(
    tools: dict[str, BaseTool], tmp_path: Path
) -> None:
    result = tools["handoff_to_human"].invoke(
        {"reason": "negotiate_price", "summary": "Wants a discount on C004"}
    )
    assert result["status"] == "queued"
    (record,) = read_lines(tmp_path / "handoffs.jsonl")
    assert record["reason"] == "negotiate_price"
    assert record["created_at"] == FIXED_NOW.isoformat()


def test_handoff_to_human_with_unknown_reason_is_rejected(
    tools: dict[str, BaseTool],
) -> None:
    with pytest.raises(ValidationError):
        tools["handoff_to_human"].invoke({"reason": "chat", "summary": "just chatting"})
