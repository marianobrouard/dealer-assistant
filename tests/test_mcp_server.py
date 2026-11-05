from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from dealer_assistant.inventory import Inventory
from dealer_assistant.mcp_server import build_server

STOCK_PATH = Path(__file__).resolve().parents[1] / "data" / "stock_sample.csv"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def server() -> Client:
    return Client(build_server(Inventory.from_csv(STOCK_PATH)))


def error_text(result: Any) -> str:
    return " ".join(block.text for block in result.content)


@pytest.mark.anyio
async def test_list_tools_exposes_only_two_read_only_tools() -> None:
    async with server() as client:
        tools = (await client.list_tools()).tools
    assert sorted(t.name for t in tools) == ["get_car_details", "search_cars"]
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)


@pytest.mark.anyio
async def test_search_with_no_match_returns_explicit_zero_count() -> None:
    async with server() as client:
        result = await client.call_tool("search_cars", {"make": "Ferrari"})
    assert not result.is_error
    assert result.structured_content is not None
    assert result.structured_content["count"] == 0
    assert result.structured_content["cars"] == []
    assert result.content, "an empty result must still send a visible content block"


@pytest.mark.anyio
async def test_search_limit_returns_exactly_that_many_newest_first() -> None:
    async with server() as client:
        result = await client.call_tool("search_cars", {"limit": 3})
    assert result.structured_content is not None
    cars = result.structured_content["cars"]
    assert result.structured_content["count"] == len(cars) == 3
    years = [car["year"] for car in cars]
    assert years == sorted(years, reverse=True)


@pytest.mark.anyio
async def test_search_hides_reserved_and_sold_unless_asked() -> None:
    async with server() as client:
        default = await client.call_tool("search_cars", {"limit": 10})
        everything = await client.call_tool(
            "search_cars", {"limit": 10, "include_unavailable": True}
        )
    assert default.structured_content is not None
    assert everything.structured_content is not None
    assert {c["status"] for c in default.structured_content["cars"]} == {"available"}
    assert {c["status"] for c in everything.structured_content["cars"]} - {"available"}


@pytest.mark.anyio
async def test_search_echoes_applied_filters() -> None:
    async with server() as client:
        result = await client.call_tool("search_cars", {"make": "Toyota", "limit": 2})
    assert result.structured_content is not None
    applied = result.structured_content["applied_filters"]
    assert applied["make"] == "Toyota" and applied["limit"] == 2


@pytest.mark.anyio
async def test_search_price_without_currency_is_a_clear_error() -> None:
    async with server() as client:
        result = await client.call_tool("search_cars", {"max_price": 20000})
    assert result.is_error
    assert "currency" in error_text(result)


@pytest.mark.anyio
async def test_search_limit_above_maximum_is_rejected() -> None:
    async with server() as client:
        result = await client.call_tool("search_cars", {"limit": 999})
    assert result.is_error
    assert "limit" in error_text(result)


@pytest.mark.anyio
async def test_get_car_details_normalizes_the_id_and_keeps_price_unconverted() -> None:
    async with server() as client:
        result = await client.call_tool("get_car_details", {"car_id": " c004 "})
    assert not result.is_error
    assert result.structured_content is not None
    assert result.structured_content["id"] == "C004"
    assert result.structured_content["currency"] in {"ARS", "USD"}


@pytest.mark.anyio
async def test_get_car_details_unknown_id_is_an_error() -> None:
    async with server() as client:
        result = await client.call_tool("get_car_details", {"car_id": "ZZ99"})
    assert result.is_error


@pytest.mark.anyio
async def test_get_car_details_malformed_id_is_rejected_and_not_echoed() -> None:
    async with server() as client:
        result = await client.call_tool(
            "get_car_details", {"car_id": "llamame al +54 351 555 1234"}
        )
    assert result.is_error
    assert "555" not in error_text(result)
