"""Read-only MCP server (stdio): lets an AI assistant query the dealership stock.

Least privilege: only `search_cars` and `get_car_details` are exposed. Leads and handoffs
write personal data and need human confirmation, so they are NOT available here.
It needs no model API key: only the stock file.
"""

import logging
import re
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, ValidationError

from dealer_assistant.config import StockSettings
from dealer_assistant.inventory import MAX_RESULTS, Inventory
from dealer_assistant.models import (
    Car,
    Category,
    Condition,
    Currency,
    Fuel,
    Transmission,
)
from dealer_assistant.tools import SearchCarsInput

logger = logging.getLogger("dealer_assistant.mcp")

MAX_TEXT_CHARS = 40
CAR_ID_PATTERN = re.compile(r"\s*[A-Za-z0-9]{1,10}\s*")

READ_ONLY = ToolAnnotations(
    read_only_hint=True, idempotent_hint=True, open_world_hint=False
)


class SearchResult(BaseModel):
    """One block with an explicit count, so an empty result is never ambiguous."""

    count: int
    cars: list[Car]
    applied_filters: dict[str, object]


def _problems(error: ValidationError) -> str:
    """Field names and rules only, never the values the caller sent."""
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors(include_url=False, include_input=False)
    )


def build_server(inventory: Inventory) -> MCPServer:
    mcp = MCPServer("dealer-stock")

    @mcp.tool(
        name="search_cars",
        title="Search dealership stock",
        description=(
            "Search the dealership stock with exact filters and return up to `limit` cars, "
            "newest first. `count` 0 means NO car matches: never invent other cars. "
            "Each price is in the car's own currency (ARS or USD) and is never converted. "
            "`max_price` requires `currency`. `applied_filters` echoes what was applied."
        ),
        annotations=READ_ONLY,
    )
    def search_cars(
        make: str | None = None,
        model: str | None = None,
        category: Category | None = None,
        min_year: Annotated[int | None, Field(ge=1990, le=2100)] = None,
        max_price: Annotated[int | None, Field(gt=0)] = None,
        currency: Currency | None = None,
        max_km: Annotated[int | None, Field(ge=0)] = None,
        transmission: Transmission | None = None,
        fuel: Fuel | None = None,
        condition: Condition | None = None,
        include_unavailable: bool = False,
        limit: Annotated[int, Field(ge=1, le=MAX_RESULTS)] = 5,
    ) -> SearchResult:
        for name, text in (("make", make), ("model", model)):
            if text is not None and len(text) > MAX_TEXT_CHARS:
                raise ToolError(f"Invalid filters: {name} is too long.")
        try:
            filters = SearchCarsInput(
                make=make,
                model=model,
                category=category,
                min_year=min_year,
                max_price=max_price,
                currency=currency,
                max_km=max_km,
                transmission=transmission,
                fuel=fuel,
                condition=condition,
                include_unavailable=include_unavailable,
                limit=limit,
            )
        except ValidationError as error:
            raise ToolError(f"Invalid filters: {_problems(error)}") from error
        search_args = filters.model_dump(exclude={"include_unavailable"})
        try:
            cars = inventory.search(
                **search_args, only_available=not filters.include_unavailable
            )
        except ValueError as error:  # cross-field rule, e.g. max_price without currency
            raise ToolError(f"Invalid filters: {error}") from error
        logger.info("search_cars results=%d", len(cars))
        applied = {
            key: value for key, value in search_args.items() if value is not None
        }
        return SearchResult(count=len(cars), cars=cars, applied_filters=applied)

    @mcp.tool(
        name="get_car_details",
        title="Get one car by id",
        description="Return the full record of one car by its id (for example 'C004').",
        annotations=READ_ONLY,
    )
    def get_car_details(car_id: str) -> Car:
        # Validated here, not with a Field pattern: the SDK would echo the rejected value.
        if not CAR_ID_PATTERN.fullmatch(car_id):
            raise ToolError("Invalid car id format.")
        car = inventory.get(car_id)
        if car is None:
            raise ToolError("No car with that id.")
        return car

    return mcp


def main() -> None:
    # stdout is the MCP wire: log to stderr only.
    logging.basicConfig(level=logging.INFO)
    stock_path = StockSettings().stock_path
    inventory = Inventory.from_csv(stock_path)
    logger.info("stock loaded: %s (%d cars)", stock_path.name, len(inventory))
    build_server(inventory).run()


if __name__ == "__main__":
    main()
