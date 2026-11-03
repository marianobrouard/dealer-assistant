"""The tools the sales agent can call.

They are plain functions with validated inputs, so they can be tested without any LLM
or network. Dependencies (stock, output files, clock) are injected by `make_tools`.
"""

import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from dealer_assistant.inventory import MAX_RESULTS, Inventory
from dealer_assistant.models import Category, Condition, Currency, Fuel, Transmission

LeadSource = Literal[
    "instagram", "facebook", "whatsapp", "marketplace", "website", "walk_in", "other"
]
HandoffReason = Literal[
    "negotiate_price",
    "test_drive",
    "financing_details",
    "trade_in_appraisal",
    "complaint",
    "other",
]


class SearchCarsInput(BaseModel):
    make: str | None = Field(
        None, description="Brand, e.g. 'Toyota'. Partial text is fine."
    )
    model: str | None = Field(
        None, description="Model or trim text, e.g. 'Hilux' or 'SRV'."
    )
    category: Category | None = None
    min_year: int | None = Field(None, ge=1990, le=2100)
    max_price: int | None = Field(
        None,
        gt=0,
        description="Maximum price, in the given currency. Requires currency.",
    )
    currency: Currency | None = Field(None, description="ARS (pesos) or USD (dollars).")
    max_km: int | None = Field(None, ge=0)
    transmission: Transmission | None = Field(
        None, description="MT manual, AT automatic, CVT."
    )
    fuel: Fuel | None = None
    condition: Condition | None = Field(None, description="'used' or 'new' (0 km).")
    include_unavailable: bool = Field(
        False, description="Also list reserved and sold cars."
    )
    limit: int = Field(5, ge=1, le=MAX_RESULTS)


class LeadInput(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    phone: str = Field(pattern=r"^\+?[0-9 ()-]{8,20}$")
    source: LeadSource
    car_id: str | None = None
    note: str | None = Field(None, max_length=300)


class HandoffInput(BaseModel):
    reason: HandoffReason
    summary: str = Field(min_length=5, max_length=500)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_tools(
    inventory: Inventory,
    leads_path: Path,
    handoffs_path: Path,
    clock: Callable[[], datetime] = _utc_now,
) -> list[BaseTool]:
    @tool("search_cars", args_schema=SearchCarsInput)
    def search_cars(
        make: str | None = None,
        model: str | None = None,
        category: Category | None = None,
        min_year: int | None = None,
        max_price: int | None = None,
        currency: Currency | None = None,
        max_km: int | None = None,
        transmission: Transmission | None = None,
        fuel: Fuel | None = None,
        condition: Condition | None = None,
        include_unavailable: bool = False,
        limit: int = 5,
    ) -> list[dict[str, object]] | dict[str, object]:
        """Search the dealership stock with exact filters and return up to `limit` cars,
        newest first. An empty list means NO car matches: never suggest or invent other
        cars. Each price is in the car's own currency (ARS or USD) and is never converted.
        If the filters are invalid it returns {"error": ..., "message": ...}: read the
        message, then fix the arguments or ask the customer."""
        try:
            cars = inventory.search(
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
                only_available=not include_unavailable,
                limit=limit,
            )
        except ValueError as exc:
            return {"error": "invalid_filters", "message": str(exc)}
        return [car.model_dump() for car in cars]

    @tool("get_car_details")
    def get_car_details(car_id: str) -> dict[str, object]:
        """Return the full record of one car by its id (for example 'C004'),
        or an error if that id does not exist."""
        car = inventory.get(car_id)
        return car.model_dump() if car else {"error": "not_found", "car_id": car_id}

    @tool("register_lead", args_schema=LeadInput)
    def register_lead(
        name: str,
        phone: str,
        source: LeadSource,
        car_id: str | None = None,
        note: str | None = None,
    ) -> dict[str, object]:
        """Save the contact data of a prospective buyer. Call it only after the customer
        gave their name and phone and agreed to be contacted. `source` is where the
        customer found the dealership."""
        if car_id is not None and inventory.get(car_id) is None:
            return {"error": "unknown_car_id", "car_id": car_id}
        lead_id = _new_id()
        _append_jsonl(
            leads_path,
            {
                "lead_id": lead_id,
                "created_at": clock().isoformat(),
                "name": name,
                "phone": phone,
                "source": source,
                "car_id": car_id,
                "note": note,
            },
        )
        return {"status": "registered", "lead_id": lead_id}

    @tool("handoff_to_human", args_schema=HandoffInput)
    def handoff_to_human(reason: HandoffReason, summary: str) -> dict[str, object]:
        """Ask a salesperson to take over the conversation. Use it for anything the
        assistant must not decide: price negotiation, test-drive scheduling, credit
        approval, trade-in appraisal and complaints."""
        handoff_id = _new_id()
        _append_jsonl(
            handoffs_path,
            {
                "handoff_id": handoff_id,
                "created_at": clock().isoformat(),
                "reason": reason,
                "summary": summary,
            },
        )
        return {"status": "queued", "handoff_id": handoff_id}

    return [search_cars, get_car_details, register_lead, handoff_to_human]
