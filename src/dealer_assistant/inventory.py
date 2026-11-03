"""Load the stock and search it with exact filters (no LLM involved)."""

import csv
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path

from dealer_assistant.models import (
    Car,
    Category,
    Condition,
    Currency,
    Fuel,
    Transmission,
)

MAX_RESULTS = 10


def normalize(text: str) -> str:
    """Lowercase and strip accents, so 'Citroën' matches 'citroen'."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return plain.casefold().strip()


class Inventory:
    def __init__(self, cars: Sequence[Car]) -> None:
        ids = [car.id for car in cars]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate car ids in stock")
        self._cars = list(cars)
        self._by_id = {car.id: car for car in cars}

    @classmethod
    def from_csv(cls, path: Path) -> "Inventory":
        with path.open(encoding="utf-8", newline="") as handle:
            return cls([Car.model_validate(row) for row in csv.DictReader(handle)])

    def __len__(self) -> int:
        return len(self._cars)

    def get(self, car_id: str) -> Car | None:
        return self._by_id.get(car_id.strip().upper())

    def search(
        self,
        *,
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
        only_available: bool = True,
        limit: int = 5,
    ) -> list[Car]:
        """Cars matching every given filter, newest first (then fewest km).

        Prices are never converted between currencies, so a price limit needs a currency.
        """
        if max_price is not None and currency is None:
            raise ValueError("currency is required when max_price is given")

        predicates: list[Callable[[Car], bool]] = []
        if make:
            make_key = normalize(make)
            predicates.append(lambda c: make_key in normalize(c.make))
        if model:
            model_key = normalize(model)
            predicates.append(lambda c: model_key in normalize(f"{c.model} {c.trim}"))
        if category:
            predicates.append(lambda c: c.category == category)
        if min_year is not None:
            predicates.append(lambda c: c.year >= min_year)
        if currency:
            predicates.append(lambda c: c.currency == currency)
        if max_price is not None:
            predicates.append(lambda c: c.price <= max_price)
        if max_km is not None:
            predicates.append(lambda c: c.km <= max_km)
        if transmission:
            predicates.append(lambda c: c.transmission == transmission)
        if fuel:
            predicates.append(lambda c: c.fuel == fuel)
        if condition:
            predicates.append(lambda c: c.condition == condition)
        if only_available:
            predicates.append(lambda c: c.status == "available")

        matches = [car for car in self._cars if all(p(car) for p in predicates)]
        matches.sort(key=lambda c: (-c.year, c.km))
        return matches[: max(1, min(limit, MAX_RESULTS))]
