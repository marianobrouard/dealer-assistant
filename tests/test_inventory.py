from pathlib import Path

import pytest

from dealer_assistant.inventory import MAX_RESULTS, Inventory, normalize

STOCK_PATH = Path(__file__).resolve().parents[1] / "data" / "stock_sample.csv"


@pytest.fixture(scope="module")
def inventory() -> Inventory:
    return Inventory.from_csv(STOCK_PATH)


def ids(cars: list) -> list[str]:  # type: ignore[type-arg]  # helper for readable asserts
    return [car.id for car in cars]


def test_from_csv_with_sample_stock_loads_thirty_cars(inventory: Inventory) -> None:
    assert len(inventory) == 30


def test_normalize_with_accents_and_uppercase_returns_plain_lowercase() -> None:
    assert normalize("  Citroën ") == "citroen"


def test_search_by_make_ignores_case_and_returns_only_that_brand(
    inventory: Inventory,
) -> None:
    cars = inventory.search(make="TOYOTA")
    assert len(cars) == 4
    assert all(car.make == "Toyota" for car in cars)


def test_search_by_make_without_accent_finds_accented_brand(
    inventory: Inventory,
) -> None:
    assert ids(inventory.search(make="citroen")) == ["C023"]


def test_search_by_model_matches_model_and_trim_text(inventory: Inventory) -> None:
    assert ids(inventory.search(model="hilux")) == ["C004"]
    assert ids(inventory.search(model="SRV")) == ["C004"]


def test_search_with_max_price_but_no_currency_raises(inventory: Inventory) -> None:
    with pytest.raises(ValueError, match="currency is required"):
        inventory.search(max_price=20_000_000)


def test_search_with_max_price_keeps_only_cars_priced_in_that_currency(
    inventory: Inventory,
) -> None:
    cars = inventory.search(max_price=20_000_000, currency="ARS", limit=10)
    assert cars
    assert all(car.currency == "ARS" and car.price <= 20_000_000 for car in cars)
    assert "C004" not in ids(cars)  # priced in USD
    assert "C030" not in ids(cars)  # priced in USD


def test_search_in_usd_never_compares_against_pesos(inventory: Inventory) -> None:
    cars = inventory.search(max_price=40_000, currency="USD")
    assert set(ids(cars)) == {"C004", "C030"}


def test_search_by_default_hides_sold_and_reserved_cars(inventory: Inventory) -> None:
    assert inventory.search(model="2008") == []
    assert ids(inventory.search(model="2008", only_available=False)) == ["C010"]


def test_search_orders_newest_first_then_fewest_km(inventory: Inventory) -> None:
    assert ids(inventory.search(make="toyota")) == ["C022", "C021", "C003", "C004"]


def test_search_with_limit_above_the_maximum_is_capped(inventory: Inventory) -> None:
    cars = inventory.search(only_available=False, limit=100)
    assert len(cars) == MAX_RESULTS


def test_search_with_filters_that_match_nothing_returns_empty_list(
    inventory: Inventory,
) -> None:
    assert inventory.search(make="toyota", category="motorcycle") == []


def test_search_with_combined_filters_applies_all_of_them(inventory: Inventory) -> None:
    cars = inventory.search(
        category="suv", transmission="AT", max_km=60_000, min_year=2022
    )
    assert set(ids(cars)) == {"C016", "C025"}


def test_get_with_lowercase_id_finds_the_car(inventory: Inventory) -> None:
    car = inventory.get("c001")
    assert car is not None and car.make == "Volkswagen"


def test_get_with_unknown_id_returns_none(inventory: Inventory) -> None:
    assert inventory.get("X999") is None


def test_inventory_with_duplicate_ids_raises(inventory: Inventory) -> None:
    car = inventory.get("C001")
    assert car is not None
    with pytest.raises(ValueError, match="duplicate"):
        Inventory([car, car])
