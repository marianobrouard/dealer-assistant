import csv
from pathlib import Path

STOCK_PATH = Path(__file__).resolve().parents[1] / "data" / "stock_sample.csv"


def load_rows() -> list[dict[str, str]]:
    with STOCK_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_sample_stock_covers_both_currencies_and_every_status() -> None:
    rows = load_rows()
    assert {row["currency"] for row in rows} == {"ARS", "USD"}
    assert {row["status"] for row in rows} == {"available", "reserved", "sold"}


def test_sample_stock_has_enough_available_cars_to_search() -> None:
    available = [row for row in load_rows() if row["status"] == "available"]
    assert len(available) >= 25


def test_sample_stock_new_cars_have_zero_km() -> None:
    assert all(row["km"] == "0" for row in load_rows() if row["condition"] == "new")
