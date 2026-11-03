from decimal import Decimal

from dealer_assistant.grounding import extract_numbers, ungrounded_numbers


def test_extract_numbers_with_argentine_thousands_dots_returns_plain_values() -> None:
    found = extract_numbers("Precio: 27.900.000 ARS, 109.000 km, año 2019")
    assert found == {Decimal(27_900_000), Decimal(109_000), Decimal(2019)}


def test_extract_numbers_with_json_style_text_reads_unseparated_values() -> None:
    found = extract_numbers('{"year": 2021, "km": 72000, "price": 38000}')
    assert found == {Decimal(2021), Decimal(72_000), Decimal(38_000)}


def test_extract_numbers_with_mil_and_millones_scales_the_value() -> None:
    assert extract_numbers("unos 38 mil dólares") == {Decimal(38_000)}
    assert extract_numbers("27,9 millones de pesos") == {Decimal(27_900_000)}
    assert extract_numbers("20 millones") == {Decimal(20_000_000)}


def test_extract_numbers_ignores_small_numbers_that_are_not_claims() -> None:
    assert extract_numbers("3 autos 4x4 motor 1.4 y 2,8") == set()


def test_ungrounded_numbers_with_value_from_a_tool_result_returns_nothing() -> None:
    evidence = ['[{"price": 38000, "year": 2021}]']
    assert ungrounded_numbers("La Hilux 2021 sale 38.000 USD", evidence) == []


def test_ungrounded_numbers_with_an_invented_price_returns_it() -> None:
    evidence = ['[{"price": 38000}]']
    assert ungrounded_numbers("Te la dejo en 31.000 USD", evidence) == [Decimal(31_000)]


def test_ungrounded_numbers_with_a_figure_the_customer_gave_accepts_it() -> None:
    reply = "Con tu presupuesto de 20.000.000 puedo mostrarte varias opciones"
    assert ungrounded_numbers(reply, ["hasta 20 millones"]) == []


def test_ungrounded_numbers_with_an_invented_installment_plan_flags_it() -> None:
    assert ungrounded_numbers(
        "Podés pagarlo en 60 cuotas", ["tienen financiación?"]
    ) == [Decimal(60)]


def test_ungrounded_numbers_with_no_evidence_flags_every_big_number() -> None:
    assert ungrounded_numbers("Año 2020 y 91.000 km", []) == [
        Decimal(2020),
        Decimal(91_000),
    ]
