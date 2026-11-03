from pathlib import Path

from dealer_assistant.exam import CaseRun, Failure
from dealer_assistant.report import build_report, write_report


def run(case_id: str, attempt: int = 1, error: str | None = None) -> CaseRun:
    return CaseRun(
        case_id=case_id,
        case_type="stock_lookup",
        attempt=attempt,
        customer_messages=["¿Tienen una pickup?"],
        replies=["Sí, una Hilux."],
        tool_calls=[("search_cars", {"category": "pickup"})],
        tool_outputs=["[]"],
        error=error,
    )


def test_build_report_with_all_passing_runs_says_nothing_failed() -> None:
    report = build_report("demo", [(run("c01"), []), (run("c02"), [])])
    assert "Ninguna ejecución puntuada falló." in report
    assert "Aprobadas: 2 de 2 puntuadas" in report


def test_build_report_with_a_failure_shows_the_conversation_and_the_checks() -> None:
    failed = (
        run("c03"),
        [Failure("grounding", "numbers not in any tool result: 31000")],
    )
    report = build_report("demo", [(run("c01"), []), failed])
    failures_part = report.split("## Resultado por caso")[0]
    assert "### c03" in failures_part
    assert "¿Tienen una pickup?" in failures_part and "Sí, una Hilux." in failures_part
    assert "grounding" in failures_part and "31000" in failures_part
    assert "Aprobadas: 1 de 2 puntuadas" in report


def test_build_report_with_service_errors_counts_them_apart_from_failures() -> None:
    erroring: list[tuple[CaseRun, list[Failure]]] = [
        (run("c04", error="ServerError: 503"), []),
        (run("c05", error="ServerError: 503"), []),
    ]
    report = build_report("demo", [(run("c01"), []), *erroring])
    assert "Con error de servicio (sin puntuar): 2" in report
    assert "c04, c05: ServerError: 503" in report
    assert "### c04" not in report


def test_build_report_with_repeated_attempts_summarizes_per_case() -> None:
    graded = [
        (run("c01", 1), []),
        (run("c01", 2), [Failure("expected_tools", "not called: search_cars")]),
        (run("c01", 3), []),
    ]
    report = build_report("demo", graded)
    assert "| c01 | stock_lookup | 2 | 3 | 0 |" in report


def test_write_report_with_unsafe_name_creates_a_sanitized_file(tmp_path: Path) -> None:
    path = write_report(tmp_path / "reports", "run/one two", "hola")
    assert path.name == "run_one_two.md"
    assert path.read_text(encoding="utf-8") == "hola"
