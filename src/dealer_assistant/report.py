"""Readable Markdown report of an exam run: failures first, infrastructure errors apart."""

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from dealer_assistant.exam import CaseRun, Failure

Graded = tuple[CaseRun, list[Failure]]


def _status(run: CaseRun, failures: list[Failure]) -> str:
    if run.error is not None:
        return "error"
    return "fail" if failures else "pass"


def _transcript(run: CaseRun) -> list[str]:
    lines: list[str] = []
    for customer, reply in zip(run.customer_messages, run.replies, strict=False):
        lines += [f"- **Cliente:** {customer}", f"- **Asistente:** {reply.strip()}"]
    calls = ", ".join(f"{name}({args})" for name, args in run.tool_calls) or "(ninguna)"
    return [*lines, f"- **Herramientas usadas:** {calls}"]


def _per_case_table(graded: Sequence[Graded]) -> list[str]:
    stats: dict[str, dict[str, int]] = {}
    types: dict[str, str] = {}
    for run, failures in graded:
        types[run.case_id] = run.case_type
        counts = stats.setdefault(run.case_id, {"pass": 0, "fail": 0, "error": 0})
        counts[_status(run, failures)] += 1
    lines = [
        "| caso | tipo | aprobadas | puntuadas | con error |",
        "|---|---|---|---|---|",
    ]
    for case_id in sorted(stats):
        counts = stats[case_id]
        scored = counts["pass"] + counts["fail"]
        lines.append(
            f"| {case_id} | {types[case_id]} | {counts['pass']} | {scored} | {counts['error']} |"
        )
    return lines


def build_report(name: str, graded: Sequence[Graded]) -> str:
    ordered = sorted(graded, key=lambda item: (item[0].case_id, item[0].attempt))
    counts = Counter(_status(run, failures) for run, failures in ordered)
    scored = counts["pass"] + counts["fail"]

    lines = [
        f"# Informe del examen: {name}",
        "",
        "## Resumen",
        f"- Ejecuciones: {len(ordered)}",
        f"- Aprobadas: {counts['pass']} de {scored} puntuadas",
        f"- Con error de servicio (sin puntuar): {counts['error']}",
        "",
        "## Lo que falló (mira esto primero)",
        "",
    ]
    failed = [
        (run, failures) for run, failures in ordered if _status(run, failures) == "fail"
    ]
    if not failed:
        lines += ["Ninguna ejecución puntuada falló.", ""]
    for run, failures in failed:
        lines += [f"### {run.case_id} ({run.case_type}), intento {run.attempt} ❌"]
        lines += _transcript(run)
        lines += [f"- **{f.check}:** {f.detail}" for f in failures] + [""]

    errors = [run for run, failures in ordered if _status(run, failures) == "error"]
    if errors:
        by_message: dict[str, list[str]] = {}
        for run in errors:
            by_message.setdefault(str(run.error)[:200], []).append(run.case_id)
        lines += ["## Errores de servicio (sin puntuar, no son fallos del agente)", ""]
        lines += [
            f"- {', '.join(ids)}: {message}" for message, ids in by_message.items()
        ]
        lines.append("")

    lines += ["## Resultado por caso", "", *_per_case_table(ordered)]
    return "\n".join(lines) + "\n"


def write_report(directory: Path, name: str, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.md"
    path.write_text(content, encoding="utf-8")
    return path
