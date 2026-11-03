"""Play the exam cases against the real agent and write a readable report.

    uv run python -m dealer_assistant.evaluate --name first-run --repeat 1

Everything stays local: leads and handoffs go to a temporary folder, and the report is git-ignored.
"""

import argparse
import logging
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from dealer_assistant.agent import PROVIDER_ERRORS, build_agent, build_model, chat_turn
from dealer_assistant.config import ROOT, Settings
from dealer_assistant.exam import (
    CASES_PATH,
    Case,
    CaseRun,
    Failure,
    check_run,
    load_cases,
)
from dealer_assistant.inventory import Inventory
from dealer_assistant.report import build_report, write_report
from dealer_assistant.tools import make_tools

REPORTS_DIR = ROOT / "evals" / "reports"


def run_case(
    case: Case,
    attempt: int,
    *,
    model_factory: Callable[[], BaseChatModel],
    inventory: Inventory,
    pause_s: float,
) -> CaseRun:
    """Play one conversation with a fresh agent (and a fresh, throwaway lead file)."""
    replies: list[str] = []
    calls: list[tuple[str, dict[str, object]]] = []
    outputs: list[str] = []
    error: str | None = None
    with tempfile.TemporaryDirectory() as tmp:
        tools = make_tools(
            inventory, Path(tmp) / "leads.jsonl", Path(tmp) / "handoffs.jsonl"
        )
        agent = build_agent(model_factory(), tools)
        thread_id = uuid.uuid4().hex
        try:
            for message in case.turns:
                if pause_s > 0:
                    time.sleep(pause_s)
                turn = chat_turn(agent, thread_id, message)
                replies.append(turn.reply)
                calls.extend(turn.tool_calls)
                outputs.extend(turn.tool_outputs)
        except PROVIDER_ERRORS as exc:
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
    return CaseRun(
        case_id=case.id,
        case_type=case.type,
        attempt=attempt,
        customer_messages=list(case.turns),
        replies=replies,
        tool_calls=calls,
        tool_outputs=outputs,
        error=error,
    )


def progress_mark(run: CaseRun, failures: list[Failure]) -> str:
    """One-line verdict printed while the exam runs."""
    if run.error is not None:
        return f"⚠ error ({run.error[:70]})"
    if failures:
        return "❌ " + ", ".join(failure.check for failure in failures)
    return "✅"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sales-agent exam")
    parser.add_argument(
        "--name", default="exam", help="name of the run (used for the report file)"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="times each case is played"
    )
    parser.add_argument(
        "--pause", type=float, default=2.0, help="seconds before each customer message"
    )
    parser.add_argument("--only", default="", help="comma-separated case ids to run")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    settings = Settings()  # type: ignore[call-arg]  # google_api_key comes from the environment
    inventory = Inventory.from_csv(settings.stock_path)
    wanted = {item.strip() for item in args.only.split(",") if item.strip()}
    cases = [case for case in load_cases(args.cases) if not wanted or case.id in wanted]

    graded: list[tuple[CaseRun, list[Failure]]] = []
    try:
        for attempt in range(1, args.repeat + 1):
            for case in cases:
                run = run_case(
                    case,
                    attempt,
                    model_factory=lambda: build_model(settings),
                    inventory=inventory,
                    pause_s=args.pause,
                )
                failures = check_run(case, run)
                graded.append((run, failures))
                print(
                    f"{case.id} #{attempt}: {progress_mark(run, failures)}", flush=True
                )
    finally:
        # Always write what we have, even if something crashes or you press Ctrl+C.
        path = write_report(REPORTS_DIR, args.name, build_report(args.name, graded))
        print(f"\nInforme: {path}")


if __name__ == "__main__":
    main()
