---
name: run-exam
description: Run the dealer-assistant exam against the real model and summarize pass/fail, real failures and infrastructure errors. Use after changing the system prompt, tools or exam cases.
disable-model-invocation: true
argument-hint: "[run-name] [--only c01,c02]"
---

# Run the exam

The exam calls the real model and costs money, so run it only when the user invokes this skill.

## Steps
1. Run the offline checks first and stop if they fail:
   `uv run ruff check . && uv run mypy --strict src tests && uv run pytest -q`
2. Pick a run name: the first argument, or `exam-YYYYMMDD-HHMM`. Do not reuse an existing report name.
3. Run ONE exam at a time (never in parallel, it triggers rate limits):
   `uv run python -m dealer_assistant.evaluate --name NAME --repeat 3 --pause 2`
   Add `--only ...` if the user passed case ids.
4. Read the report at `evals/reports/NAME.md` and summarize:
   - Passed / total, per case (a case with repeats that fails once is flaky: say so).
   - **Real failures**: what the assistant said or called versus what the case expected.
   - **Infrastructure errors** (429/503): list them separately. They are not quality failures; offer to re-run only those cases with `--only`.
5. If a case fails, propose a fix (prompt, tool or expectation) but do not apply it without asking. Never weaken an expectation just to make it pass.

## Rules
- Never read, print or write `.env`. Keys come from the environment.
- Reports may contain conversation text: do not paste them whole, and keep names and phones out of the summary.
- If you could not run the exam, say so explicitly. Do not report results you did not see.
