# dealer-assistant

A sales assistant for a car dealership. It answers customer questions **using the real stock**,
registers interested buyers together with **where they came from**, and hands off to a person
whatever it must not decide (price negotiation, credit approval, test drives).

> Status: **step 1b — tools plus an agent you can chat with in the terminal.** The evaluation and
> the service layer come next. The stock in `data/stock_sample.csv` is fictional.

## Try it

```bash
uv sync
cp .env.example .env     # add GOOGLE_API_KEY; never commit .env
uv run python -m dealer_assistant.cli
```

By default only the conversation is shown. For debugging: `--show-tools` prints every tool call the
agent makes (the best way to see how it decides) and `--verbose` adds timings and every model call
and retry.
The agent (LangChain `create_agent` on LangGraph) keeps the conversation history per thread, caps the
number of steps per message, validates every tool argument and cannot state a price, year, km or
availability that did not come from a tool.

## Why tools instead of a document search

A car is structured data: price, year, kilometres, availability. Those must be looked up with exact
filters, not retrieved "by meaning". The agent therefore gets tools and is forbidden to state any
price, year, km or availability that did not come from one of them. Policy-style questions
(financing, trade-ins, FAQs) are documents and are a job for retrieval (see the `rag-docs` project).

## The tools

| Tool | Purpose |
|---|---|
| `search_cars` | Exact filters: make, model, category, year, price + currency, km, transmission, fuel, condition |
| `get_car_details` | Full record of one car by id |
| `register_lead` | Saves name, phone and **source** (instagram, facebook, whatsapp, marketplace, website, walk_in, other) |
| `handoff_to_human` | Queues a salesperson for price negotiation, test drive, financing, trade-in appraisal, complaints |

Design decisions

- **Prices are never converted** between ARS and USD; a price limit requires a currency.
- Every tool input is validated with Pydantic (enums for categories and sources, phone format, length limits).
- Matching ignores case and accents (`citroen` finds `Citroën`).
- Dependencies (stock, output files, clock) are injected, so every tool is tested offline.
- Customer data is written to local JSONL files that are git-ignored.

## Roadmap

1. **1a Tools** (done) · **1b Agent and terminal chat** (done) · 1c Evaluation: every number the
   agent states must come from a tool result.
2. Web service (FastAPI) and lead logging with source tags.
3. "Shadow mode" pilot: the assistant drafts replies, a person sends them.
4. A small report: advertising cost per lead and per sale.

## Real data

Real dealership stock is messy (make, model, trim and gearbox often share one title field, and prices
mix pesos and dollars), so loading it needs a normalisation step. Real stock and customer data stay in
`data/private/` or `data/stock_real.csv`, which are git-ignored.

## The exam

`evals/cases.json` holds 20 scripted conversations (stock lookups, no match, missing currency,
slang and typos, follow-ups, negotiation and financing, lead capture, gibberish, off-topic,
prompt injection, prompt leak). Checks are automatic and need no judge model:

| Check | What it catches |
|---|---|
| **Grounding** | any price, year or km in a reply that is not in a tool result or in the customer's own message |
| Expected / forbidden tools | not looking up the stock; registering a lead without contact data |
| Reply content | must (or must not) contain given text, e.g. ask for the currency, never leak the rules |

```bash
uv run python -m dealer_assistant.evaluate --name first-run --repeat 1
```

Each conversation uses a fresh agent and throwaway lead files. The report (`evals/reports/`, git-ignored)
lists failures first with the full conversation, and model-service errors apart, so a 503 is never
confused with an agent mistake. Use `--repeat 3`: without a fixed temperature the same case can pass once
and fail the next time.

Known limits: grounding looks at numbers only (not at claims like "automatic"), small numbers (10 or less)
are ignored, and phone numbers written with different spacing than the customer's may be flagged. The
expected phrases are in Spanish and can be too strict; adjust `cases.json` when a failure is a false alarm.

## Privacy note

Customer messages can contain names and phone numbers. If LangSmith tracing is enabled
(`LANGSMITH_TRACING=true`), those conversations are sent to LangSmith. Keep tracing off, or agree on
a data policy, before connecting real customers.

## Development

```bash
uv sync
uv run ruff format . && uv run ruff check .
uv run mypy --strict src tests
uv run pytest
```
