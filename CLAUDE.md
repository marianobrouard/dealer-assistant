# dealer-assistant

Sales assistant for a car dealership: an agent that answers customers using the REAL stock,
registers leads with their source, and hands off to a human what it must not decide.
Python 3.12+, managed with `uv`. Code, comments and commits in English; the assistant itself
talks to customers in Spanish.

## Commands
- Install: `uv sync`
- Format and lint: `uv run ruff format . && uv run ruff check .`
- Types: `uv run mypy --strict src tests`
- Tests (offline, no API keys): `uv run pytest`
- Chat: `uv run python -m dealer_assistant.cli [--show-tools] [--verbose]`
- MCP server (stdio, read-only: `search_cars`, `get_car_details`; no API key, reads `STOCK_PATH`):
  `uv run python -m dealer_assistant.mcp_server`. Register in Claude Code:
  `claude mcp add dealer-stock -- uv run --directory <this repo> python -m dealer_assistant.mcp_server`
  (mcp SDK 2.x: `MCPServer`, `ToolError`). Never expose `register_lead`/`handoff_to_human` there.
- Exam (uses the real model; run one at a time):
  `uv run python -m dealer_assistant.evaluate --name NAME [--repeat 3] [--pause 2] [--only c01,c02]`

## Rules for this project
- Never read, print or write `.env`. Keys go in the environment.
- Real stock and customer data live only in `data/private/` or `data/stock_real.csv`
  (git-ignored). The repository contains the fictional sample stock only.
- The assistant must never state a price, year, km or availability that did not come from a tool.
- Prices are never converted between ARS and USD (no invented exchange rate).
- Leads contain personal data: no names or phones in logs, tests or commits.
- Anything the assistant must not decide (price negotiation, credit approval, trade-in
  appraisal, test-drive scheduling) goes through `handoff_to_human`.
- Do not set `temperature`/`top_p` on the model: current models use fixed sampling defaults
  (Gemini warns and ignores them; some Claude models reject non-default values).
- Unit tests must not call any LLM or network.
