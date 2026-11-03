"""Chat with the assistant from the terminal: `uv run python -m dealer_assistant.cli`."""

import argparse
import logging
import time
import uuid

from dotenv import load_dotenv

from dealer_assistant.agent import (
    MAX_INPUT_CHARS,
    PROVIDER_ERRORS,
    build_agent,
    build_model,
    chat_turn,
)
from dealer_assistant.config import Settings
from dealer_assistant.inventory import Inventory
from dealer_assistant.tools import make_tools

EXIT_WORDS = {"salir", "exit", "quit"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dealership sales assistant (terminal)"
    )
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="print every tool call the agent makes (to see how it decides)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log every call to the model and every retry (to see where time goes)",
    )
    args = parser.parse_args()

    # Exports LANGSMITH_* from .env so tracing sees them (Settings does not).
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    settings = Settings()  # type: ignore[call-arg]  # google_api_key comes from the environment
    inventory = Inventory.from_csv(settings.stock_path)
    tools = make_tools(
        inventory,
        leads_path=settings.private_dir / "leads.jsonl",
        handoffs_path=settings.private_dir / "handoffs.jsonl",
    )
    agent = build_agent(build_model(settings), tools)
    thread_id = uuid.uuid4().hex

    print("Asistente de la concesionaria. Escribí 'salir' para terminar.\n")
    while True:
        try:
            text = input("Cliente> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in EXIT_WORDS:
            break
        if not text:
            continue
        if len(text) > MAX_INPUT_CHARS:
            print(f"(Mensaje demasiado largo: máximo {MAX_INPUT_CHARS} caracteres.)\n")
            continue
        if args.verbose:
            print("(pensando…)", flush=True)
        started = time.perf_counter()
        try:
            turn = chat_turn(agent, thread_id, text)
        except PROVIDER_ERRORS:
            print(
                "(El servicio de IA no está disponible ahora. Probá de nuevo en un momento.)\n"
            )
            continue
        if args.show_tools:
            for name, call_args in turn.tool_calls:
                print(f"  [tool] {name}({call_args})")
        print(f"Asistente> {turn.reply}\n")
        if args.verbose:
            print(f"  [{time.perf_counter() - started:.1f} s]\n")


if __name__ == "__main__":
    main()
