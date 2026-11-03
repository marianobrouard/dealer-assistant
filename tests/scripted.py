"""Test doubles: a chat model that replays a fixed script, with no network."""

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class ScriptedModel(GenericFakeChatModel):
    """Replays the given messages in order and accepts any tools."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self


def tool_call(name: str, args: dict[str, object], call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )
