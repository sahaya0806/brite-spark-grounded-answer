"""
Chat provider interface and implementations for grounded answer generation.

Architecture
------------
``ChatProvider`` is a Protocol that any chat backend must implement:
- Production code uses ``OpenAIChatProvider`` backed by GPT-4o mini.
- Tests use ``FakeChatProvider`` for deterministic, offline testing without
  API keys or network calls.

Configuration
-------------
The OpenAI provider reads from environment variables:
    OPENAI_API_KEY      — required for production generation
    OPENAI_CHAT_MODEL   — optional, defaults to "gpt-4o-mini"
"""

from __future__ import annotations

import os
import re
from typing import Callable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ChatProvider(Protocol):
    """
    Minimal interface for a chat completion backend.
    """

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a text response given a list of message dicts.

        Parameters
        ----------
        messages:
            List of message dicts, e.g. [{"role": "system", "content": "..."}, ...]
        temperature:
            Sampling temperature. Defaults to 0.0 for deterministic output.

        Returns
        -------
        str
            The generated text content.
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAIChatProvider:
    """
    Chat provider backed by the OpenAI Chat Completions API (GPT-4o mini).

    Parameters
    ----------
    model:
        Model name. Defaults to the OPENAI_CHAT_MODEL env var, or "gpt-4o-mini".
    api_key:
        OpenAI API key. Defaults to the OPENAI_API_KEY env var.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        import openai  # noqa: F401 — ensure installed

        self._model = (
            model
            or os.environ.get("OPENAI_CHAT_MODEL")
            or self.DEFAULT_MODEL
        )
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set the OPENAI_API_KEY environment variable or pass api_key=."
            )

    def _client(self):
        import openai
        return openai.OpenAI(api_key=self._api_key)

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        """Send chat messages to OpenAI and return response string."""
        if not messages:
            raise ValueError("messages must not be empty")

        response = self._client().chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content or ""


# ---------------------------------------------------------------------------
# Deterministic fake for testing
# ---------------------------------------------------------------------------

class FakeChatProvider:
    """
    Deterministic chat provider for testing without network or API keys.

    Parameters
    ----------
    canned_response:
        Optional fixed response string to return for all calls.
    responder:
        Optional custom function ``(messages) -> str`` to compute the response.
    """

    def __init__(
        self,
        canned_response: str | None = None,
        responder: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> None:
        self.canned_response = canned_response
        self.responder = responder
        self.call_history: list[list[dict[str, str]]] = []

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        self.call_history.append(messages)

        if self.canned_response is not None:
            return self.canned_response

        if self.responder is not None:
            return self.responder(messages)

        # Default synthetic response builder based on input messages
        user_msg = ""
        for m in messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "")

        # Extract any clause citations mentioned in the prompt
        clause_matches = re.findall(r'\[§([\d\.]+)\]', user_msg)
        if clause_matches:
            cited = f"[§{clause_matches[0]}]"
            return (
                f"According to policy, the requirement is established in {cited}. "
                f"Applicants and recipients must comply with the specified provisions."
            )

        return "According to the supplied policy manual, the conditions are satisfied."
