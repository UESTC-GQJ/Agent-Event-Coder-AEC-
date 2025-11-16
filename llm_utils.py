"""
Utility functions for interacting with language models.

This module centralises calls to external LLM services such as OpenAI's
GPT‑4o.  The default implementation relies on the ``openai``
package.  If no API key is configured or the package is not
installed, a fallback stub is used that raises ``RuntimeError`` to
alert the user.  This design allows the rest of the pipeline to be
written agnostically of the underlying LLM provider while still
supporting offline development.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]


def call_llm(messages: List[Dict[str, str]], model: str = "gpt-4o") -> str:
    """Call an OpenAI chat model with the given messages.

    Parameters
    ----------
    messages : List[Dict[str, str]]
        A list of chat messages following the OpenAI API format.  Each
        message must have a ``"role"`` and a ``"content"`` field.
    model : str, optional
        The name of the model to call.  Defaults to ``"gpt-4o"``.

    Returns
    -------
    str
        The ``content`` of the assistant's reply.

    Raises
    ------
    RuntimeError
        If the ``openai`` package is not installed or the API key is
        missing.
    """
    if openai is None:
        raise RuntimeError(
            "openai package is not installed; please install it to use LLM features."
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set; cannot call OpenAI API."
        )
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message["content"].strip()


def extract_trigger_event_pairs(
    text: str,
    event_definitions: str,
    *,
    model: str = "gpt-4o",
) -> List[Tuple[str, str]]:
    """Use an LLM to extract (trigger, event_type) pairs from ``text``.

    The prompt incorporates event type definitions (e.g. Python class
    representations of events) and asks the model to identify which
    triggers evoke which event types.  The expected output is a JSON
    list of objects with ``"trigger"`` and ``"event_type"`` fields.
    """
    system_prompt = (
        "You are an assistant for event extraction.  Given a piece of text and "
        "definitions of event types (as Python dataclasses), produce a JSON "
        "array of objects where each object has keys 'trigger' and 'event_type'."
    )
    user_prompt = (
        f"Event definitions:\n{event_definitions}\n\n"
        f"Text:\n{text}\n\n"
        "Return a JSON array of {{'trigger': str, 'event_type': str}} objects."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    reply = call_llm(messages, model=model)
    # Try to parse JSON; if parsing fails, return empty list
    try:
        import json  # import inside function to avoid dependency at module import time

        data = json.loads(reply)
        pairs: List[Tuple[str, str]] = []
        if isinstance(data, list):
            for item in data:
                trigger = item.get("trigger")
                evt_type = item.get("event_type")
                if isinstance(trigger, str) and isinstance(evt_type, str):
                    pairs.append((trigger, evt_type))
        return pairs
    except Exception:
        return []


def select_event_definition(
    event_type: str,
    event_definitions: str,
    *,
    model: str = "gpt-4o",
) -> str:
    """Use an LLM to select the definition of a specific event type.

    Given a set of event type definitions (as Python dataclasses) and a
    target ``event_type``, this helper constructs a prompt that asks
    the model to identify which definition corresponds to the given
    event.  The expected output is the complete text of the matching
    definition.  If the model cannot determine a match or an error
    occurs, an empty string is returned.

    Parameters
    ----------
    event_type : str
        The name of the event type to select.
    event_definitions : str
        A string containing multiple dataclass definitions (e.g. as
        returned by :meth:`OntologyManager.build_definitions`).
    model : str, optional
        Name of the model to query.  Defaults to ``"gpt-4o"``.

    Returns
    -------
    str
        The text of the definition corresponding to ``event_type``, or an
        empty string if selection fails.
    """
    system_prompt = (
        "You are a helpful assistant for event ontology selection. "
        "Given a list of event definitions and the name of a target event type, "
        "return exactly the definition (including its class header and fields) "
        "that matches the target event type. If no matching definition is present, return an empty string."
    )
    user_prompt = (
        f"Event definitions:\n{event_definitions}\n\n"
        f"Target event type: {event_type}\n\n"
        "Return the matching definition verbatim."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        reply = call_llm(messages, model=model)
    except Exception:
        return ""
    return reply.strip()