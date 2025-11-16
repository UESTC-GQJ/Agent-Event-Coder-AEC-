"""
Coding agent for the AEC pipeline.

The coding agent takes the highest‑ranked trigger hypothesis produced
by the planning agent and constructs executable Python code that
instantiates an event object conforming to the given schema.  In a
complete AEC system this step uses natural language understanding to
extract argument values from the input text and populate the event
object accordingly.  Here we implement a very conservative baseline
that leaves all argument lists empty and simply records the trigger.

Two methods are provided: ``generate_code`` returns a string of
Python code that, when executed, yields an instance of
``EventObject``; ``generate_event_object`` performs the same logic
directly and returns an ``EventObject`` instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .event_schema import EventSchema, EventObject
from .planning_agent import Hypothesis


@dataclass
class CodingAgent:
    """A simplistic coding agent that produces event objects and code."""

    def generate_event_object(
        self, hypothesis: Hypothesis, schema: EventSchema, text: str
    ) -> EventObject:
        """Instantiate an :class:`EventObject` from a trigger hypothesis.

        This implementation does not attempt to extract argument
        values.  Instead, it initialises every role in the schema to an
        empty list.  Subclasses can override this method to implement
        proper argument extraction.
        """
        args: Dict[str, List[str]] = {role: [] for role in schema.roles}
        return EventObject(event_type=schema.event_type, trigger=hypothesis.trigger, arguments=args)

    def generate_code(
        self, hypothesis: Hypothesis, schema: EventSchema, text: str
    ) -> str:
        """Return a Python code snippet that instantiates the event object.

        The returned code imports ``EventObject`` from ``aec.event_schema``
        and constructs an instance using the trigger and empty
        arguments.  The code ends with a reference to the created
        object so that its value is the result of executing the
        snippet in an interactive session.
        """
        # Build the arguments dictionary literal
        arg_dict_items = ", ".join([f"'{role}': []" for role in schema.roles])
        code = (
            "from aec.event_schema import EventObject\n"
            f"event = EventObject(event_type='{schema.event_type}', trigger='{hypothesis.trigger}', arguments={{ {arg_dict_items} }})\n"
            "event"
        )
        return code