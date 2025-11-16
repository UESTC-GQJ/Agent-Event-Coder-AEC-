"""
End‑to‑end AEC pipeline.

This module exposes the :class:`AECPipeline` class, which ties
together the retrieval, planning, coding and verification agents into
a coherent workflow.  Given an input text and an event schema the
pipeline uses a dual‑loop refinement algorithm inspired by the
procedure described in the AEC paper (Algorithm 1).  The outer loop
iterates over trigger hypotheses ranked by confidence; the inner loop
attempts to generate and verify a candidate event object multiple
times, allowing for patches in response to verification errors.

Because this implementation does not currently support automatic
patching, each failed attempt simply retries the same hypothesis up
to the specified limit.  Users interested in refining event objects
should subclass :class:`CodingAgent` and override its
``generate_event_object`` method to adjust triggers and arguments
based on verification feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .event_schema import EventSchema, EventObject
from .retrieval_agent import RetrievalAgent
from .planning_agent import PlanningAgent, Hypothesis
from .coding_agent import CodingAgent
from .verification_agent import VerificationAgent, VerificationError
from .ontology import OntologyManager  # for type hints


@dataclass
class AECPipeline:
    """High‑level orchestrator for the AEC multi‑agent pipeline.

    Besides the four agents, the pipeline can optionally be provided with an
    :class:`OntologyManager` to look up event schemas for a given dataset.
    If a schema is not explicitly passed to :meth:`run`, the
    ``dataset`` argument will be used with the ontology manager to
    select the appropriate schema.
    """

    retrieval_agent: RetrievalAgent = field(default_factory=RetrievalAgent)
    planning_agent: PlanningAgent = field(default_factory=PlanningAgent)
    coding_agent: CodingAgent = field(default_factory=CodingAgent)
    verification_agent: VerificationAgent = field(default_factory=VerificationAgent)
    ontology_manager: Optional["OntologyManager"] = None  # type: ignore[name-defined]
    max_hypotheses: int = 3
    max_patches: int = 2
    # When set to True the pipeline uses an LLM to generate trigger
    # hypotheses (via PlanningAgent.generate_hypotheses_with_llm) and to
    # select event definitions prior to coding.  If False, the
    # heuristic planner is used.  Note that enabling this option
    # requires the OpenAI API to be available and configured via
    # environment variables.
    use_llm_plan: bool = False

    def run(
        self,
        text: str,
        schema: Optional[EventSchema] = None,
        *,
        dataset: Optional[str] = None,
        event_type: Optional[str] = None,
        use_llm_plan: Optional[bool] = None,
    ) -> Optional[EventObject]:
        """Execute the pipeline on ``text``.

        Parameters
        ----------
        text : str
            The input sentence or paragraph.
        schema : EventSchema, optional
            Explicitly provide an event schema.  If ``None``, the
            ``dataset`` and ``event_type`` arguments will be used with
            the ontology manager to look up the schema.
        dataset : str, optional
            Name of the dataset whose ontology should be used.  Must be
            provided if ``schema`` is not.
        event_type : str, optional
            Name of the event type to extract.  Must be provided if
            ``schema`` is not.

        Returns
        -------
        Optional[EventObject]
            A validated event object if the pipeline succeeds, otherwise ``None``.
        """
        # Resolve schema from ontology if not explicitly provided
        if schema is None:
            if self.ontology_manager is None:
                raise ValueError(
                    "Ontology manager is not set; please provide a schema or configure an ontology manager."
                )
            if not dataset or not event_type:
                raise ValueError("dataset and event_type must be provided when schema is None")
            schema = self.ontology_manager.get_schema(dataset, event_type)
            if schema is None:
                raise ValueError(f"No schema found for event type '{event_type}' in dataset '{dataset}'")
        # Step 1: retrieve exemplars (unused by the heuristic planner but included for completeness)
        examples = self.retrieval_agent.retrieve(schema, k=self.max_hypotheses)
        # Step 2: generate hypotheses.  If LLM planning is enabled we
        # build the event definitions from the ontology and use the LLM
        # to extract (trigger, event_type) pairs.  Otherwise we fall
        # back to the heuristic planner.  If use_llm_plan is passed to
        # this method it overrides the instance setting on the
        # pipeline.
        use_llm = self.use_llm_plan if use_llm_plan is None else use_llm_plan
        hypotheses: List[Hypothesis]
        if use_llm:
            if self.ontology_manager is None:
                raise ValueError(
                    "use_llm_plan=True requires an ontology manager to build event definitions."
                )
            if not dataset:
                raise ValueError(
                    "use_llm_plan=True requires a dataset argument to build event definitions."
                )
            # Build definitions for the entire dataset and request LLM to extract
            event_defs = self.ontology_manager.build_definitions(dataset)
            # Use LLM-based planning to produce hypotheses
            # We request more candidates than needed to allow filtering by event_type
            llm_hyps = self.planning_agent.generate_hypotheses_with_llm(
                text=text,
                event_definitions=event_defs,
                max_candidates=self.max_hypotheses * 2,
            )
            # Filter hypotheses by event_type if specified
            if event_type:
                llm_hyps = [h for h in llm_hyps if h.event_type.lower() == event_type.lower()]
            # Truncate to max_hypotheses
            hypotheses = llm_hyps[: self.max_hypotheses]
        else:
            # Fallback heuristic planning
            hypotheses = self.planning_agent.generate_hypotheses(
                text=text,
                schema=schema,
                examples=examples,
                k=self.max_hypotheses,
            )
        # Outer loop over hypotheses sorted by confidence
        for hyp_index, hypothesis in enumerate(hypotheses, start=1):
            # If LLM planning is enabled, optionally select the event
            # definition corresponding to this hypothesis using the
            # event ontology.  This step mirrors the coding agent
            # reading the correct definition.  We call the
            # select_event_definition helper to issue the prompt.  The
            # return value is not used further in this baseline but
            # demonstrates the expected interaction pattern.
            if use_llm:
                try:
                    from .llm_utils import select_event_definition  # local import
                except Exception:
                    select_event_definition = None  # type: ignore
                if select_event_definition is not None and dataset:
                    event_defs = self.ontology_manager.build_definitions(dataset)
                    _ = select_event_definition(
                        event_type=hypothesis.event_type,
                        event_definitions=event_defs,
                    )
            # Inner loop: allow multiple attempts to patch the event
            for attempt in range(1, self.max_patches + 1):
                # Step 3: generate candidate event object
                event_obj = self.coding_agent.generate_event_object(
                    hypothesis=hypothesis, schema=schema, text=text
                )
                try:
                    # Step 4: verify the event
                    self.verification_agent.verify(event_obj, schema, text)
                    # Success: return event object
                    return event_obj
                except VerificationError:
                    # In a full implementation we would inspect the error and attempt
                    # to patch the event (e.g. adjust trigger or fill missing arguments).
                    pass
            # If we reach here the current hypothesis failed max_patches attempts
            continue
        # All hypotheses exhausted
        return None