"""
Ontology management for event schemas.

This module provides utilities to load event schemas from dataset
ontology files and to select the appropriate schema for a given
dataset and event type.  Ontology files are expected to be simple
Python scripts containing a series of ``@dataclass`` class
definitions, where the class name corresponds to the event type and
the fields correspond to the argument roles.  The CASIE example
provided by the user follows this format.

The :class:`OntologyManager` can load multiple ontology files at
initialisation and expose a dictionary mapping dataset names to
``EventSchema`` objects keyed by event type.  When asked for a
schema it performs a case‑insensitive lookup on the event type
within the specified dataset.
"""

from __future__ import annotations

import ast
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

from .event_schema import EventSchema


@dataclass
class OntologyManager:
    """Load and store event schemas for one or more datasets."""

    dataset_files: Mapping[str, str]
    _schemas: Dict[str, Dict[str, EventSchema]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._schemas = {}
        for dataset, file_path in self.dataset_files.items():
            self._schemas[dataset] = self._load_schemas_from_file(file_path)

    def _load_schemas_from_file(self, file_path: str) -> Dict[str, EventSchema]:
        """Parse a Python or plain text file containing dataclass event definitions.

        The ontology files may not be valid Python because some field names
        contain hyphens (e.g. ``number-of-data``), which are illegal in
        Python identifiers.  We first attempt to use Python's AST to
        parse the file.  If that fails with a ``SyntaxError`` we fall
        back to a simple line‑based parser that extracts class names
        and role names heuristically.
        """
        path = pathlib.Path(file_path)
        text = path.read_text(encoding="utf-8")
        schemas: Dict[str, EventSchema] = {}
        # Try parsing via AST for well‑formed Python files
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    if any(
                        isinstance(decorator, ast.Name) and decorator.id == "dataclass"
                        or (isinstance(decorator, ast.Attribute) and decorator.attr == "dataclass")
                        for decorator in node.decorator_list
                    ):
                        event_type = node.name
                        roles: Dict[str, type] = {}
                        for stmt in node.body:
                            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                                role_name = stmt.target.id
                                # Default to str for simplicity
                                roles[role_name] = str
                        schemas[event_type.lower()] = EventSchema(event_type, roles)
            return schemas
        # Fallback manual parsing for non‑Python files (e.g. hyphenated field names)
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("@dataclass"):
                # Expect a class definition on the next non‑empty line
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                if j >= len(lines):
                    break
                class_line = lines[j].strip()
                # Extract class name up to the opening parenthesis or colon
                match = re.match(r"class\s+([A-Za-z0-9_]+)", class_line)
                if match:
                    event_type = match.group(1)
                    roles: Dict[str, type] = {}
                    # Read subsequent indented lines as field definitions
                    k = j + 1
                    while k < len(lines):
                        raw = lines[k]
                        # stop if indentation ends or a new decorator/class starts
                        if raw.lstrip() == raw or raw.strip().startswith("@dataclass"):
                            break
                        field_line = raw.strip()
                        # Look for "name:" pattern
                        if ":" in field_line:
                            role_name = field_line.split(":", 1)[0].strip()
                            # skip the mention field; the trigger is handled separately
                            if role_name != "mention":
                                roles[role_name] = str
                        k += 1
                    schemas[event_type.lower()] = EventSchema(event_type, roles)
                    i = k
                    continue
            i += 1
        return schemas

    def get_schema(self, dataset: str, event_type: str) -> Optional[EventSchema]:
        """Return the schema for a given dataset and event type, if present."""
        ds_schemas = self._schemas.get(dataset)
        if not ds_schemas:
            return None
        return ds_schemas.get(event_type.lower())

    def build_definitions(self, dataset: str) -> str:
        """Construct Python dataclass definitions for all event types in a dataset.

        The returned string can be embedded in a prompt to an LLM.  Each
        definition includes the ``@dataclass`` decorator, the class
        name, and one annotated attribute per role (using ``List`` without
        specifying the element type to reduce verbosity).  If the
        requested dataset is unknown, an empty string is returned.
        """
        ds_schemas = self._schemas.get(dataset)
        if not ds_schemas:
            return ""
        lines: List[str] = []
        lines.append("from dataclasses import dataclass")
        lines.append("from typing import List")
        lines.append("")
        for schema in ds_schemas.values():
            lines.append("@dataclass")
            lines.append(f"class {schema.event_type}:")
            lines.append("    mention: str")
            for role in schema.roles.keys():
                lines.append(f"    {role}: List")
            lines.append("")
        return "\n".join(lines)

    @classmethod
    def from_directory(cls, dir_path: str) -> "OntologyManager":
        """Construct an ``OntologyManager`` by scanning a directory for ontology files.

        This convenience method looks for Python files (``*.py``) in the
        given directory and treats each file as an ontology for a
        dataset.  The dataset name is derived from the filename stem
        (converted to lowercase).  For example, a file named
        ``casie.py`` would be loaded under the ``casie`` dataset key.

        Parameters
        ----------
        dir_path : str
            Path to a directory containing ontology definition files.

        Returns
        -------
        OntologyManager
            An instance loaded with schemas for all recognised datasets.
        """
        base = pathlib.Path(dir_path)
        dataset_files: Dict[str, str] = {}
        for file in base.glob("*.py"):
            dataset_name = file.stem.lower()
            dataset_files[dataset_name] = str(file)
        return cls(dataset_files=dataset_files)