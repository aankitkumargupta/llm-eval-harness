"""
The `Document` value object.

A three-field dataclass, in its own module because of what importing it used to
cost. `Document` lived in `ingest.py`, which imports `qdrant_client` and (via
the Together client) the `openai` SDK. Anything that merely wanted to *read* a
corpus: `profiles/loaders.py`, and therefore the CLI and the Streamlit app,
paid for both.

On a cold Windows install that was ~26 seconds of import time before a single
line of user code ran: 13s for `openai`, 13s for `qdrant_client`, to obtain
three strings. The app looked hung on launch.

Data objects belong apart from the machinery that happens to use them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Document:
    """One source document, before chunking."""

    doc_id: str
    text: str
    source_uri: str = ""
