"""Shared constants and helpers.

``NONE_KEY``, ``NONE_DESCRIPTION``, ``MAX_DESCRIPTION_CHARS``, ``sanitize`` and ``uri_to_key`` are copied
from the Yale model's ``training/recon_data.py`` so that examples built here look exactly like theirs.
Keep them identical to whatever your inference code sends the model.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

NONE_KEY = "none"
NONE_DESCRIPTION = "None of the listed records is the entity referred to in the text"
TASK = "entity"
MAX_DESCRIPTION_CHARS = 200
KINDS = ("concept", "person", "group", "place", "unit")

_MARKERS = re.compile(r"\[(?:P|L|C|E|R|DESCRIPTION|EXAMPLE|OUTPUT)\]")


def sanitize(value: str) -> str:
    """gliner2 reserves its marker tokens and bare parentheses in labels."""
    value = _MARKERS.sub(" ", str(value))
    value = value.replace("(", "[").replace(")", "]")
    return " ".join(value.split()) or "-"


def uri_to_key(uri: str, source: str) -> str:
    """``http://vocab.getty.edu/aat/300037222`` -> ``aat:300037222``; Wikidata -> ``wikidata:Q41207``."""
    return f"{source}:{uri.rstrip('/').rsplit('/', 1)[-1]}"


def summarise(label: str, desc: str | None) -> str:
    """``label — description``, description cut to ``MAX_DESCRIPTION_CHARS`` on a word boundary."""
    desc = " ".join((desc or "").split())
    if len(desc) > MAX_DESCRIPTION_CHARS:
        desc = desc[:MAX_DESCRIPTION_CHARS].rsplit(" ", 1)[0] + "…"
    return sanitize(f"{label} — {desc}" if desc else label)


def model_text(term: str, kind: str, record_text: str) -> str:
    """The input layout GLiNER2.5-Decide sees at inference (``question_format: criteria``)."""
    return f"Mention: {term}\nEntity type: {kind}\nText: {record_text}"


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """One dict per line; ``.gz`` files are decompressed on the fly."""
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def chunked(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
