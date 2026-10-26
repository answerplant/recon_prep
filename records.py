"""Read your collection records and turn them into terms to reconcile, driven by a mapping file.

A mapping file (JSON) says which fields hold the record id, the context text and the terms:

    {
      "id_field": "object_number",
      "institution_field": "institution",
      "text_fields": {"title": "Title", "object_name": "Object name", "description": "Description"},
      "max_text_chars": 1500,
      "terms": [
        {"field": "object_name", "kind": "concept", "split": ";", "gold_field": "object_name_aat"},
        {"field": "maker",       "kind": "person",  "split": ";"},
        {"field": "place_made",  "kind": "place"}
      ]
    }

* ``text_fields`` maps a field to the caption used in the record text the model reads ("Title: ...").
* ``split`` breaks a multi-valued field into several terms; JSON lists are split automatically.
* ``gold_field`` (optional) holds authority URIs you already have, aligned with ``field`` after splitting.
  Use ``none`` for a value you know has no match. These become gold labels without manual review.
* Dotted field names (``production.maker.name``) reach into nested JSON. Lists are walked and flattened,
  so ``produced_by.part.carried_out_by._label`` collects every maker of every part.
* A ``[path=value]`` (or ``[path!=value]``) filter after a name keeps only the list items whose ``path``
  has that value: ``referred_to_by[classified_as.id=http://vocab.getty.edu/aat/300435416].content``
  is the description and nothing else.
* ``kind_field`` (instead of, or as well as, ``kind``) takes each value's kind from the record, aligned like
  ``gold_field``: ``{"field": "about._label", "kind_field": "about.type"}``. Linked Art types are mapped by
  ``LINKED_ART_KINDS``; ``kind_map`` adds to or overrides that; ``kind`` is the fallback for unmapped types.
* ``entity_field`` (optional, aligned like ``gold_field``) names the entity each value stands for, e.g.
  ``produced_by.carried_out_by.id``. ``review.py`` then asks once per entity, not once per mention.
* ``record_where`` (optional, same syntax as a filter) skips records that do not match, e.g.
  ``"type=HumanMadeObject"`` to ignore entity records in a Linked Art bundle folder.

CSV (UTF-8, header row) and JSONL inputs are supported, gzipped JSONL too. A folder is searched (recursively)
for ``.jsonl`` / ``.ndjson`` files, gzipped or not, such as the bundles nrich_la writes.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from common import KINDS, iter_jsonl

LINKED_ART_KINDS = {"Person": "person", "Group": "group", "Place": "place", "Type": "concept",
                    "Material": "concept", "Language": "concept", "MeasurementUnit": "unit"}
JSONL_SUFFIXES = (".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")

_SEGMENT = re.compile(r"([^.\[]+)(?:\[([^\]]*)\])?(?:\.|$)")


@dataclass
class Term:
    pid: str
    institution: str
    field: str
    kind: str
    term: str
    record_text: str
    gold_uri: str | None = None  # a URI, "none", or None for unlabelled
    term_raw: str | None = None  # the value as the record had it, when the cleaning step changed it
    entity: str | None = None  # the record's own id for the thing the value names, from ``entity_field``

    def to_dict(self) -> dict:
        return asdict(self)


def load_mapping(path: str | Path) -> dict:
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    for spec in mapping["terms"]:
        if not spec.get("kind") and not spec.get("kind_field"):
            raise ValueError(f"term field {spec['field']!r}: give a kind, a kind_field, or both")
        for kind in [spec.get("kind")] + list(spec.get("kind_map", {}).values()):
            if kind and kind not in KINDS:
                raise ValueError(f"term field {spec['field']!r}: kind must be one of {KINDS}, got {kind!r}")
    if isinstance(mapping.get("text_fields"), list):
        mapping["text_fields"] = {f: f for f in mapping["text_fields"]}
    return mapping


def iter_records(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.name.lower().endswith(JSONL_SUFFIXES))
        if not files:
            raise ValueError(f"no .jsonl / .ndjson files (gzipped or not) under {path}")
        for p in files:
            yield from iter_jsonl(p)
    elif path.name.lower().endswith(JSONL_SUFFIXES):
        yield from iter_jsonl(path)
    elif path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as fh:
            yield from csv.DictReader(fh)
    else:
        raise ValueError(f"unsupported record file {path.name}: use .csv, .jsonl (optionally .gz) or a folder")


def _segments(dotted: str) -> list[tuple[str, str | None]]:
    return [(m.group(1), m.group(2)) for m in _SEGMENT.finditer(dotted)]


def matches(item: Any, condition: str) -> bool:
    """``path=value`` / ``path!=value``: does ``path`` inside ``item`` have (or lack) that value?"""
    negate = "!=" in condition
    path, _, expected = condition.partition("!=" if negate else "=")
    found = expected.strip() in as_values(get_field(item, path.strip()), None) if isinstance(item, dict) else False
    return found != negate


def _step(value: Any, name: str, condition: str | None) -> Any:
    if not isinstance(value, dict):
        return None
    got = value.get(name)
    if condition:
        items = got if isinstance(got, list) else [] if got is None else [got]
        got = [item for item in items if matches(item, condition)]
    return got


def get_field(record: dict, dotted: str) -> Any:
    value: Any = record
    for name, condition in _segments(dotted):
        if isinstance(value, list):
            out = []
            for v in value:
                got = _step(v, name, condition)
                out.extend(got if isinstance(got, list) else [got])  # flatten, keeping None as a placeholder
            value = out
        else:
            value = _step(value, name, condition)
    return value


def as_values(value: Any, split: str | None) -> list[str]:
    """Normalise a field value to a list of non-empty strings (keeps positions for gold alignment)."""
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(as_values(v, split) or [""])
        return out
    text = str(value)
    parts = text.split(split) if split else [text]
    return [" ".join(p.split()) for p in parts]


def record_text(record: dict, mapping: dict) -> str:
    lines = []
    for field, caption in mapping.get("text_fields", {}).items():
        values = [v for v in as_values(get_field(record, field), None) if v]
        if values:
            lines.append(f"{caption}: {'; '.join(values)}")
    text = "\n".join(lines)
    limit = mapping.get("max_text_chars")
    if limit and len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text


def _aligned(record: dict, spec: dict, key: str) -> list[str]:
    return as_values(get_field(record, spec[key]), spec.get("split")) if spec.get(key) else []


def iter_terms(path: str | Path, mapping: dict) -> Iterator[Term]:
    """One ``Term`` per (record, term field, value), skipping blanks and duplicates within a record."""
    where = mapping.get("record_where")
    for record in iter_records(path):
        if where and not matches(record, where):
            continue
        pid = str(get_field(record, mapping["id_field"]) or "").strip()
        if not pid:
            continue
        inst_field = mapping.get("institution_field")
        institution = "; ".join(v for v in as_values(get_field(record, inst_field), None) if v) if inst_field else ""
        text = record_text(record, mapping)
        seen = set()
        for spec in mapping["terms"]:
            values = as_values(get_field(record, spec["field"]), spec.get("split"))
            golds = _aligned(record, spec, "gold_field")
            types = _aligned(record, spec, "kind_field")
            entities = _aligned(record, spec, "entity_field")
            kind_map = LINKED_ART_KINDS | spec.get("kind_map", {})
            for i, value in enumerate(values):
                kind = kind_map.get(types[i]) if i < len(types) else None
                kind = kind or spec.get("kind")
                if not value or not kind or (kind, value.lower()) in seen:
                    continue
                seen.add((kind, value.lower()))
                gold = golds[i] if i < len(golds) and golds[i] else None
                entity = entities[i] if i < len(entities) and entities[i] else None
                yield Term(pid, institution, spec["field"], kind, value, text, gold, entity=entity)
