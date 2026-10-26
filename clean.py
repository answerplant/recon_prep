"""Optional pipeline step between reading terms and retrieving candidates: tidy catalogue headings into search terms.

Library records carry cataloguing punctuation and qualifiers that break name search: ``Scott, Walter,`` would
be searched as ``Walter, Scott`` and ``Fowler, John, Sir, 1817-1898`` as ``John, Sir, 1817-1898 Fowler``.
This step fixes the term only; the record text the model reads is left as the record had it.

    python build_candidates.py records mapping.json out/candidates.jsonl            # cleaning on (default)
    python build_candidates.py records mapping.json out/candidates.jsonl --no-clean # terms exactly as recorded
    python clean.py records mapping.json            # preview: what the step would change, before retrieval
    python clean.py --term person "Baker, Benjamin, Sir, 1840-1907"

What it does (``clean_term``), in order:
* every kind: compose accents (Unicode NFC). Decomposed ``Bruère`` is otherwise searched as ``brue re``;
* every kind: drop ``[...]`` role notes (``[distributor]``) and the brackets around supplied text, turn
  placeholders (``s.n.``, ``publisher not identified``) into nothing, so the term is dropped;
* place: a broad-to-narrow hierarchy (``Europe, United Kingdom, Scotland, Aberdeen, Aberdeen (inhabited place)``)
  becomes its narrowest named place, skipping street numbers and keeping TGN's inverted names whole
  (``…, Perth and Kinross, Tay, Loch (lake)`` -> ``Tay, Loch (lake)``);
* group: drop an imprint's lead-in (``printed by``), and keep the first name of a list (up to ``;`` or an address);
* person, group, concept: drop trailing life dates (``1771-1832``, ``-1827``, ``fl. 1760-1788``, ``active 1813``);
* person: keep ``Surname, Forenames`` and drop titles and epithets after it (``Sir``, ``Duke of``, ``of Fearn``);
* every kind: drop trailing punctuation (``,`` ``;`` ``:`` ``/`` and a ``.`` that does not end an initial or
  an abbreviation such as ``Co.``).

Changing these rules changes what the model is trained on: run the same ``clean_term`` (or none) in production.
Each row from ``build_candidates.py`` keeps the original in ``term_raw`` when the step changed it.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Iterable, Iterator

from records import Term, iter_terms, load_mapping

_ROLE_NOTE = re.compile(r"\[(?:distributor|publisher|printer|editor|author|compiler)\]", re.I)
_PLACEHOLDER = re.compile(r"^(?:s\.\s?[nl]\.?|n\.\s?d\.?|sine nomine|sine loco|unknown|anon\.?|"
                          r"(?:publisher|place of publication|printer|manufacturer)s? not identified)$", re.I)
_CONTINENTS = {"world", "europe", "asia", "africa", "north and central america", "north america", "south america",
               "oceania", "antarctica"}
_INVERTED = {"Loch", "Lake", "River", "Glen", "Ben", "Mount", "Bay", "Sound", "Strait", "Firth", "Kyle", "Mull", "Isle"}
_IMPRINT_LEAD =re.compile(r"^(?:(?:re-?)?printed|published|sold|engraved|lithographed|printed and published)"
                           r"(?: and (?:sold|published))?(?: at [^,]+?)? (?:by|for)\s+", re.I)
_ADDRESS = re.compile(r",\s*(?:no\.\s*)?\d")
_DATES = re.compile(r"(?:,\s*|\s+)(?:(?:b|d|fl|ca?|active|born|died)\.?\s*)?\d{0,4}\??\s*-\s*(?:(?:ca?)\.?\s*)?\d{0,4}\??"
                    r"(?:\s+or\s+\d+)?$|(?:,\s*|\s+)(?:(?:b|d|fl|ca?|active|born|died)\.?\s*)?\d{3,4}\??$", re.I)
_TRAILING = re.compile(r"[\s,;:/=?]+$")
_KEEP_DOT = re.compile(r"(?:\b[A-Za-z]|\b(?:co|ltd|inc|jr|sr|bros|st|esq|plc))\.$", re.I)


def _strip_trailing(text: str) -> str:
    while True:
        text = _TRAILING.sub("", text)
        if text.endswith(".") and not _KEEP_DOT.search(text):
            text = text[:-1]
            continue
        return text


def _narrowest_place(text: str) -> str:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 3 or parts[0].lower() not in _CONTINENTS:
        return text
    named = [p for p in parts if not p[:1].isdigit()]  # "Pleasance, 117-121 (street address)" -> "Pleasance"
    head = named[-1].split(" (")[0]
    if len(named) > 1 and (head in _INVERTED or head.endswith(" of")):  # TGN's "Tay, Loch (lake)" is one name
        return f"{named[-2]}, {named[-1]}"
    return named[-1]


def clean_term(term: str, kind: str) -> str:
    """The search form of a catalogue heading; empty when nothing searchable is left."""
    text = _ROLE_NOTE.sub(" ", unicodedata.normalize("NFC", term)).replace("[", " ").replace("]", " ")
    text = _strip_trailing(" ".join(text.split()))
    if kind == "place":
        text = _narrowest_place(text)
    if kind == "group":
        text = _IMPRINT_LEAD.sub("", text)
        text = _ADDRESS.split(text.split(";")[0])[0]
    if kind in ("person", "group", "concept"):
        text = _DATES.sub("", _strip_trailing(text))
    if kind == "person":
        parts = [p.strip() for p in text.split(",")]
        if len(parts) > 2:
            text = ", ".join(parts[:2])
    text = _strip_trailing(text.strip())
    return "" if _PLACEHOLDER.match(text) else text


def clean_terms(terms: Iterable[Term], stats: Counter | None = None) -> Iterator[Term]:
    """The pipeline step: clean each term, drop the empty ones, and merge terms a record now has twice."""
    stats = stats if stats is not None else Counter()
    seen: set[tuple[str, str, str]] = set()
    for t in terms:
        cleaned = clean_term(t.term, t.kind)
        if not cleaned:
            stats["clean_dropped"] += 1
            continue
        key = (t.pid, t.kind, cleaned.lower())
        if key in seen:  # "Scott, Walter," and "Scott, Walter, 1771-1832." in one record
            stats["clean_merged"] += 1
            continue
        seen.add(key)
        if cleaned == t.term:
            yield t
        else:
            stats["clean_changed"] += 1
            yield replace(t, term=cleaned, term_raw=t.term)


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview the cleaning step on your records, or clean one term.")
    ap.add_argument("records", nargs="?", help="records file or folder, as for build_candidates.py")
    ap.add_argument("mapping", nargs="?", help="mapping JSON (see records.py)")
    ap.add_argument("--term", nargs=2, metavar=("KIND", "TEXT"), help="clean a single term and print it")
    ap.add_argument("--examples", type=int, default=10, help="changed terms shown per kind")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many terms")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # headings carry accents a Windows console code page lacks
    if args.term:
        print(clean_term(args.term[1], args.term[0]) or "(dropped)")
        return
    if not (args.records and args.mapping):
        ap.error("give records and mapping, or --term KIND TEXT")

    stats: Counter = Counter()
    raw: Counter = Counter()
    changes: dict[str, Counter] = defaultdict(Counter)
    def counted(terms: Iterable[Term]) -> Iterator[Term]:
        for i, t in enumerate(terms):
            if args.limit and i >= args.limit:
                return
            raw[t.kind] += 1
            yield t
    for t in clean_terms(counted(iter_terms(args.records, load_mapping(args.mapping))), stats):
        if t.term_raw is not None:
            changes[t.kind][(t.term_raw, t.term)] += 1
    print(f"{sum(raw.values())} terms: {stats['clean_changed']} changed, {stats['clean_dropped']} dropped, "
          f"{stats['clean_merged']} merged into another term of the same record")
    for kind in sorted(raw):
        print(f"\n{kind}: {raw[kind]} terms, {sum(changes[kind].values())} changed")
        for (before, after), n in changes[kind].most_common(args.examples):
            print(f"  {n:6}  {before!r} -> {after!r}")


if __name__ == "__main__":
    main()
