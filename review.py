"""Step 2: get a gold answer for every term, via a spreadsheet people fill in.

    python review.py export out/candidates.jsonl out/review.csv            # terms without a gold URI
    python review.py import out/candidates.jsonl out/labelled.jsonl out/review.csv [more.csv ...]

In the CSV, fill the ``choice`` column with one of:
* the candidate's number (``3``), key (``aat:300037222``) or full URI;
* ``none`` when no candidate is the thing the record means (this teaches the model to abstain);
* ``skip`` (or leave blank) when the record is too unclear to judge — the term is left out.

Terms that already had a gold URI from the mapping's ``gold_field`` are carried through on import.

When the mapping has an ``entity_field``, every mention of one entity (with the same cleaned term) is one row,
and its answer is given to all of them on import; ``mentions`` says how many. Rows come most mentions first.
The record text shows one record in full and the first line of a few others. If they are clearly different
people or things under one name, answer ``skip`` and re-export that term with ``--per-mention``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict

from common import NONE_KEY, read_jsonl, write_jsonl

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
COLUMNS = ["id", "kind", "term", "field", "mentions", "record_text", "candidates", "choice", "notes"]


def review_key(r: dict, per_mention: bool = False) -> str:
    """Mentions of one entity with the same search term share a candidate list, so they share an answer."""
    if per_mention or not r.get("entity"):
        return r["id"]
    return f"{r['entity']}|{r['kind']}|{r['term']}"


def group_rows(rows: list[dict], per_mention: bool = False) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[review_key(r, per_mention)].append(r)
    return groups


def export(args) -> None:
    rows = read_jsonl(args.candidates)
    todo = [r for r in rows if args.all or not r.get("gold_uri")]
    todo = [r for r in todo if r["candidates"]]  # nothing to choose from: retrieval problem, not a labelling one
    groups = sorted(group_rows(todo, args.per_mention).items(), key=lambda kv: -len(kv[1]))  # biggest payoff first
    with open(args.csv, "w", encoding="utf-8-sig", newline="") as fh:  # BOM so Excel reads UTF-8
        w = csv.DictWriter(fh, COLUMNS)
        w.writeheader()
        for key, members in groups:
            r = members[0]
            listing = "\n".join(f"{i}. {c['key']}  {c['description']}" for i, c in enumerate(r["candidates"], 1))
            text = r["record_text"][: args.text_chars]
            others = [m["record_text"].split("\n", 1)[0] for m in members[1 : 1 + args.other_records]]
            if others:  # first lines (titles) of other records, so a reviewer can spot two people sharing a name
                text += "\n\nAlso in:\n" + "\n".join(others)
            if len(members) > 1 + len(others):
                text += f"\n… and {len(members) - 1 - len(others)} more records"
            w.writerow({"id": key, "kind": r["kind"], "term": r["term"],
                        "field": "; ".join(sorted({m["field"] for m in members})), "mentions": len(members),
                        "record_text": text, "candidates": listing, "choice": "", "notes": ""})
    print(f"{len(groups)} questions covering {len(todo)} terms -> {args.csv} "
          f"({len(rows) - len(todo)} already labelled or without candidates)")


def resolve(choice: str, cands: list[dict]) -> str | None:
    """Turn a reviewer's choice into a candidate key, ``none``, or None (skip/invalid)."""
    c = choice.strip()
    if not c or c.lower() == "skip":
        return None
    if c.lower() == NONE_KEY:
        return NONE_KEY
    if c.isdigit() and 1 <= int(c) <= len(cands):
        return cands[int(c) - 1]["key"]
    for cand in cands:
        if c in (cand["key"], cand["uri"]) or c.rstrip("/") == cand["uri"].rstrip("/"):
            return cand["key"]
    raise ValueError(f"choice {choice!r} is not a candidate number, key, URI, 'none' or 'skip'")


def gold_from_uri(uri: str, cands: list[dict]) -> str | None:
    """A gold URI from the source data -> candidate key; None when retrieval did not find it."""
    if uri.strip().lower() == NONE_KEY:
        return NONE_KEY
    for cand in cands:
        if cand["uri"].rstrip("/") == uri.strip().rstrip("/"):
            return cand["key"]
    return None


def do_import(args) -> None:
    rows = {r["id"]: r for r in read_jsonl(args.candidates)}
    groups = group_rows(list(rows.values()))
    stats: Counter = Counter()
    for r in rows.values():
        r["gold"] = None
        if r.get("gold_uri"):
            r["gold"] = gold_from_uri(r["gold_uri"], r["candidates"])
            stats["from_mapping" if r["gold"] else "mapping_gold_not_retrieved"] += 1
    for path in args.csv:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for line, rec in enumerate(csv.DictReader(fh), 2):
                targets = groups.get(rec["id"]) or ([rows[rec["id"]]] if rec["id"] in rows else [])
                if not targets:
                    stats["unknown_id"] += 1
                    continue
                try:
                    gold = resolve(rec.get("choice", ""), targets[0]["candidates"])
                except ValueError as err:
                    print(f"{path}:{line}: {err}")
                    stats["invalid"] += 1
                    continue
                if gold is None:
                    stats["skipped"] += len(targets)
                    continue
                stats["answers"] += 1
                for r in targets:
                    if gold != NONE_KEY and gold not in {c["key"] for c in r["candidates"]}:
                        stats["group_candidates_differ"] += 1  # built in different runs; not safe to copy
                        continue
                    r["gold"] = gold
                    r["reviewer_notes"] = rec.get("notes", "")
                    stats["from_review"] += 1
    labelled = [r for r in rows.values() if r["gold"]]
    write_jsonl(args.out, labelled)
    nulls = sum(r["gold"] == NONE_KEY for r in labelled)
    print(f"{len(labelled)} labelled terms ({nulls / max(len(labelled), 1):.1%} none) -> {args.out}; {dict(stats)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write a review spreadsheet")
    e.add_argument("candidates")
    e.add_argument("csv")
    e.add_argument("--all", action="store_true", help="include terms that already have a gold URI (to audit them)")
    e.add_argument("--text-chars", type=int, default=600)
    e.add_argument("--per-mention", action="store_true",
                   help="one row per mention even when the mapping gives entity ids (no answer sharing)")
    e.add_argument("--other-records", type=int, default=3,
                   help="for a shared row, how many other records' first lines to show")
    i = sub.add_parser("import", help="merge filled-in spreadsheets into labelled.jsonl")
    i.add_argument("candidates")
    i.add_argument("out")
    i.add_argument("csv", nargs="*", default=[])
    args = ap.parse_args()
    export(args) if args.cmd == "export" else do_import(args)


if __name__ == "__main__":
    main()
