"""Step 3: labelled terms -> train / val / test JSONL in the format the Yale ``train.py`` reads.

    python make_splits.py out/labelled.jsonl out/splits --val 0.1 --test 0.1
    python make_splits.py out/labelled.jsonl out/splits --by institution --test-groups "Museum A" "Museum B"

Each output row is ``{pid, term, kind, field, text, labels, gold}`` — the same "parsed example" shape their
``recon_data.py`` writes — so ``train.py --train out/splits/train.jsonl --val out/splits/val.jsonl`` runs as is.
Splits are made by record (``pid``) so no record's terms land on both sides; ``--by institution`` holds out
whole collections instead, which is the honest test of how the model copes with a catalogue it has not seen.
``train.py`` carves its own checkpoint-selection slice out of train, so ``val`` stays a clean test set; ``test``
is a further hold-out for choosing an abstention threshold without touching ``val``.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from common import NONE_DESCRIPTION, NONE_KEY, model_text, read_jsonl, write_jsonl


def to_example(r: dict) -> dict:
    labels = {c["key"]: c["description"] for c in r["candidates"] if c["key"] != NONE_KEY}
    labels[NONE_KEY] = NONE_DESCRIPTION
    return {"pid": r["pid"], "institution": r.get("institution", ""), "term": r["term"], "kind": r["kind"],
            "field": r["field"], "text": model_text(r["term"], r["kind"], r["record_text"]), "labels": labels,
            "gold": r["gold"]}


def entity_key(r: dict) -> str:
    """What ``--max-per-entity`` counts: the record's entity id if it has one, else the term itself and its answer."""
    if r.get("entity"):
        return f"{r['entity']}|{r['kind']}|{r['term']}"
    return f"{r['kind']}|{r['term'].lower()}|{r['gold']}"


def bucket(value: str, seed: str) -> float:
    return int(hashlib.sha1(f"{seed}:{value}".encode()).hexdigest(), 16) % 10_000 / 10_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("labelled", help="labelled.jsonl from review.py import")
    ap.add_argument("out_dir")
    ap.add_argument("--by", choices=["pid", "institution"], default="pid")
    ap.add_argument("--val", type=float, default=0.1, help="share of groups for val (hash split)")
    ap.add_argument("--test", type=float, default=0.1, help="share of groups for test (hash split)")
    ap.add_argument("--val-groups", nargs="*", default=[], help="with --by institution: name the val institutions")
    ap.add_argument("--test-groups", nargs="*", default=[], help="with --by institution: name the test institutions")
    ap.add_argument("--min-candidates", type=int, default=1,
                    help="drop terms with fewer real candidates than this (one candidate + none still teaches abstention)")
    ap.add_argument("--max-per-entity", type=int, default=0,
                    help="keep at most this many mentions of one entity per split (0: all). Stops a publisher named on "
                         "every record from flooding training with near-identical examples")
    ap.add_argument("--seed", default="42")
    args = ap.parse_args()

    splits: dict[str, list[dict]] = defaultdict(list)
    entity_of: dict[int, str] = {}
    dropped: Counter = Counter()
    for r in read_jsonl(args.labelled):
        ex = to_example(r)
        if len(ex["labels"]) - 1 < args.min_candidates:
            dropped["too_few_candidates"] += 1
            continue
        if ex["gold"] not in ex["labels"]:
            dropped["gold_not_in_candidates"] += 1  # classification needs the answer on the list
            continue
        group = str(ex["institution"] if args.by == "institution" else ex["pid"])
        if group in args.test_groups:
            name = "test"
        elif group in args.val_groups:
            name = "val"
        elif args.val_groups or args.test_groups:
            name = "train"
        else:
            b = bucket(group, args.seed)
            name = "test" if b < args.test else "val" if b < args.test + args.val else "train"
        splits[name].append(ex)
        entity_of[id(ex)] = entity_key(r)

    if args.max_per_entity:
        for name, rows in splits.items():
            by_entity: dict[str, list[dict]] = defaultdict(list)
            for ex in rows:
                by_entity[entity_of[id(ex)]].append(ex)
            keep = set()
            for mentions in by_entity.values():  # a stable, seeded choice of which records keep the mention
                mentions.sort(key=lambda e: bucket(f"{e['pid']}|{e['field']}", args.seed))
                keep.update(id(e) for e in mentions[: args.max_per_entity])
            dropped["over_max_per_entity"] += len(rows) - len(keep)
            splits[name] = [ex for ex in rows if id(ex) in keep]

    out = Path(args.out_dir)
    for name in ("train", "val", "test"):
        rows = splits.get(name, [])
        write_jsonl(out / f"{name}.jsonl", rows)
        nulls = sum(e["gold"] == NONE_KEY for e in rows)
        kinds = dict(Counter(e["kind"] for e in rows))
        avg = sum(len(e["labels"]) for e in rows) / max(len(rows), 1)
        print(f"{name:5} {len(rows):6} examples, {len({e['pid'] for e in rows})} records, "
              f"{nulls / max(len(rows), 1):.1%} none, {avg:.1f} labels avg, {kinds}")
    if dropped:
        print(f"dropped: {dict(dropped)}")
    train_pids = {(e["institution"], e["pid"]) for e in splits.get("train", [])}
    leaks = train_pids & {(e["institution"], e["pid"]) for n in ("val", "test") for e in splits.get(n, [])}
    if leaks:
        raise SystemExit(f"{len(leaks)} records appear in train and val/test: check --by and the group lists")


if __name__ == "__main__":
    main()
