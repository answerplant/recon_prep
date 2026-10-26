"""Step 1: records -> terms -> candidate lists (``candidates.jsonl``).

    python build_candidates.py records.csv mapping.json out/candidates.jsonl \
        --prefer Scotland "United Kingdom" Scottish British

The pipeline inside is: read terms (``records.py``) -> clean them (``clean.py``; turn off with ``--no-clean``)
-> retrieve candidates (``retrieve.py``). Use the same cleaning setting when you run the model in production.
Getty AAT / ULAN / TGN only by default. Add Wikidata with ``--wikidata-k 7 --contact you@example.org``.
Pass ``--getty-index cache/getty.sqlite`` (from ``build_index.py``) to search a local copy of the Getty
downloads instead of the live endpoint: milliseconds per term, same candidates.
Each output row is one term with its record text, its candidates and any gold URI the mapping supplied.
Re-runs are cheap: HTTP responses are cached in ``--cache``. Use ``--resume`` to skip terms already written.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from common import read_jsonl
from clean import clean_terms
from records import iter_terms, load_mapping
from retrieve import Http, Retriever

log = logging.getLogger("build")


def term_id(t) -> str:
    return f"{t.pid}|{t.field}|{t.term}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("records", help="your records: .csv, .jsonl (optionally .gz), or a folder of .jsonl[.gz] files")
    ap.add_argument("mapping", help="mapping JSON (see records.py)")
    ap.add_argument("out", help="output candidates.jsonl")
    ap.add_argument("--contact", default="",
                    help="contact (email or URL) for the User-Agent; required with --wikidata-k (Wikimedia policy)")
    ap.add_argument("--cache", default="cache/http.sqlite")
    ap.add_argument("--getty-index", help="local SQLite index from build_index.py; omit to query Getty's live endpoint")
    ap.add_argument("--getty-k", type=int, default=22, help="candidates kept from the Getty vocabulary")
    ap.add_argument("--wikidata-k", type=int, default=0, help="candidates kept from Wikidata (0: Getty only)")
    ap.add_argument("--prefer", nargs="*", default=[], help="tie-break hints found in hierarchy/biography text")
    ap.add_argument("--language", default="en")
    ap.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True,
                    help="tidy catalogue headings before searching (see clean.py); --no-clean searches terms as recorded")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many terms (smoke tests)")
    ap.add_argument("--resume", action="store_true", help="append, skipping terms already in --out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.wikidata_k and not args.contact:
        ap.error("--wikidata-k needs --contact: Wikimedia asks API clients to identify themselves")

    http = Http(args.cache, user_agent=f"recon-prep/0.1 ({args.contact})" if args.contact else "recon-prep/0.1")
    retriever = Retriever(http, args.getty_k, args.wikidata_k, args.prefer, args.language, args.getty_index)
    mapping = load_mapping(args.mapping)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out)} if args.resume and out.exists() else set()
    stats: Counter = Counter()
    terms = iter_terms(args.records, mapping)
    if args.clean:
        terms = clean_terms(terms, stats)
    log.info(f"cleaning step: {'on' if args.clean else 'off'}")
    with open(out, "a" if args.resume else "w", encoding="utf-8") as fh:
        for i, t in enumerate(terms):
            if args.limit and i >= args.limit:
                break
            if term_id(t) in done:
                continue
            try:
                cands = retriever.candidates(t.term, t.kind)
            except Exception as err:  # one bad term must not kill a long build; rerun with --resume
                log.warning(f"retrieval failed for {t.term!r} ({t.kind}): {err}")
                stats["failed"] += 1
                continue
            row = t.to_dict() | {"id": term_id(t), "candidates": [c.to_dict() for c in cands]}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["terms"] += 1
            stats[f"kind:{t.kind}"] += 1
            stats["no_candidates"] += not cands
            stats["has_exact"] += any(c.match == 3 for c in cands)
            if t.gold_uri and t.gold_uri.lower() != "none":
                stats["gold_given"] += 1
                stats["gold_retrieved"] += any(c.uri.rstrip("/") == t.gold_uri.rstrip("/") for c in cands)
            if stats["terms"] % 50 == 0:
                log.info(f"{stats['terms']} terms (cache {http.hits} hits / {http.misses} fetched)")
    if stats["gold_given"]:
        stats["gold_recall_%"] = round(100 * stats["gold_retrieved"] / stats["gold_given"], 1)
    log.info("done: " + json.dumps(dict(sorted(stats.items()))))


if __name__ == "__main__":
    main()
