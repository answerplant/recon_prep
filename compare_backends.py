"""Check that the local Getty index gives the same candidates as the live SPARQL endpoint.

    python compare_backends.py --index cache/getty.sqlite                       # built-in test terms
    python compare_backends.py --index cache/getty.sqlite --candidates out/candidates.jsonl --limit 200

For each term it compares the two candidate lists: same set, same order, same top candidate, and identical
candidate text (what the model reads). With equal pool caps the only expected differences are records Getty
has edited since your download; rerun it after refreshing the downloads.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from common import read_jsonl
from retrieve import SOURCES, Getty, Http, LocalGetty

DEFAULT_TERMS = [
    ("coin", "concept"), ("bronze", "concept"), ("silver", "concept"), ("oil paint", "concept"), ("canvas", "concept"),
    ("painting", "concept"), ("sampler", "concept"), ("token", "concept"), ("vase", "concept"), ("sculpture", "concept"),
    ("stoneware", "concept"), ("jasper", "concept"), ("linen", "concept"), ("chapbooks", "concept"),
    ("broadsides", "concept"), ("maps", "concept"), ("plates", "concept"), ("illustrations", "concept"),
    ("woodcuts", "concept"), ("engravings", "concept"), ("bookplates", "concept"), ("pamphlets", "concept"),
    ("sermons", "concept"), ("ballads", "concept"), ("almanacs", "concept"),
    ("centimetres", "unit"), ("millimetres", "unit"), ("inches", "unit"),
    ("Barbara Hepworth", "person"), ("Wedgwood", "person"), ("Mary Wood", "person"), ("Robert Burns", "person"),
    ("Allan Ramsay", "person"), ("Henry Raeburn", "person"), ("Charles Rennie Mackintosh", "person"),
    ("W. & A.K. Johnston", "group"), ("Foulis Press", "group"),
    ("York", "place"), ("St Ives", "place"), ("Stoke-on-Trent", "place"), ("Edinburgh", "place"), ("Glasgow", "place"),
    ("Aberdeen", "place"), ("Inverness", "place"), ("Perth", "place"), ("Yorkshire", "place"), ("Zurich", "place"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index", required=True)
    ap.add_argument("--candidates", help="take terms from a candidates.jsonl instead of the built-in list")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=22)
    ap.add_argument("--cache", default="cache/http.sqlite")
    ap.add_argument("--contact", default="")
    ap.add_argument("--prefer", nargs="*", default=["Scotland", "England", "United Kingdom", "Scottish", "British", "English"])
    ap.add_argument("--schemes", nargs="*", default=["aat", "ulan", "tgn"])
    ap.add_argument("--show", type=int, default=3, help="print up to this many differing candidates per term")
    ap.add_argument("--live-pool", type=int, default=20_000,
                    help="row cap for the live search (retrieve.py's default); lower it to 1000 to see what a small cap loses")
    args = ap.parse_args()

    terms = DEFAULT_TERMS
    if args.candidates:
        terms = list(dict.fromkeys((r["term"], r["kind"]) for r in read_jsonl(args.candidates)))
    if args.limit:
        terms = terms[: args.limit]
    live = Getty(Http(args.cache, f"recon-prep/0.1 ({args.contact})" if args.contact else "recon-prep/0.1"), pool=args.live_pool)
    local = LocalGetty(args.index)

    stats: Counter = Counter()
    for term, kind in terms:
        scheme = SOURCES[kind][0]
        if scheme not in args.schemes:
            continue
        a = live.search(term, kind, scheme, args.k, args.prefer)
        b = local.search(term, kind, scheme, args.k, args.prefer)
        ka, kb = [c.key for c in a], [c.key for c in b]
        da, db = {c.key: c.description for c in a}, {c.key: c.description for c in b}
        same_text = [k for k in set(ka) & set(kb) if da[k] == db[k]]
        stats["terms"] += 1
        stats[f"{scheme}:terms"] += 1
        stats["same_set"] += set(ka) == set(kb)
        stats["same_order"] += ka == kb
        stats["same_top"] += ka[:1] == kb[:1]
        stats["shared"] += len(set(ka) & set(kb))
        stats["live_total"] += len(ka)
        stats["same_text"] += len(same_text)
        if ka != kb or len(same_text) != len(set(ka) & set(kb)):
            print(f"[{scheme}] {term!r}: live {len(ka)} / local {len(kb)}, shared {len(set(ka) & set(kb))}, "
                  f"top {ka[:1]} vs {kb[:1]}")
            for k in [k for k in ka if k not in kb][: args.show]:
                print(f"    live only : {da[k][:110]}")
            for k in [k for k in kb if k not in ka][: args.show]:
                print(f"    local only: {db[k][:110]}")
            for k in [k for k in set(ka) & set(kb) if da[k] != db[k]][: args.show]:
                print(f"    text differs {k}:\n      live : {da[k][:110]}\n      local: {db[k][:110]}")
    n = max(stats["terms"], 1)
    print(json.dumps({
        "terms": stats["terms"],
        "same_candidate_set_%": round(100 * stats["same_set"] / n, 1),
        "same_order_%": round(100 * stats["same_order"] / n, 1),
        "same_top_candidate_%": round(100 * stats["same_top"] / n, 1),
        "live_candidates_also_local_%": round(100 * stats["shared"] / max(stats["live_total"], 1), 1),
        "identical_text_of_shared_%": round(100 * stats["same_text"] / max(stats["shared"], 1), 1),
        "by_scheme": {k.split(":")[0]: v for k, v in stats.items() if k.endswith(":terms")},
    }, indent=2))


if __name__ == "__main__":
    main()
