"""Candidate retrieval from Getty AAT / ULAN / TGN (SPARQL), optionally plus Wikidata (wbsearchentities).

Getty only by default (``wikidata_k=0``). Turn Wikidata on later by giving it a share of the candidates;
a model trained on Getty-only lists should be retrained before it is shown Wikidata candidates.

Getty's full-text index (``luc:term``) matches tokens but does not rank by relevance: "coin" returns
"coin cabinets" and "coin dies" long before ``coins (money)``, and "York" matches about 60 TGN places.
So each Getty search pulls a wide pool of (subject, label) pairs, ranks them here by how well a label
matches the term, fetches details for the best few dozen, then breaks ties with ``prefer`` hints
(e.g. "England", "British") found in the place hierarchy or biography.

Every HTTP response is cached in SQLite, so re-running a build costs nothing and is reproducible.
Use the same retrieval (and the same cache) at inference time, or the candidate lists the model sees
in production will differ from the ones it was trained on.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from common import chunked, summarise, uri_to_key

GETTY_SPARQL = "https://vocab.getty.edu/sparql.json"
BASE_URI = "http://vocab.getty.edu/"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Which vocabularies to search for each kind of term, and how many candidates to keep from each.
SOURCES = {
    "concept": ("aat", "wikidata"),
    "unit": ("aat", "wikidata"),
    "person": ("ulan", "wikidata"),
    "group": ("ulan", "wikidata"),
    "place": ("tgn", "wikidata"),
}
# Keep AAT to real concepts and ULAN to agents (no <guide terms> or facets). ULAN searches cover people AND
# groups for both kinds: catalogue "maker" fields mix them ("Wedgwood" is usually the firm, not a person).
_AGENT = "?Subject a ?agentType . filter(?agentType in (gvp:PersonConcept, gvp:GroupConcept))"
GETTY_TYPE = {"person": _AGENT, "group": _AGENT, "concept": "?Subject a gvp:Concept .", "unit": "?Subject a gvp:Concept ."}

_WORD = re.compile(r"\w+", re.UNICODE)
_PAREN = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")


# ---------------------------------------------------------------------------------------------- HTTP


class Http:
    """GET-with-JSON, polite per-host delay, retries with backoff, and a SQLite response cache."""

    def __init__(self, cache_path: str | Path, user_agent: str, delay: float = 0.5, retries: int = 4, timeout: int = 90):
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(cache_path))
        self.db.execute("create table if not exists cache (key text primary key, value text)")
        self.user_agent, self.delay, self.retries, self.timeout = user_agent, delay, retries, timeout
        self._last: dict[str, float] = {}
        self.hits = self.misses = 0

    def get_json(self, url: str, params: dict, post: bool = False) -> dict:
        """``post=True`` sends the parameters as a form body: Getty answers 403 to GET URLs over ~2 KB."""
        body = urllib.parse.urlencode(sorted(params.items()))
        key = f"{'POST ' if post else ''}{url}?{body}"
        row = self.db.execute("select value from cache where key = ?", (key,)).fetchone()
        if row:
            self.hits += 1
            return json.loads(row[0])
        self.misses += 1
        data = self._fetch(url, body, post)
        self.db.execute("insert or replace into cache values (?, ?)", (key, json.dumps(data)))
        self.db.commit()
        return data

    def _fetch(self, url: str, body: str, post: bool) -> dict:
        host = urllib.parse.urlparse(url).netloc
        for attempt in range(self.retries + 1):
            wait = self.delay - (time.monotonic() - self._last.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()
            req = urllib.request.Request(url if post else f"{url}?{body}", data=body.encode() if post else None,
                                         headers={"User-Agent": self.user_agent, "Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                if err.code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise
                backoff = float(err.headers.get("Retry-After") or 2 ** (attempt + 1))
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.retries:
                    raise
                backoff = 2 ** (attempt + 1)
            time.sleep(backoff)
        raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------------------------- matching


def norm(text: str) -> str:
    """Lower-case, drop parenthetical qualifiers ("coins (money)" -> "coins"), keep word characters."""
    return " ".join(_WORD.findall(_PAREN.sub("", text).lower()))


def variants(term: str, kind: str) -> list[str]:
    """Search forms of a term: singular/plural for concepts (AAT headings are plural), inverted names
    for people ("Barbara Hepworth" <-> "Hepworth, Barbara")."""
    base = _PAREN.sub("", term).strip()
    out = [base]
    if kind in ("person", "group") and "," in base:
        last, _, first = base.partition(",")
        out.append(f"{first.strip()} {last.strip()}")
    elif kind == "person" and len(base.split()) > 1:
        *first, last = base.split()
        out.append(f"{last}, {' '.join(first)}")
    if kind in ("concept", "unit"):
        low = base.lower()
        if low.endswith("ies"):
            out.append(base[:-3] + "y")
        elif low.endswith(("ches", "shes", "sses", "xes")):
            out.append(base[:-2])
        elif low.endswith("s") and not low.endswith("ss"):
            out.append(base[:-1])
        elif low.endswith("y") and low[-2:-1] not in "aeiou":
            out.append(base[:-1] + "ies")
        elif low.endswith(("ch", "sh", "ss", "x")):
            out.append(base + "es")
        else:
            out.append(base + "s")
    return list(dict.fromkeys(v for v in out if v))


def lucene_query(forms: list[str]) -> str:
    """``(barbara AND hepworth) OR (hepworth AND barbara)`` — token-only, so nothing needs escaping."""
    groups = []
    for form in forms:
        tokens = [t for t in _WORD.findall(form.lower()) if len(t) > 1 or t.isdigit()]
        if tokens:
            groups.append("(" + " AND ".join(tokens) + ")")
    return " OR ".join(dict.fromkeys(groups))


def match_score(label: str, forms: list[str]) -> int:
    """3 exact (after normalising), 2 every query word appears, 1 anything else the index returned."""
    lab = norm(label)
    normed = [norm(f) for f in forms]
    if lab in normed:
        return 3
    words = set(lab.split())
    if any(set(f.split()) <= words for f in normed if f):
        return 2
    return 1


def prefer_hits(text: str, prefer: list[str]) -> int:
    low = text.lower()
    return sum(p.lower() in low for p in prefer)


# ---------------------------------------------------------------------------------------------- sources


@dataclass
class Candidate:
    key: str
    uri: str
    source: str
    label: str
    description: str  # already "label — description", sanitised and capped, as the model sees it
    match: int = 1
    prefer: int = 0
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"key": self.key, "uri": self.uri, "source": self.source, "label": self.label,
                "description": self.description, "match": self.match}


class Getty:
    # The pool cap matches LocalGetty's: at 1,000 rows the live endpoint cut common words ("painting", "maps")
    # off in no particular order and lost candidates; at 20,000 both backends return identical lists.
    def __init__(self, http: Http, pool: int = 20_000, detail_top: int = 60):
        self.http, self.pool, self.detail_top = http, pool, detail_top

    def _select(self, query: str) -> list[dict]:
        data = self.http.get_json(GETTY_SPARQL, {"query": query}, post=True)
        return [{k: v["value"] for k, v in b.items()} for b in data["results"]["bindings"]]

    def pool_rows(self, query: str, scheme: str, kind: str) -> list[dict]:
        """``[{"Subject": uri, "Label": name}]`` for every name of every record the full-text query matches."""
        type_filter = GETTY_TYPE[kind] if scheme in ("aat", "ulan") else ""
        return self._select(
            f'select distinct ?Subject ?Label {{ ?Subject luc:term "{query}" ; skos:inScheme {scheme}: . {type_filter} '
            f"?Subject (xl:prefLabel|xl:altLabel)/xl:literalForm ?Label . }} limit {self.pool}"
        )

    def search(self, term: str, kind: str, scheme: str, k: int, prefer: list[str]) -> list[Candidate]:
        forms = variants(term, kind)
        q = lucene_query(forms)
        if not q:
            return []
        best: dict[str, int] = {}
        for r in self.pool_rows(q, scheme, kind):
            best[r["Subject"]] = max(best.get(r["Subject"], 0), match_score(r["Label"], forms))
        # Ties broken by URI, not by the order the index returned them, so live and local agree.
        shortlist = sorted(best, key=lambda s: (-best[s], s))[: self.detail_top]
        details = self.details(shortlist)
        cands = []
        for uri in shortlist:
            d = details.get(uri)
            if not d:
                continue
            desc = self.describe(scheme, d)
            cands.append(Candidate(uri_to_key(uri, scheme), uri, scheme, d["Pref"], summarise(d["Pref"], desc),
                                   match=best[uri], prefer=prefer_hits(desc, prefer), raw=d))
        # A preferred-label match beats one found only via an alternate label ("moils" lists "coin").
        cands.sort(key=lambda c: (-c.match, -match_score(c.label, forms), -c.prefer, len(c.label), c.key))
        return cands[:k]

    def details(self, uris: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for batch in chunked(uris, 30):
            values = " ".join(f"<{u}>" for u in batch)
            for r in self._select(
                f"select ?Subject ?Pref ?Note ?Parents ?Type ?Bio {{ values ?Subject {{ {values} }} "
                "?Subject gvp:prefLabelGVP/xl:literalForm ?Pref . "
                "optional { ?Subject skos:scopeNote [ dct:language gvp_lang:en ; rdf:value ?Note ] } "
                "optional { ?Subject gvp:parentStringAbbrev ?Parents } "
                "optional { ?Subject gvp:placeTypePreferred/gvp:prefLabelGVP/xl:literalForm ?Type } "
                "optional { ?Subject foaf:focus/gvp:biographyPreferred/schema:description ?Bio } }"
            ):
                out.setdefault(r["Subject"], r)
        return out

    @staticmethod
    def describe(scheme: str, d: dict) -> str:
        parents = (d.get("Parents") or "").replace(", ... ", ", ").removesuffix(" Facet").strip(", ")
        if scheme == "ulan":
            return d.get("Bio") or parents
        if scheme == "tgn":
            return ", ".join(x for x in (d.get("Type"), parents) if x)
        return d.get("Note") or (f"broader: {parents}" if parents else "")


class LocalGetty(Getty):
    """The same searches against a SQLite index built by ``build_index.py`` from Getty's N-Triples downloads.

    Answers ``pool_rows`` and ``details`` with the same fields as the live endpoint, so ranking and candidate
    text are shared with ``Getty``. The full-text index mimics ``luc:term`` (see ``build_index.py``); with the
    same pool cap the two return identical candidate lists, except where Getty has edited records since the
    download's snapshot.
    """

    _FILTER = {
        "concept": "and instr(',' || s.type || ',', ',Concept,')",
        "unit": "and instr(',' || s.type || ',', ',Concept,')",
        "person": "and (instr(',' || s.type || ',', ',PersonConcept,') or instr(',' || s.type || ',', ',GroupConcept,'))",
    }
    _FILTER["group"] = _FILTER["person"]

    def __init__(self, path: str | Path, pool: int = 20_000, detail_top: int = 60):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path}: build it with build_index.py")
        self.db = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
        self.pool, self.detail_top = pool, detail_top
        self.built = dict(self.db.execute("select scheme, finished from built").fetchall())

    def pool_rows(self, query: str, scheme: str, kind: str) -> list[dict]:
        if scheme not in self.built:
            raise LookupError(f"the index has no {scheme.upper()}: run build_index.py --schemes {scheme}")
        # Lucene "(barbara AND hepworth) OR (...)" -> FTS5 with every token quoted (tokens are word characters).
        fts = re.sub(r"[^\s()]+", lambda m: m[0] if m[0] in ("AND", "OR") else f'"{m[0]}"', query)
        type_filter = self._FILTER[kind] if scheme in ("aat", "ulan") else ""
        ids = [i for (i,) in self.db.execute(
            f"select f.id from label_fts f join subject s on s.id = f.id "
            f"where label_fts match ? and f.scheme = ? {type_filter} limit ?", (fts, scheme, self.pool))]
        # Like the live query: every name of each matching record, not only the names that matched.
        rows = []
        for batch in chunked(ids, 500):
            rows += self.db.execute(f"select id, label from label where id in ({','.join('?' * len(batch))})", batch).fetchall()
        return [{"Subject": BASE_URI + i, "Label": label} for i, label in rows]

    def details(self, uris: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for batch in chunked([u.removeprefix(BASE_URI) for u in uris], 500):
            marks = ",".join("?" * len(batch))
            for i, pref, parents, note, bio, ptype in self.db.execute(
                f"""select s.id, p.label, s.parents, n.text, b.text, tp.label from subject s
                    join pref p on p.id = s.id
                    left join note n on n.id = s.id
                    left join bio b on b.id = s.id
                    left join ptype t on t.id = s.id
                    left join pref tp on tp.id = t.aat
                    where s.id in ({marks})""", batch):
                row = {"Subject": BASE_URI + i, "Pref": pref, "Parents": parents, "Note": note, "Bio": bio, "Type": ptype}
                out[BASE_URI + i] = {k: v for k, v in row.items() if v is not None}
        return out


class Wikidata:
    def __init__(self, http: Http, language: str = "en"):
        self.http, self.language = http, language

    def search(self, term: str, kind: str, k: int, prefer: list[str]) -> list[Candidate]:
        forms = variants(term, kind)
        data = self.http.get_json(WIKIDATA_API, {"action": "wbsearchentities", "search": forms[0], "language": self.language,
                                                 "uselang": self.language, "type": "item", "limit": k, "format": "json"})
        cands = []
        for hit in data.get("search", []):
            label = hit.get("label") or hit["id"]
            desc = hit.get("description") or ""
            uri = hit.get("concepturi") or f"http://www.wikidata.org/entity/{hit['id']}"
            cands.append(Candidate(uri_to_key(uri, "wikidata"), uri, "wikidata", label, summarise(label, desc),
                                   match=match_score(label, forms), prefer=prefer_hits(desc, prefer)))
        return cands  # keep Wikidata's own ranking, which is relevance-based


class Retriever:
    def __init__(self, http: Http, getty_k: int = 22, wikidata_k: int = 0, prefer: list[str] | None = None,
                 language: str = "en", getty_index: str | Path | None = None):
        self.getty = LocalGetty(getty_index) if getty_index else Getty(http)
        self.wikidata = Wikidata(http, language)
        self.getty_k, self.wikidata_k, self.prefer = getty_k, wikidata_k, prefer or []

    def candidates(self, term: str, kind: str) -> list[Candidate]:
        out: list[Candidate] = []
        for source in SOURCES[kind]:
            if source == "wikidata":
                if self.wikidata_k > 0:
                    out += self.wikidata.search(term, kind, self.wikidata_k, self.prefer)
            else:
                out += self.getty.search(term, kind, source, self.getty_k, self.prefer)
        seen, unique = set(), []
        for c in out:
            if c.key not in seen:
                seen.add(c.key)
                unique.append(c)
        return unique
