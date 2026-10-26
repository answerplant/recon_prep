"""Build a local SQLite index of Getty AAT / ULAN / TGN from the "explicit" N-Triples downloads.

    python build_index.py C:/path/to/vocabularies --out cache/getty.sqlite              # all three
    python build_index.py C:/path/to/vocabularies --out cache/getty.sqlite --schemes aat

Expects ``<dir>/<scheme>/explicit.zip`` (any member already extracted next to it as ``*.nt`` is read from
disk instead). It holds just what ``retrieve.py`` asks the live SPARQL endpoint for:

* ``label``                  every preferred and alternate name (``xl:prefLabel|xl:altLabel / xl:literalForm``)
* ``label_fts``              one full-text document per record: all its names, qualifiers dropped, accents
                             kept — which is how the live ``luc:term`` index behaves ("oil paint" finds
                             "oil sticks" via its names "oil stick" + "paint stick"; "token" does not find
                             "patakons (nonmonetary tokens)"; "millimetre" does not find "papier millimétré")
* ``subject``                record type (gvp:Concept, gvp:PersonConcept, ...) and ``gvp:parentStringAbbrev``
* ``pref``                   ``gvp:prefLabelGVP`` name
* ``note``                   English ``skos:scopeNote``                       (ScopeNotes.nt)
* ``bio``                    ``gvp:biographyPreferred / schema:description``  (ULAN Biographies.nt)
* ``ptype``                  ``gvp:placeTypePreferred`` (an AAT id)           (TGN PlaceTypes.nt)

Build AAT as well as TGN: TGN place types are AAT concepts and get their names from the AAT rows.
Identifiers are stored without the ``http://vocab.getty.edu/`` prefix (``aat/300037222``).
Rough cost: AAT a few minutes, ULAN 10-15, TGN 30-60 (it streams ~15 GB of triples).
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sqlite3
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

log = logging.getLogger("index")

BASE = "http://vocab.getty.edu/"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDF_VALUE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#value"
XL_PREF = "http://www.w3.org/2008/05/skos-xl#prefLabel"
XL_ALT = "http://www.w3.org/2008/05/skos-xl#altLabel"
XL_LITERAL = "http://www.w3.org/2008/05/skos-xl#literalForm"
GVP = "http://vocab.getty.edu/ontology#"
SCOPE_NOTE = "http://www.w3.org/2004/02/skos/core#scopeNote"
DCT_LANGUAGE = "http://purl.org/dc/terms/language"
SCHEMA_DESCRIPTION = "http://schema.org/description"

SCHEMA = """
create table if not exists subject (id text primary key, scheme text, type text, parents text);
create table if not exists pref (id text primary key, label text);
create table if not exists note (id text primary key, text text);
create table if not exists bio (id text primary key, text text);
create table if not exists ptype (id text primary key, aat text);
create table if not exists label (id text, scheme text, label text);
create index if not exists label_id on label (id);
create virtual table if not exists label_fts using fts5 (names, id unindexed, scheme unindexed,
    tokenize = 'unicode61 remove_diacritics 0');
create table if not exists built (scheme text primary key, finished text, seconds real);
"""
STAGING = """
create temp table subj_raw (id text, kind text, value text);
create temp table term_link (subject text, term text, rel text);
create temp table term_lit (term text primary key, literal text) without rowid;
create temp table note_link (subject text, note text);
create temp table note_val (note text primary key, value text) without rowid;
create temp table note_lang (note text primary key, lang text) without rowid;
create temp table bio_link (subject text, bio text);
create temp table bio_desc (bio text primary key, text text) without rowid;
"""

_ESCAPE = re.compile(r"\\(u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|.)")
_SIMPLE = {"t": "\t", "n": "\n", "r": "\r", "b": "\b", "f": "\f", '"': '"', "'": "'", "\\": "\\"}


def unescape(s: str) -> str:
    if "\\" not in s:
        return s
    return _ESCAPE.sub(lambda m: chr(int(m[1][1:], 16)) if m[1][0] in "uU" else _SIMPLE.get(m[1], m[1]), s)


def short(uri: str) -> str:
    return uri[len(BASE):] if uri.startswith(BASE) else uri


def triples(lines: Iterator[str], predicates: set[str]) -> Iterator[tuple[str, str, str]]:
    """``(subject, predicate, object)`` for lines whose predicate is wanted; literals are unescaped, without
    their ``@lang`` / ``^^type`` suffix. Splits by hand: a full N-Triples parser is several times slower."""
    for line in lines:
        if not line.startswith("<"):
            continue
        s_end = line.find("> <", 1)
        p_end = line.find("> ", s_end + 3)
        if s_end < 0 or p_end < 0:
            continue
        p = line[s_end + 3 : p_end]
        if p not in predicates:
            continue
        o = line[p_end + 2 :].rstrip().removesuffix(".").rstrip()
        if o.startswith("<"):
            o = o[1:-1]
        elif o.startswith('"'):
            o = unescape(o[1 : o.rfind('"')])
        yield line[1:s_end], p, o


@contextmanager
def open_member(vocab_dir: Path, scheme: str, name: str):
    """Yield the lines of ``<SCHEME>Out_<name>.nt``: the extracted file if present, else straight from the zip."""
    member = f"{scheme.upper()}Out_{name}.nt"
    loose = vocab_dir / scheme / member
    if loose.exists():
        with open(loose, encoding="utf-8") as fh:
            yield fh
        return
    with zipfile.ZipFile(vocab_dir / scheme / "explicit.zip") as zf:
        with zf.open(member) as raw:
            yield io.TextIOWrapper(raw, encoding="utf-8")


class Loader:
    def __init__(self, db: sqlite3.Connection, batch: int = 50_000):
        self.db, self.batch = db, batch

    def load(self, sql: str, rows: Iterator[tuple]) -> int:
        n, buf = 0, []
        for row in rows:
            buf.append(row)
            if len(buf) >= self.batch:
                self.db.executemany(sql, buf)
                n += len(buf)
                buf.clear()
        if buf:
            self.db.executemany(sql, buf)
            n += len(buf)
        return n


def progress(lines: Iterator[str], label: str, every: int = 5_000_000) -> Iterator[str]:
    t0 = time.time()
    for i, line in enumerate(lines, 1):
        if i % every == 0:
            log.info(f"  {label}: {i / 1e6:.0f}M lines ({time.time() - t0:.0f}s)")
        yield line


_QUALIFIER = re.compile(r"\s*\([^()]*\)\s*$")


def build_fts(db: sqlite3.Connection, scheme: str) -> None:
    """One document per record from the ``label`` table: its names joined, trailing "(qualifier)" removed."""
    log.info(f"[{scheme}] full-text index")
    db.execute("delete from label_fts where scheme = ?", (scheme,))

    def docs():
        current, names = None, []
        for i, label in db.execute("select id, label from label where scheme = ? order by id", (scheme,)):
            if i != current and current is not None:
                yield " | ".join(dict.fromkeys(names)), current, scheme
                names = []
            current = i
            names.append(_QUALIFIER.sub("", label))
        if current is not None:
            yield " | ".join(dict.fromkeys(names)), current, scheme

    # Materialise first: SQLite cannot insert into one table while a cursor is still reading another.
    Loader(db).load("insert into label_fts (names, id, scheme) values (?, ?, ?)", iter(list(docs())))
    db.commit()


def build_scheme(db: sqlite3.Connection, vocab_dir: Path, scheme: str) -> None:
    t0 = time.time()
    prefix = f"{scheme}/"
    load = Loader(db).load
    for table in ("subject", "pref", "note", "bio", "ptype", "label"):
        db.execute(f"delete from {table} where id like ?", (prefix + "%",))
    db.executescript(STAGING)

    log.info(f"[{scheme}] subjects")
    with open_member(vocab_dir, scheme, "1Subjects") as fh:  # staged in SQLite: TGN has millions of subjects
        load("insert into subj_raw values (?, ?, ?)",
             ((s, "type" if p == RDF_TYPE else "parents", o.removeprefix(GVP))
              for s, p, o in ((short(s), p, o) for s, p, o in triples(progress(fh, "subjects"), {RDF_TYPE, GVP + "parentStringAbbrev"}))
              if s.startswith(prefix) and "/" not in s[len(prefix):]))
    db.execute("""insert or replace into subject
                  select id, ?, group_concat(case when kind = 'type' then value end, ','),
                         max(case when kind = 'parents' then value end)
                  from subj_raw group by id""", (scheme,))

    log.info(f"[{scheme}] terms")
    rels = {XL_PREF: "pref", XL_ALT: "alt", GVP + "prefLabelGVP": "gvp"}
    with open_member(vocab_dir, scheme, "2Terms") as fh:
        links, lits = [], []
        for s, p, o in triples(progress(fh, "terms"), {XL_PREF, XL_ALT, GVP + "prefLabelGVP", XL_LITERAL}):
            if p == XL_LITERAL:
                lits.append((short(s), o))
            else:
                links.append((short(s), short(o), rels[p]))
            if len(links) + len(lits) >= 200_000:
                db.executemany("insert into term_link values (?, ?, ?)", links)
                db.executemany("insert or ignore into term_lit values (?, ?)", lits)
                links.clear()
                lits.clear()
        db.executemany("insert into term_link values (?, ?, ?)", links)
        db.executemany("insert or ignore into term_lit values (?, ?)", lits)
    db.execute("create index temp.term_link_rel on term_link (rel)")
    db.execute("""insert into label select distinct l.subject, ?, t.literal from term_link l
                  join term_lit t on t.term = l.term where l.rel in ('pref', 'alt')""", (scheme,))
    db.execute("""insert or replace into pref select l.subject, t.literal from term_link l
                  join term_lit t on t.term = l.term where l.rel = 'gvp'""")
    build_fts(db, scheme)

    log.info(f"[{scheme}] scope notes")
    with open_member(vocab_dir, scheme, "ScopeNotes") as fh:
        for s, p, o in triples(fh, {SCOPE_NOTE, RDF_VALUE, DCT_LANGUAGE}):
            if p == SCOPE_NOTE:
                db.execute("insert into note_link values (?, ?)", (short(s), short(o)))
            elif p == RDF_VALUE:
                db.execute("insert or ignore into note_val values (?, ?)", (short(s), o))
            else:
                db.execute("insert or ignore into note_lang values (?, ?)", (short(s), short(o)))
    # The lowest-numbered English note, so repeated builds pick the same one.
    db.execute("""insert or ignore into note select l.subject, v.value from note_link l
                  join note_lang g on g.note = l.note and g.lang = 'language/en'
                  join note_val v on v.note = l.note
                  order by l.subject, cast(substr(l.note, instr(l.note, 'scopeNote/') + 10) as integer)""")

    if scheme == "ulan":
        log.info("[ulan] biographies")
        with open_member(vocab_dir, scheme, "Biographies") as fh:
            for s, p, o in triples(progress(fh, "biographies"), {GVP + "biographyPreferred", SCHEMA_DESCRIPTION}):
                if p == SCHEMA_DESCRIPTION:
                    db.execute("insert or ignore into bio_desc values (?, ?)", (short(s), o))
                else:  # ulan/500010387-agent -> ulan/500010387 (what foaf:focus links in AgentMap)
                    db.execute("insert into bio_link values (?, ?)", (short(s).removesuffix("-agent"), short(o)))
        db.execute("insert or replace into bio select l.subject, d.text from bio_link l join bio_desc d on d.bio = l.bio")

    if scheme == "tgn":
        log.info("[tgn] place types")
        with open_member(vocab_dir, scheme, "PlaceTypes") as fh:
            load("insert or replace into ptype values (?, ?)",
                 ((short(s), short(o)) for s, _, o in triples(progress(fh, "place types"), {GVP + "placeTypePreferred"})
                  if s.startswith(BASE + "tgn/") and "/rel/" not in s))

    for table in ("subj_raw", "term_link", "term_lit", "note_link", "note_val", "note_lang", "bio_link", "bio_desc"):
        db.execute(f"drop table temp.{table}")
    db.execute("insert or replace into built values (?, datetime('now'), ?)", (scheme, round(time.time() - t0, 1)))
    db.commit()
    counts = {t: db.execute(f"select count(*) from {t} where id like ?", (prefix + "%",)).fetchone()[0]
              for t in ("subject", "label", "pref", "note", "bio", "ptype")}
    log.info(f"[{scheme}] done in {time.time() - t0:.0f}s: {counts}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("vocab_dir", help="folder with aat/, ulan/, tgn/ each holding explicit.zip")
    ap.add_argument("--out", default="cache/getty.sqlite")
    ap.add_argument("--schemes", nargs="+", default=["aat", "ulan", "tgn"], choices=["aat", "ulan", "tgn"])
    ap.add_argument("--temp-dir", help="where SQLite puts its temporary files (needs several GB for TGN)")
    ap.add_argument("--fts-only", action="store_true",
                    help="only rebuild the full-text index (all built schemes) from the stored names; minutes, not an hour")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.temp_dir:  # read by SQLite when it first needs a temp file, so set before connecting
        import os
        os.environ["SQLITE_TMPDIR"] = os.environ["TMP"] = os.environ["TEMP"] = args.temp_dir
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.out)
    db.execute("pragma journal_mode = off")
    db.execute("pragma synchronous = off")
    db.execute("pragma cache_size = -1000000")  # ~1 GB page cache
    db.execute("pragma temp_store = file")
    if args.fts_only:
        db.execute("drop table if exists label_fts")  # recreated below with the current tokenizer settings
    db.executescript(SCHEMA)
    if args.fts_only:
        for (scheme,) in db.execute("select scheme from built").fetchall():
            build_fts(db, scheme)
    else:
        for scheme in sorted(args.schemes, key=["aat", "ulan", "tgn"].index):
            build_scheme(db, Path(args.vocab_dir), scheme)
    log.info("optimising full-text index")
    db.execute("insert into label_fts (label_fts) values ('optimize')")
    db.commit()
    db.close()


if __name__ == "__main__":
    main()
