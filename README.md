# recon_prep — training data for a GLiNER2.5-Decide reconciliation model

Turns your collection records into training data for fine-tuning a GLiNER2.5-Decide reconciliation model,
in the format the training script from
[yale-cultural-heritage/GLiNER2.5-Decide-linkedart-recon](https://huggingface.co/yale-cultural-heritage/GLiNER2.5-Decide-linkedart-recon)
(`training/train.py`) reads. The model links a term in a record (e.g. "coin") to Getty AAT / ULAN / TGN,
or answers `none`. Retrieval is **Getty only** by default; Wikidata can be switched on later (see below).

The approach is supervised fine-tuning of an encoder model for entity disambiguation with NIL prediction against
Getty vocabularies, using a retrieve-then-rerank pipeline.

This repo needs Python 3.10+ and the standard library only. Training and inference also need `gliner2` and a GPU
(step 6).

## How it fits together

```text
records (CSV / JSONL / Linked Art bundles)
   │ build_candidates.py:  read terms (records.py) → clean (clean.py, optional) → retrieve (retrieve.py)
   ▼
candidates.jsonl ──review.py export──▶ review.csv (people fill "choice") ──review.py import──▶ labelled.jsonl
                                                                                                   │
                                                              make_splits.py ◀─────────────────────┘
                                                                   ▼
                                              splits/{train,val,test}.jsonl ──train.py──▶ runs/<name>/checkpoints/best
                                                                                                   │
                                   new records → same read / clean / retrieve → Classifier ◀───────┘
```

In words, the pipeline runs in five stages:

1. **Read.** `records.py` reads your records (CSV, JSONL or nrich_la's Linked Art bundles) through a mapping file.
   The mapping says which field is the record id, which fields make up the context text the model will read, and
   which fields hold the terms to link, with each term's kind (concept, person, group or place).
2. **Clean and retrieve.** `build_candidates.py` drives this stage. Each term is optionally tidied by `clean.py`
   (catalogue dates, titles and punctuation removed) and then searched in Getty by `retrieve.py`, using either the
   live service or the local index built by `build_index.py`. The result is `candidates.jsonl`: one row per term,
   with its record text and up to 22 ranked candidates.
3. **Review.** `review.py export` turns the candidates into a spreadsheet, one question per term or per entity.
   People pick the right candidate, `none` or `skip` in the `choice` column. `review.py import` reads their answers
   back and writes `labelled.jsonl`.
4. **Split.** `make_splits.py` divides the labelled terms into train, val and test files, by record or by
   institution, and writes each row in the layout Yale's `train.py` expects, with the `none` label added.
5. **Train and use.** `train.py`, from Yale's model repo, fine-tunes the model on a GPU and saves the best
   checkpoint. To link new records, they go through exactly the same read, clean and retrieve steps, and the
   checkpoint, loaded with gliner2's `Classifier`, picks one candidate or `none` for each term.

Stages 1 to 4 need only this repo and the standard library. Stage 5 needs `gliner2` and a GPU. The same code builds
the model's input in training and in use, which is what keeps the two consistent.

| File | What it does |
| --- | --- |
| `records.py` | Reads CSV/JSONL records (gzipped, or a folder of them) through a mapping file: which fields are the id, the context text and the terms (with kind) |
| `clean.py` | Optional step between reading terms and searching: turns catalogue headings into search terms (`--no-clean` turns it off). Run on its own to preview what it changes |
| `retrieve.py` | Searches Getty (live SPARQL, or a local index) and optionally Wikidata for candidates, ranks them, caches responses in SQLite |
| `build_index.py` | Builds the local Getty index (`getty.sqlite`) from the downloaded N-Triples files |
| `compare_backends.py` | Checks the local index against the live service: same candidates, order and text |
| `build_candidates.py` | Step 3: one row per term with up to 22 candidates, plus retrieval statistics |
| `review.py` | Step 4: exports a spreadsheet for reviewers and imports their choices |
| `make_splits.py` | Step 5: writes train/val/test files, split by record or by institution, in the format `train.py` reads |
| `common.py` | `sanitize`, the `none` label, the input layout. These are copied from Yale's `recon_data.py` and must stay identical |
| `sample/` | Six example records and a mapping file to try the pipeline on, plus `linkedart_mapping.json` for nrich_la's Linked Art bundles |
| `docs/` | Time estimates for fine-tuning on the National Bibliography of Scotland (PDF), and the script that generates it |

## Quick start on the sample

```bash
python build_candidates.py sample/records.csv sample/mapping.json out/candidates.jsonl --prefer England "United Kingdom" British English
python review.py export out/candidates.jsonl out/review.csv
# open out/review.csv, fill the "choice" column, save
python review.py import out/candidates.jsonl out/labelled.jsonl out/review.csv
python make_splits.py out/labelled.jsonl out/splits
```

Six records are enough to check the pipeline runs but far too few to train on. For a real run, follow the guide below.

## NLS MARC dataset in this repo

`data/nls-marc/` holds everything needed to train on the NLS MARC datasets (Forth Bridge, Hutton drawings, Soviet
posters, theatre photos and posters, maps and so on). It covers 1,168 records, from nrich_la's Linked Art bundles.
The steps run on a machine with no network storage and no Getty index, because retrieval has already been done.

| File | What it is |
| --- | --- |
| `candidates.jsonl.gz` (7.7 MB) | 14,489 terms with their Getty candidates: the output of steps 2–3, cleaning on, `--prefer Scotland "United Kingdom" Scottish British`, Getty snapshot AAT Aug 2026 / ULAN, TGN Jan 2026 |
| `review.csv` | 1,327 questions covering the 12,326 terms that have candidates, most-mentioned first, **answered automatically** (see below) |
| `backup-review.csv` | The same sheet with `choice` blank, for a review by people. Its name keeps it out of `review*.csv` on import |

Terms with no candidates (2,163) aren't in the sheet. They can't become training examples.

**How `review.csv` was answered.** No one has reviewed it yet. `notes` says where each answer came from:
- **`auto (rule)`, 594 questions:** concepts where exactly one candidate's AAT name matched exactly.
- **`auto (AI)`, 733 questions:** Claude agents read each question's record text and candidates and answered
  645 with a candidate number, 78 `none` and 10 `skip`, each with a short reason.

These are machine labels. Training on them works, but `val` and `test` scores measure agreement with those
labels, not with people. For gold labels, have people check `review.csv`, starting with the `none` and
`skip` rows and the rule answers for common words, or fill in `backup-review.csv` from scratch.

**Reviewing by hand.** Fill `choice` as in [step 4](#4-review) and commit the sheet. Reviewers can split it by rows
and commit `review-<name>.csv` files instead. A partly filled sheet is fine: blank rows are skipped. In Excel, save
with **CSV UTF-8**, or accented names will be garbled.

**On the training machine:**

```bash
git clone <this repo> && cd recon_prep
python review.py import data/nls-marc/candidates.jsonl.gz out/labelled.jsonl data/nls-marc/review*.csv
python make_splits.py out/labelled.jsonl out/splits --max-per-entity 20
# then step 6: fetch Yale's training scripts and run train.py on out/splits/
```

The splits take seconds to rebuild, so they aren't committed. The current `review.csv` gives 4,609 train,
776 val and 894 test examples (about 20 MB). Without `--max-per-entity` it gives about 10,000 train, but
"National Library of Scotland" (publisher) and "Edinburgh" (place) are 952 of them each.

**Too few `none` answers.** Only 3.8% of train is `none`, against the 15–25% that teaches a model to abstain.
Most MARC headings have their exact match in Getty, so a model trained on this set will rarely say `none`.

**Licences.** The record text and headings come from National Library of Scotland metadata, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The candidate labels and descriptions come from the Getty
vocabularies (AAT, ULAN, TGN), licensed [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/). Credit both
wherever the data or a model trained on it is shared. The NLS records were changed on the way in:
- converted to Linked Art by nrich_la;
- reduced to the fields in `sample/linkedart_mapping.json`;
- headings tidied by `clean.py`.

To rebuild `data/nls-marc/` after changing the mapping, the cleaning rules or the Getty index (this needs the index):

```bash
python build_candidates.py ../nrich_la/out/nls/nls-marc sample/linkedart_mapping.json out/nls-marc/candidates.jsonl --getty-index C:/Users/AlastairPlant/repos/nrich_la/vocabularies/getty.sqlite --prefer Scotland "United Kingdom" Scottish British
gzip -9 -c out/nls-marc/candidates.jsonl > data/nls-marc/candidates.jsonl.gz
python review.py export data/nls-marc/candidates.jsonl.gz data/nls-marc/review.csv   # overwrites the sheet: copy answers across first
```

## Step-by-step guide

The examples use the NLS Linked Art bundles. Swap in your own records and mapping as needed.

### 1. Build the local Getty index (once)

The live service needs about 2–3 s per unique term to stay polite to Getty. A local index answers in milliseconds.
It gives the same candidates, and it lets training and production search the same snapshot.

```bash
python build_index.py C:/Users/AlastairPlant/repos/nrich_la/vocabularies --out C:/Users/AlastairPlant/repos/nrich_la/vocabularies/getty.sqlite
python compare_backends.py --index C:/Users/AlastairPlant/repos/nrich_la/vocabularies/getty.sqlite
```

The index already exists at that path (1.65 GB). See [Local Getty index](#local-getty-index) for details.

### 2. Write a mapping and preview the cleaning step

The mapping says which fields are terms and which are context; its format is at the top of `records.py`.
`sample/linkedart_mapping.json` covers nrich_la's bundles. Then see what the cleaning step would change:

```bash
python clean.py ../nrich_la/out/nls sample/linkedart_mapping.json
```

If a rule does more harm than good on your data, change it in `clean.py` now, or turn the step off with
`--no-clean` in step 3. **Decide here and don't change it later:** the trained model expects whichever form it was trained on.

### 3. Retrieve candidates

```bash
python build_candidates.py ../nrich_la/out/nls sample/linkedart_mapping.json out/candidates.jsonl --getty-index C:/Users/AlastairPlant/repos/nrich_la/vocabularies/getty.sqlite --prefer Scotland "United Kingdom" Scottish British
```

Check the statistics it logs at the end:

- `no_candidates`: terms that will be left out, because there's nothing to choose from.
- `gold_recall_%` (only if your mapping has `gold_field`): how often retrieval found the known answer. That's
  the upper limit on the model's accuracy.

Use `--limit 300` for a first look and `--resume` to carry on after an interruption.

### 4. Review

```bash
python review.py export out/candidates.jsonl out/review.csv
```

Reviewers fill the `choice` column with one of:

- a candidate number, key (`aat:300037222`) or URI;
- `none`, when no candidate is right;
- `skip`, when the record is too unclear to judge.

Rows come most-mentioned first, and with `entity_field` one row covers every mention of an entity, so the first
few hundred answers go a long way. Aim for 15–25% `none` overall: too few and the model never learns to abstain.
Split the sheet between reviewers if you like; import takes several CSVs:

```bash
python review.py import out/candidates.jsonl out/labelled.jsonl out/review.csv [out/review2.csv ...]
```

The import prints the share of `none` and how many terms were labelled.

### 5. Split

```bash
python make_splits.py out/labelled.jsonl out/splits
```

This splits by record: 80% train, 10% val, 10% test. Add `--max-per-entity 20` when a few entities appear on
most records, such as a publisher or a place named on every record. It keeps 20 mentions of each entity per
split, picked by a seeded hash so that every run keeps the same ones. Without entity ids, it counts the same term
with the same answer as one entity. To measure how the model copes with a collection it has
never seen, hold whole institutions out with `--by institution --test-groups "Collection A"`.

- `val` is what `train.py` reports scores on. It doesn't choose checkpoints with it, so the scores are fair.
- `test` is held back for choosing the abstention threshold (step 7).

### 6. Train (GPU machine)

The training code is Yale's, not part of this repo. It does full fine-tuning: the whole encoder and the
classification head are updated.

**a. Set up.** You need a CUDA GPU. With the defaults, the longest batch peaks at about 21 GiB, so a 24 GB card
or larger. Make a virtual environment with [uv](https://docs.astral.sh/uv/) (current Debian and Ubuntu refuse
`pip install` into the system Python), install the packages and fetch the scripts:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once, if uv isn't installed
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch numpy transformers gliner2 huggingface_hub
python -c "import torch; print(torch.cuda.is_available())"   # must print True
hf download yale-cultural-heritage/GLiNER2.5-Decide-linkedart-recon training/train.py training/recon_data.py --local-dir .
```

Python 3.12 rather than the system Python, because PyTorch wheels lag new Python releases. `torch`, `numpy` and
`transformers` are listed explicitly because `gliner2` doesn't install them. The default Linux `torch` wheel bundles
CUDA 12; if the check prints `False`, run `nvidia-smi` to see the CUDA version your driver supports and install a
matching build, for example `uv pip install torch --index-url https://download.pytorch.org/whl/cu126`. The rest of
this guide assumes the environment is activated; otherwise prefix each `python` command with `uv run`.

Copy `out/splits/` to the same machine. `train.py` imports `recon_data.py` from its own folder, so keep the two together.

**b. Check memory.** This runs the longest batch once and exits:

```bash
python training/train.py --train out/splits/train.jsonl --val out/splits/val.jsonl --out runs/preflight --gradient-checkpointing --preflight-only
```

If it runs out of memory, halve the batch and double the accumulation (`--batch-size 4 --grad-accum 4`), so the
effective batch stays at 16. On a GPU without bf16 support (older than Ampere), add `--no-bf16`.

**c. Smoke test.** This takes a few minutes and proves the data loads and the loss falls:

```bash
python training/train.py --train out/splits/train.jsonl --val out/splits/val.jsonl --out runs/smoke --max-train 200 --max-val 100 --epochs 1 --gradient-checkpointing
```

**d. Full run:**

```bash
python training/train.py --train out/splits/train.jsonl --val out/splits/val.jsonl --out runs/full --gradient-checkpointing
```

Defaults are Yale's settings: 2 epochs, effective batch 16, encoder learning rate 1e-5, head 5e-4, seed 42. Yale's
run of 24,318 examples took 283 minutes on one L40S; `docs/NBS_fine-tuning_estimates.pdf` has estimates for NBS.
Every 400 steps it checks loss on a small slice of train (`--dev-frac 0.05`) and keeps the best checkpoint.

`--base` chooses the starting model:

- `fastino/GLiNER2.5-Decide` (Apache-2.0) is the default.
- `yale-cultural-heritage/GLiNER2.5-Decide-linkedart-recon` gives you a head start, but your model then inherits
  its **CC BY-NC 4.0** licence (no commercial use).
  - `--replay <file> --replay-n 2000` mixes in 2,000 examples from another train file in the same format, so the
    model doesn't forget what it already knew. Use one built from Yale's `mds-linkedart-recon` data with their `recon_data.py`.

On a Slurm cluster, `training/train.sbatch` in the same model repo is Yale's job script.

**e. Read the results.** `runs/full/` contains:

| File | What's in it |
| --- | --- |
| `checkpoints/best/` | **The trained weights**: the best checkpoint by dev loss (`checkpoints/final/` if there's no `best`) |
| `benchmark.json` | Scores for the base model and the tuned model on `val`, overall and per kind, plus a threshold sweep |
| `predictions-base.jsonl`, `predictions-tuned.jsonl` | One prediction per val example: read these to see what it gets wrong |
| `train-summary.json` | Training settings and timings |

The four headline scores in `benchmark.json`:

- `overall_exact`: right answer, counting `none` as an answer.
- `accuracy_answerable`: accuracy on terms that have a right candidate.
- `abstention_recall`: how often it says `none` when `none` is right.
- `false_link_rate`: how often it links when it should have said `none`.

For a catalogue, the false-link rate is usually the one to watch, because a wrong link is worse than no link.
Compare tuned against base: the `improvement` field is the gain in `overall_exact`.

### 7. Score the hold-out and choose a threshold

The model always picks something, including `none`. You can also make it abstain when it isn't sure: treat any
answer whose probability falls below a threshold as `none`. Choose that threshold on `test`, not `val`, so the
`val` scores stay honest:

```bash
python training/train.py --train out/splits/train.jsonl --val out/splits/test.jsonl --out runs/full-test --eval-checkpoint runs/full/checkpoints/best --label test
```

`runs/full-test/benchmark.json` has `threshold_sweep`: the four scores at thresholds from 0.0 to 0.9. Pick the
lowest threshold whose false-link rate you can live with. For reference, Yale's model went from 0.123 to 0.115 at 0.8.

### 8. Use the trained weights

Load the checkpoint folder with gliner2's `Classifier`. The text and candidates must be built **exactly** as in
training, so build them with this repo's own code, as below:

- the same mapping (record text);
- the same cleaning setting;
- the same Getty index, `--prefer`, `k` values and description format;
- the `none` candidate.

```python
from gliner2.classification import ClassificationSchema, Classifier

from clean import clean_terms
from make_splits import to_example
from records import iter_terms, load_mapping
from retrieve import Http, Retriever

THRESHOLD = 0.8  # from step 7
clf = Classifier.from_pretrained("runs/full/checkpoints/best")
retriever = Retriever(Http("cache/http.sqlite", "recon-prep/0.1"),
                      getty_index="C:/Users/AlastairPlant/repos/nrich_la/vocabularies/getty.sqlite",
                      prefer=["Scotland", "United Kingdom", "Scottish", "British"])

terms = clean_terms(iter_terms("new_records/", load_mapping("sample/linkedart_mapping.json")))  # drop clean_terms if you trained with --no-clean
for t in terms:
    cands = [c.to_dict() for c in retriever.candidates(t.term, t.kind)]
    if not cands:
        continue  # nothing to choose from: leave unlinked
    ex = to_example(t.to_dict() | {"candidates": cands, "gold": None})  # adds the `none` label, like training
    result = clf.classify(ex["text"], ClassificationSchema().single("entity", ex["labels"]))
    key = result.value("entity")
    p = result.probabilities("entity")[key]
    uri = next((c["uri"] for c in cands if c["key"] == key), None)  # None when the answer is "none"
    if uri and p >= THRESHOLD:
        print(t.pid, t.term_raw or t.term, "->", uri, f"{p:.2f}")
```

`to_example` is the function that wrote the training rows, so the model sees the same input layout
(`Mention: … / Entity type: … / Text: …`) and the same labels (`aat:…` keys → `label — description`).
The code up to `clf.classify` has been tested in this repo. The two gliner2 calls follow Yale's model card, but
gliner2 isn't installed here, so check them against the version you install.

A GPU makes it faster: about 0.1 s per term on one GPU, about 0.15 s on an Apple M4 Max. Pass the links back to
nrich_la as `equivalent`, and the next export will have them as gold labels.

**Sharing the model:** the checkpoint folder is the whole model. Copy it, or upload it with
`hf upload <your-org>/<model-name> runs/full/checkpoints/best`. In the model card, state the base model
and its licence, and the Getty snapshot dates (Getty data is ODC-BY 1.0). Credit the training data's source too:
for the NLS dataset, National Library of Scotland metadata, CC BY 4.0.

## Local Getty index

The current index lives at `C:/Users/AlastairPlant/repos/nrich_la/vocabularies/getty.sqlite` (1.65 GB), next to the
downloads it was built from. The repo's `.gitignore` already excludes `vocabularies/`, so it won't be committed.
Snapshot dates: AAT August 2026, ULAN and TGN January 2026.

- **Input:** `aat/`, `ulan/` and `tgn/` folders, each holding `explicit.zip`. Members are read straight from the
  zip, or from an extracted `.nt` file beside it if there is one. Besides names and hierarchy, the build uses
  the definitions (`ScopeNotes`), ULAN biographies and TGN place types, which only exist inside the zips.
- **Build AAT as well as TGN:** TGN place types are AAT concepts, and get their names ("inhabited places") from the AAT rows.
- **How it matches the live search:** each record is one search document made of all its names, with the
  bracketed qualifier dropped and accents kept. That's how Getty's `luc:term` behaves.
  `compare_backends.py` reported identical candidates, order and text on its AAT test terms.
- **Snapshot:** the index reflects your download, not today's Getty. That's a benefit, because training and
  production see exactly the same data. Rebuild after refreshing the downloads and rerun `compare_backends.py`.
  Retrain after a rebuild: the descriptions the model learnt from may have changed.
- **Rebuilding:** `--schemes aat` rebuilds one vocabulary. `--fts-only` rebuilds just the search index from
  the stored names. A full build takes about 17 minutes (AAT 1 minute, ULAN 2.5, TGN 12, search index 1.5).

## Linked Art records (nrich_la output)

`build_candidates.py` reads the JSONL bundles nrich_la writes, either a single `.jsonl.gz` file or a whole folder.

- **What the mapping reads:**
  - The terms are creators, contributors, publishers, places of making and publication, and subjects (`about`).
  - `record_where` skips the entity records that are in the same folder.
- **Filters:** a filter such as `referred_to_by[classified_as.id=…/300435416].content` keeps the description and
  leaves out licence and admin notes. The format is documented in `records.py`.
- **Kinds:** `kind_field` takes each value's kind from its Linked Art `type`, so subjects split into concepts and places.
  Some headings carry the wrong `type`, and the term then gets the wrong kind:
  - People in Dublin Core `about` are typed `Type`, so they're searched as concepts.
  - Some corporate bodies are typed `Person`. That's harmless, because ULAN searches cover both.
- **Not used:**
  - `classified_as` is always "collection object".
  - `language` already links to AAT.
  - `made_of` holds `dcterms:format` values ("text", "PDF"), not materials.
- **No gold labels yet:** the entities have no `equivalent` links, so every term goes to review.
- **One question per entity:** `entity_field` gives each term its nrich_la entity id. `review.py export` asks one
  question per entity and gives the answer to every mention on import. On the NLS MARC bundles that means
  1,327 questions instead of 12,326, sorted with the most-mentioned first.
  - nrich_la makes these ids from the heading text, so undated namesakes share one id. Each row shows the first
    line of a few other records so a reviewer can spot that. Answer `skip` and re-export with `--per-mention`.
  - A single answer can label over a thousand mentions: "National Library of Scotland" as publisher, for example.
    Use `make_splits.py --max-per-entity` so those near-identical examples don't dominate training.

## The cleaning step

On by default. Library headings carry cataloguing punctuation, dates and titles that break name search, so
`Fowler, John, Sir, 1817-1898` is searched as `Fowler, John`. The record text the model reads is left unchanged,
and each row keeps the original in `term_raw`. The rules are listed at the top of `clean.py`.

On the first 300 NLS MARC terms it cut terms with no candidates from 97 to 44. It's a no-op on the CSV sample.
Pass `--no-clean` to search terms exactly as recorded. Whichever you choose, **production must make the same choice
and use the same rules**, because the model learns from the cleaned mention.

## Things to know

- **Keep retrieval identical between training and use.** Use the same `retrieve.py`, the same `k` values
  and the same description format when you run the model in production. The model learns from what the descriptions say.
- **Getty quirks.** Getty's text index doesn't rank results by relevance, which is why `retrieve.py` ranks
  them itself. Getty also rejects GET URLs over about 2 KB, so its queries are sent as POST.
- **Search pool size.** Each search considers up to 20,000 names before ranking. With a cap of 1,000, common
  words like "painting" and "maps" lost good candidates, because the live service returns rows in no particular order.
- **ULAN searches cover people and groups** for both `person` and `group` terms, because catalogue maker fields mix them.
- **Getty-only lists are shorter**, especially for people: ULAN covers artists, architects and makers, so most
  authors and printers come back with one or two candidates, or none. Terms with no candidates are left out of
  review; terms with one candidate are kept, because a candidate plus `none` still teaches the model to abstain.
- **Adding Wikidata later:** pass `--wikidata-k 7 --contact you@yourorg` (the contact goes into the User-Agent;
  Wikimedia's API policy asks clients to identify themselves). Its candidates aren't filtered by type, so family
  names and companies appear. Retrain after switching it on: a model trained on Getty-only lists has never seen
  a `wikidata:` candidate.
