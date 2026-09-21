# Getting real IAM into the project

The IAM Handwriting Database is free for non-commercial research but needs a (free) registration and a
licence agreement, so it cannot be fetched automatically. Nothing downstream (writer-ID model, calibration,
`evaluate.py` on real data) has been run on real IAM yet; it is all waiting on this step.

## 1. Download (official)

Register at <https://fki.tic.heia-fr.ch/databases/iam-handwriting-database> and download:

| Archive | Contents needed | Goes to |
|---|---|---|
| `ascii.tgz` | `forms.txt`, `lines.txt`, `words.txt` (and `sentences.txt`, unused) | `data/iam/ascii/` |
| `lines.tgz` | line images, `lines/a01/a01-000u/a01-000u-00.png` ... | `data/iam/lines/` |
| `words.tgz` (optional, only for `--unit words`) | word images | `data/iam/words/` |

Resulting layout (this is what `src/data/iam.py` reads):

```
data/iam/ascii/forms.txt
data/iam/ascii/lines.txt
data/iam/ascii/words.txt
data/iam/lines/a01/a01-000u/a01-000u-00.png
data/iam/words/a01/a01-000u/a01-000u-00-00.png
```

`data/iam/*` is gitignored (licence); `data/splits/{vatr,seeded}/*.json` are not (auditable writer lists).

## 2. Validate

```powershell
venv\Scripts\python scripts\check_split.py --expect-official
```

This checks the canonical split `data/splits/vatr/{train,val,test}.json` (`data.iam.split_source: vatr`; the IAM
writer split of HWT/VATr, 283/56/161 writers, made by `scripts/make_vatr_split.py`) -- or, with
`split_source: seeded` (fallback only), creates/verifies `data/splits/seeded/*.json` from `data.iam.split_seed`. It
asserts zero writer overlap, and
checks the counts against the published totals: 657 writers, 1,539 forms, 13,353 lines, 115,320 words. Any
mismatch (a partial download, a mis-parsed `forms.txt`, missing images) exits 1. The `forms.txt` writer-ID column
and the `lines.txt`/`words.txt` field layouts are what the loader assumes from the IAM documentation; this run is
the first time they meet the real files.

## 3. Then, in order

```powershell
venv\Scripts\python scripts\train_writer_id.py                       # writes models/checkpoints/writerid_<...>/
venv\Scripts\python scripts\calibrate_legibility.py                  # CER floor of the CNN scorer on 200 real crops
venv\Scripts\python scripts\evaluate.py --split val --max-writers 10 # generators on real val writers
```
