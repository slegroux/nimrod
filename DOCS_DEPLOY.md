# Docs deploy (GitHub Pages) — status & deferred fix

Status: **failing — deferred.** The published site
(<https://slegroux.github.io/nimrod/>) is frozen at the last green build
(**Apr 6 2025**, `gh-pages` commit `0619f9f`). This is **pre-existing**: the
deploy has been red since well before the torch-2.9 / nbdev-toolchain work
(every run back through `75923a9` failed). CI (`test.yaml`) is green and
unaffected; only the docs site is stale.

## How the deploy works

```
push to main ─▶ .github/workflows/deploy.yaml
                 └─ fastai/workflows/quarto-ghp@master  (version: '3.10')
                      1. pip install -e ".[dev]"  (+ nbdev<3)
                      2. nbdev_docs ─▶ proc_nbs() EXECUTES every notebook,
                                        then `quarto render`
                      3. pushes built HTML to the gh-pages branch
GitHub Pages (source: gh-pages branch, path /)  serves the site.
```

The action only pushes to `gh-pages` **if `nbdev_docs` succeeds**, so any
execution failure freezes the live site.

## Root cause (why it's hard)

`nbdev_docs` re-executes **every** notebook during `proc_nbs`
(`nbdev/serve.py` → `nbdev/serve_drv.py:exec_nb`), and **that path does not gate
on `skip_exec`** the way `nbdev_test` does — `exec_nb` runs the notebook
unconditionally. So `skip_exec` frontmatter (which fixes CI) does **not** stop a
notebook from executing during the docs build.

The Apr-2025 build was green only because *every* notebook executed cleanly back
then. Several no longer do, for reasons independent of this work:

| Notebook(s) | Why execution fails |
|---|---|
| `image.med`, `image.blip` | vendored Salesforce BLIP imports `apply_chunking_to_forward` / `transformers.file_utils`, removed in modern `transformers` (already broken on 4.57.x) |
| `audio.utils`, `audio.features`, `models.aligners` | `torchaudio.load` now needs `torchcodec` + a matching FFmpeg; not installed in the docs runner (and torchcodec 0.14 is ABI-incompatible with torch 2.9.1) |
| `audio.datasets.stt` | needs a LibriSpeech manifest (`librispeech_recordings_dev-clean-2.jsonl.gz`) not committed to the repo |
| most notebooks (Python ≥3.11) | a dependency interaction raises `TypeError: argument of type 'function' is not iterable` during execnb execution — does **not** occur on Python 3.10 |

## What's already been done (this branch / merged)

- **`deploy.yaml` pinned to Python 3.10** (`version: '3.10'`). The docs runner
  defaulted to 3.12, which hits the `function is not iterable` bug across most
  notebooks; 3.10 matches the green-CI environment where they execute.
- **`skip_exec` frontmatter corrected** on `image.med`, `audio.utils`,
  `audio.features`, `models.aligners` (added) and `audio.datasets.stt`
  (genuine bug: it was in a *markdown* cell, which nbdev does **not** parse as
  frontmatter — moved to a *raw* cell). These help `nbdev_test`/CI, but **not**
  the docs build, per the root cause above.

These are correct improvements but **not sufficient** to green the deploy.

## Options to actually finish (pick later)

1. **Make every notebook execute** in the docs runner — fix the pre-existing
   breakage: re-vendor/replace BLIP for modern `transformers`, get
   `torchcodec` + FFmpeg working (or switch audio I/O off `torchaudio.load`),
   and provide or guard the data-dependent notebooks. Largest effort; overlaps
   the BLIP / audio tech debt tracked elsewhere.
2. **Render without executing** — replace the `quarto-ghp` action with explicit
   steps that run `quarto render` directly (bypassing `nbdev_docs`'
   `proc_nbs` execution) so the site is built from the notebooks' **committed
   outputs**. Off the beaten path; must also handle the `gh-pages` push +
   permissions the action currently does for free. Site is then only as fresh
   as the committed outputs.
3. **Leave deferred** (current choice) — CI is the gate that matters; refresh
   the docs site as its own task.

## Pointers

- Deploy workflow: `.github/workflows/deploy.yaml`
- nbdev exec path: `nbdev/quarto.py:nbdev_docs` → `nbdev/serve.py:proc_nbs` →
  `nbdev/serve_drv.py:exec_nb`
- Related tech debt: BLIP (`nimrod/image/med.py`, `blip.py`), audio I/O
  (`torchcodec`), and the nbdev 2→3 migration ([NBDEV3_MIGRATION.md](NBDEV3_MIGRATION.md)).
