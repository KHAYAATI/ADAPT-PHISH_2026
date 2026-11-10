# ADAPT-PHISH: leakage-audited cross-corpus evaluation of phishing URL detectors

Code, manifests, logs and results of the article *ADAPT-PHISH: Leakage-Audited Cross-Corpus Evaluation of Phishing URL Detectors and a Negative Result on Evidence-Gated Adaptation* (M. Khayati, A. Bourigue, Y. Ait Omar, M. Baslam; submitted to IEEE Access).

## Contents
| Folder | Content |
|---|---|
| `release/` | Package written by the notebook (cell R14) of the run behind every number of the article (v15, full profile, one NVIDIA Quadro P6000). It contains the notebook, environment, run configuration, dataset provenance, row-level split and purge manifests (SHA-256 of URLs), per-row predictions of every frozen model, ECRAC window logs, every result table and report, frozen models and `SHA256SUMS.txt`. See `release/README_RELEASE.md`. |
| `paper_scripts/` | Scripts that regenerate the result tables (`t_*.py`) and figures (`figs.py`) of the article from the run outputs. |
| `verification/` | Checks of data provenance: SHA-256 of the Kaggle copies and D4v, and the HTML-feature check of the PhreshPhish files. |

`release/configs/dataset_provenance.json` was corrected after the run (licences verified on the source pages, mirror links); `release/results/reports__dataset_provenance.json` is the record written by the run. `release/notebook/` holds the executed notebook with short cell titles; its code and outputs are those of the run.

## Data
The corpora are not redistributed here and are used exactly as published, with their original labels. The eight files used, byte-identical to the SHA-256 digests in Table 3 of the article, are mirrored at https://huggingface.co/datasets/KhayatiMohamed/DataAdaptPhish. The official PhreshPhish test split is read from its public release (commit `eca603e`). The notebook verifies every digest and stops if a file differs.

## Reproducing
1. `pip install -r requirements.txt` (Python 3.10; versions of the run in `release/environment/`).
2. Run `release/notebook/*.ipynb` from top to bottom. Results are written to `adapt_phish_v15_gpu_results/`.
3. Regenerate the tables and figures:
   ```
   set ADAPT_PHISH_RESULTS=adapt_phish_v15_gpu_results
   for %f in (t_main t_more t_ops t_ecrac t_posthoc t_abl t_seen figs) do python paper_scripts\%f.py
   ```
   Tables are written to `paper/tables/` and figures to `paper/figures/`.

Neural training used GPU mixed precision and is not bit-reproducible; the article reports the tolerances observed (Section VII).
