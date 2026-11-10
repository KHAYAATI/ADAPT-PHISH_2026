# ADAPT-PHISH — reproducibility package (v15-gpu-full)

Generated automatically by the notebook on 2026-11-10T06:27:47.

* `notebook/` — the notebook that produced every number of the manuscript (run top to bottom; data paths in cell 01).
* `environment/` — exact package versions and platform.
* `configs/` — run configuration (seeds, training profile, ECRAC constants, calibrated tau_N), dataset provenance.
* `manifests/split_manifest.csv.gz` — every row of TRAIN / VALIDATION / TEST / Track-A pool / PhishTank stream: corpus, 0-based data-row
  number in the source file (`src_row`), SHA-256 of the normalized URL, of the model input and of the registrable domain, label.
* `manifests/purged_rows.csv.gz` — every evaluation row removed because it occurs in the development files, with the reason.
* `manifests/rows_kept_after_dedup.csv.gz` — rows of each source file that survived de-duplication and label-conflict removal.
* `predictions/` — calibrated scores of every frozen model on every evaluation set, linked to `src_row`.
* `ecrac/` — per-window ECRAC logs of every variant and seed (need N, risk R, decision, reason), adaptation events, tau_N calibration windows.
* `results/` — every table (CSV / LaTeX) and report of the run, including the revision analyses (prefix `revision__R*`).
* `frozen_model/` — frozen weights, temperatures, thresholds, preprocessing and their SHA-256 manifest.
* `SHA256SUMS.txt` — checksum of every file of this package.

The source corpora are not redistributed; download them from the URLs in `configs/dataset_provenance.json` and verify their SHA-256 against
`results/reports__dataset_manifest.json` before running the notebook. Deposit this folder (or `release.zip`) on Zenodo to obtain a DOI.
