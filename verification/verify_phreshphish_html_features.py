"""Consistency check of the HTML-derived columns of phreshphish_features.csv against the HTML of the official PhreshPhish release.

The feature names (LineOfCode, LongestLineLength, CntImages, CntFilesJS, CntForms, HasTitle, ...) follow the PhiUSIIL feature
set (Prasad & Chandra, 2023). The original extractor is not published with the file, so this script RECOMPUTES the simplest of
these features from the raw HTML of a sample of release rows (a few parquet row groups, not the ~190 GB) with documented
definitions, joins on the URL and reports exact-match rate and Spearman correlation per feature. High agreement shows that the
columns were computed from the same pages; disagreement is reported as such (then: state 'HTML features not verified' and keep
every cross-corpus claim on view P, which uses no HTML feature).

    python verify_phreshphish_html_features.py <dir with phreshphish_features.csv> [row_groups=6] [out.json]
"""
import sys, json, re
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import spearmanr
from huggingface_hub import HfApi, HfFileSystem

REPO, REV = "phreshphish/phreshphish", "eca603e0ecd4f698f8d4d226c088659e72681239"
d = sys.argv[1].rstrip("/"); NRG = int(sys.argv[2]) if len(sys.argv) > 2 else 6; out = sys.argv[3] if len(sys.argv) > 3 else "phreshphish_html_check.json"
F = pd.read_csv(f"{d}/phreshphish_features.csv"); F = F.drop_duplicates("url").set_index("url")
fs = HfFileSystem(); files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REV) if f.endswith(".parquet") and "/test-" not in f)
rng = np.random.default_rng(0); rows = []
for f in rng.choice(files, min(len(files), NRG), replace=False):
    with fs.open(f"datasets/{REPO}@{REV}/{f}", "rb", block_size=2 ** 22) as fh:
        pf = pq.ParquetFile(fh); g = int(rng.integers(pf.num_row_groups))
        cols = [c for c in ["url", "html"] if c in pf.schema_arrow.names]
        if "html" not in cols: raise SystemExit(f"no 'html' column in {f}: columns {pf.schema_arrow.names}")
        rows.append(pf.read_row_group(g, columns=cols).to_pandas())
H = pd.concat(rows, ignore_index=True); H = H[H.url.isin(F.index)].drop_duplicates("url")
print(f"{len(H)} release rows with HTML matched to the feature file")
I = re.I | re.S
DEF = {  # documented re-implementation (counts of tags / flags), PhiUSIIL-style
    "LineOfCode": lambda h: h.count("\n") + 1 if h else 0,
    "LongestLineLength": lambda h: max((len(x) for x in h.split("\n")), default=0),
    "HasTitle": lambda h: int(bool(re.search(r"<title[^>]*>.*?\S.*?</title>", h, I))),
    "CntImages": lambda h: len(re.findall(r"<img\b", h, I)),
    "CntFilesCSS": lambda h: len(re.findall(r"<link\b[^>]*\.css", h, I)),
    "CntFilesJS": lambda h: len(re.findall(r"<script\b[^>]*\bsrc\s*=", h, I)),
    "CntIFrame": lambda h: len(re.findall(r"<iframe\b", h, I)),
    "CntForms": lambda h: len(re.findall(r"<form\b", h, I)),
    "HasFavicon": lambda h: int(bool(re.search(r"<link\b[^>]*rel\s*=\s*[\"']?[^>]*icon", h, I))),
    "HasDescription": lambda h: int(bool(re.search(r"<meta\b[^>]*name\s*=\s*[\"']?description", h, I))),
    "HasPasswordField": lambda h: int(bool(re.search(r"<input\b[^>]*type\s*=\s*[\"']?password", h, I))),
    "HasHiddenFields": lambda h: int(bool(re.search(r"<input\b[^>]*type\s*=\s*[\"']?hidden", h, I))),
}
rep = {"rows_compared": int(len(H)), "row_groups": NRG, "features": {}}
for k, fn in DEF.items():
    if k not in F.columns: rep["features"][k] = "column absent"; continue
    mine = H.html.fillna("").map(fn).astype(float).values; theirs = F.loc[H.url, k].astype(float).values; ok = np.isfinite(theirs)
    rep["features"][k] = {"exact_match": float(np.mean(mine[ok] == theirs[ok])), "within_10pct": float(np.mean(np.abs(mine[ok] - theirs[ok]) <= 0.1 * np.maximum(1, np.abs(theirs[ok])))),
                          "spearman": float(spearmanr(mine[ok], theirs[ok]).correlation) if ok.sum() > 2 else None}
json.dump(rep, open(out, "w"), indent=2); print(json.dumps(rep, indent=2))
