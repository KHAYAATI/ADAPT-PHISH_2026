"""Shared helpers: every number in the regenerated tables is read from the v15 result files."""
import ast, json, os, re
from pathlib import Path
import numpy as np, pandas as pd

RES = Path(os.environ.get("ADAPT_PHISH_RESULTS", "adapt_phish_v15_gpu_results"))   # the run output folder (reports/, revision/, tables/, ...)
OUT = Path(os.environ.get("ADAPT_PHISH_TABLES_OUT", "paper/tables")); OUT.mkdir(parents=True, exist_ok=True)
FACTS = {}            # key -> value used in the prose (dumped to facts.json for the text rewrite and the audit)

def rd(p, **kw):
    return pd.read_csv(RES / p, **kw)

TR = rd("reports/test_results_all.csv")
VAL = rd("reports/validation_results.csv")

# display names (manuscript) <- run names
ENS = {
    "ADAPT-X (P)": "ADAPT-X stack (P)", "ADAPT-Net (P)": "ADAPT-PHISH (P) seed-ens", "LightGBM (P)": "LightGBM (P) seed-ens",
    "Char-LR": "char 3-5-gram LR", "LightGBM (P) + $n$-grams": "LightGBM (P) + hashed char n-grams seed-ens",
    "URLNet-style CNN": "URLNet-style char+word CNN seed-ens", "BiGRU": "BiGRU character model seed-ens",
    "Simple stack (no neural)": "Simple stack (P): LightGBM + char-LR, no neural",
    "ADAPT-X (E)": "ADAPT-X stack (E)", "ADAPT-Net (E)": "ADAPT-PHISH (E) seed-ens", "LightGBM (E)": "LightGBM (E) seed-ens",
}
TUNED = {
    "ADAPT-X2 (P)": "ADAPT-X2 stack (P)", "Baseline stack (no ADAPT)": "Baseline stack (P): tuned baselines, no ADAPT",
    "ADAPT-Net v2, tuned (P)": "ADAPT-PHISH v2 (P) seed-ens", "BiGRU, tuned": "BiGRU tuned seed-ens", "URLNet-style, tuned": "URLNet tuned seed-ens",
    "LightGBM + $n$-grams, tuned": "LightGBM + n-grams tuned seed-ens", "Char-LR, tuned": "char n-gram LR tuned",
    "LightGBM (P), tuned": "LightGBM (P) tuned seed-ens", "ADAPT-Net v2-KD (P)": "ADAPT-PHISH v2-KD (P) seed-ens",
    "BiGRU-KD, tuned": "BiGRU-KD tuned seed-ens", "CANINE-s, fine-tuned": "Pretrained google/canine-s (fine-tuned)",
    "BERT-mini (URLTran-style)": "Pretrained google/bert_uncased_L-4_H-256_A-4 (fine-tuned)",
}
SANITY = {"URL length only": "Sanity: url_len only (LR)", "Has-path only": "Sanity: has_path only (LR)",
          "URL length + path depth": "Sanity: url_len + path depth (LR)"}
CORP = ["PhreshTest", "Crawling2024", "CompPhish", "D4v"]

def f3(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "--"
    s = f"{x:.{d}f}"
    return "0." + "0" * d if s in ("-0." + "0" * d,) else s

def sg(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "--"
    s = f"{x:+.{d}f}"
    return s.replace("-0." + "0" * d, "+0." + "0" * d)

def pci(s):
    """'(0.1, 0.2)' or '[0.1, 0.2]' -> (lo, hi)"""
    if not isinstance(s, str): return (np.nan, np.nan)
    v = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", s)]
    return (v[0], v[1]) if len(v) >= 2 else (np.nan, np.nan)

def ci(lo, hi, d=3):
    return f"[{f3(lo, d)}, {f3(hi, d)}]"

def num(n):
    return f"{int(n):,}"

def tm(model, ds, rule="val_mcc", subset="ALL"):
    r = TR[(TR.model == model) & (TR.dataset == ds) & (TR.threshold_rule == rule) & (TR.subset == subset)]
    assert len(r) == 1, (model, ds, rule, subset, len(r))
    return r.iloc[0]

def write(name, body):
    (OUT / f"{name}.tex").write_text(body)

def fact(k, v):
    FACTS[k] = v
    return v

def dump_facts():
    p = Path(__file__).with_name("facts.json")
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in FACTS.items()})
    p.write_text(json.dumps(old, indent=1, ensure_ascii=False, default=str))
