from lib import *
import json as _json

def tab(env, label, caption, colspec, header, rows, note, size=r"\scriptsize", sep="3pt", width=None):
    width = width or (r"0.97\textwidth" if env == "table*" else r"\columnwidth")
    L = [rf"\begin{{{env}}}[!t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", size, rf"\setlength{{\tabcolsep}}{{{sep}}}",
         rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"] + header + [r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    if note: L.append(rf"\par\smallskip\parbox{{{width}}}{{\scriptsize {note}}}")
    L.append(rf"\end{{{env}}}")
    return "\n".join(L) + "\n"

# ---------------- validation ----------------
def vrow(model):
    r = VAL[(VAL.model == model) & (VAL.corpus == "MACRO (mean of corpora)")]
    assert len(r) == 1, model
    return r.iloc[0]
rows = []
for grp in [["ADAPT-X (P)", "ADAPT-Net (P)", "LightGBM (P)", "Char-LR", "LightGBM (P) + $n$-grams", "URLNet-style CNN", "BiGRU", "Simple stack (no neural)"],
            ["ADAPT-X (E)", "ADAPT-Net (E)", "LightGBM (E)"], list(SANITY)]:
    if rows: rows.append(r"\midrule")
    for m in grp:
        r = vrow(ENS.get(m) or SANITY[m])
        for k in ["roc_auc", "pr_auc", "f1", "mcc", "fpr", "ece"]: fact(f"val|{m}|{k}", r[k])
        rows.append(f"{m} & {f3(r.roc_auc)} & {f3(r.pr_auc)} & {f3(r.f1)} & {f3(r.mcc)} & {f3(r.fpr)} & {f3(r.ece)} \\\\")
for m in ["ADAPT-Net (P)", "ADAPT-Net (E)"]:
    for c in ["Mendeley", "PhreshPhish"]:
        fact(f"val|{m}|{c}|roc_auc", VAL[(VAL.model == ENS[m]) & (VAL.corpus == c)].iloc[0].roc_auc)
write("validation", tab("table", "tab:validation", "Validation results (macro over the two development corpora).", "lrrrrrr",
      [r"Model & ROC-AUC & PR-AUC & F1 & MCC & FPR & ECE \\"], rows,
      r"Weighted-MCC threshold; macro ECE (15 bins); ADAPT-X and simple-stack scores are out-of-fold. Neural models and LightGBM: ensembles of five seeds."))

# ---------------- leave-one-corpus-out between the development corpora ----------------
PX = rd("reports/pseudo_external_check.csv"); PX = PX[PX.threshold_rule.str.startswith("val_mcc")]
NM = {("ADAPT-PHISH", "P"): "ADAPT-Net (P)", ("ADAPT-PHISH", "E"): "ADAPT-Net (E)", ("LightGBM", "P"): "LightGBM (P)", ("LightGBM", "E"): "LightGBM (E)",
      ("char 3-5-gram LR", "P"): "Char-LR (P)", ("ADAPT-X stack", "P"): "ADAPT-X (P)"}
rows = []
for src, dst, tag in [("PhreshPhish", "Mendeley", r"PP\,$\to$\,ME"), ("Mendeley", "PhreshPhish", r"ME\,$\to$\,PP")]:
    if rows: rows.append(r"\midrule")
    for (mm, v), nm in NM.items():
        r = PX[(PX["trained on"] == src) & (PX.model == mm) & (PX.view == v)].iloc[0]
        for k in ["roc_auc", "pr_auc", "mcc", "fpr"]: fact(f"loco2|{src}->{dst}|{nm}|{k}", r[k])
        rows.append(f"{tag} & {nm} & {f3(r.roc_auc)} & {f3(r.pr_auc)} & {f3(r.mcc)} & {f3(r.fpr)} \\\\")
write("loco", tab("table", "tab:loco", "Leave-one-corpus-out transfer.", "llrrrr", [r"Direction & Model & ROC-AUC & PR-AUC & MCC & FPR \\"], rows,
      r"PP: PhreshPhish; ME: Mendeley. Single seed; thresholds from the source corpus.", sep="2pt"))

# ---------------- seeds ----------------
S = rd("revision/R11_seed_summary.csv")
SN = [("ADAPT-Net (P)", "ADAPT-PHISH (P)"), ("ADAPT-Net (E)", "ADAPT-PHISH (E)"), ("LightGBM (P)", "LightGBM (P)"), ("LightGBM (E)", "LightGBM (E)"),
      ("LightGBM (P) + $n$-grams", "LightGBM + n-grams"), ("URLNet-style CNN", "URLNet-style CNN"), ("BiGRU", "BiGRU"),
      ("ADAPT-Net v2, tuned (P)", "ADAPT-PHISH v2 (P)"), ("BiGRU, tuned", "BiGRU tuned"), ("URLNet-style, tuned", "URLNet tuned"),
      ("LightGBM (P), tuned", "LightGBM (P) tuned"), ("LightGBM + $n$-grams, tuned", "LightGBM + n-grams tuned")]
rows = []
for nm, rn in SN:
    if nm == "ADAPT-Net v2, tuned (P)": rows.append(r"\midrule")
    cells = []
    for ds in CORP:
        r = S[(S.model == rn) & (S.dataset == ds)].iloc[0]
        cells += [f"{f3(r['roc_auc mean'])}$\\pm${f3(r['roc_auc sd'])}", f"{f3(r['mcc mean'])}$\\pm${f3(r['mcc sd'])}"]
        fact(f"seed|{ds}|{nm}|auc_mean", r["roc_auc mean"]); fact(f"seed|{ds}|{nm}|auc_sd", r["roc_auc sd"]); fact(f"seed|{ds}|{nm}|mcc_sd", r["mcc sd"])
        fact(f"seed|{ds}|{nm}|mcc_mean", r["mcc mean"]); fact(f"seed|{ds}|{nm}|n", r["n_seeds"])
    rows.append(f"{nm} & " + " & ".join(cells) + r" \\")
fact("seed|max_auc_sd", S["roc_auc sd"].max()); fact("seed|max_mcc_sd", S["mcc sd"].max())
fact("seed|argmax_auc_sd", S.loc[S["roc_auc sd"].idxmax(), ["model", "dataset"]].tolist()); fact("seed|argmax_mcc_sd", S.loc[S["mcc sd"].idxmax(), ["model", "dataset"]].tolist())
write("seeds", tab("table*", "tab:seeds", "Single-seed variability (mean $\\pm$ SD over five seeds).", "lrrrrrrrr",
      [r" & \multicolumn{2}{c}{PhreshTest} & \multicolumn{2}{c}{Crawling2024} & \multicolumn{2}{c}{CompPhish} & \multicolumn{2}{c}{D4v} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
       r"Model & ROC-AUC & MCC & ROC-AUC & MCC & ROC-AUC & MCC & ROC-AUC & MCC \\"], rows,
      r"Individual seeds 42, 123, 2024, 3407 and 777 at the weighted-MCC threshold (the ensembles of \tabref{tab:test} and \tabref{tab:v14test} average their logits). Lower block: tuned configurations of the post hoc round. Per-seed values are released (R11).", sep="2.5pt"))

# ---------------- leakage ----------------
O = rd("revision/R5_overlap_by_key_all_pairs.csv"); ND = rd("revision/R5_near_duplicates.csv")
C = O.columns
want = [("TRAIN", "VALIDATION"), ("TRAIN", "PhreshTest"), ("VALIDATION", "PhreshTest"), ("development (all rows)", "PhreshTest"),
        ("TRAIN", "Crawling2024"), ("VALIDATION", "Crawling2024"), ("development (all rows)", "Crawling2024"),
        ("TRAIN", "CompPhish"), ("VALIDATION", "CompPhish"), ("development (all rows)", "CompPhish"),
        ("TRAIN", "D4v"), ("VALIDATION", "D4v"), ("development (all rows)", "D4v"),
        ("TRAIN", "TrackA_pool"), ("VALIDATION", "TrackA_pool"), ("TrackA_pool", "CompPhish"),
        ("TRAIN", "PhishTank"), ("VALIDATION", "PhishTank"), ("development (all rows)", "PhishTank")]
nd = {r["evaluated set"]: r["share"] for _, r in ND.iterrows()}
rows = []
for a, b in want:
    r = O[(O["A (reference)"] == a) & (O["B (evaluated)"] == b)]
    if not len(r): print("missing leakage pair", a, b); continue
    r = r.iloc[0]; dom = 100 * r[C[7]] / r[C[2]]
    ndv = f"{100 * nd[b]:.1f}" if a.startswith("development") and b in nd else ""
    an = "Dev.\\ (all)" if a.startswith("development") else a.replace("TrackA_pool", "Track-A pool"); b_ = b.replace("TrackA_pool", "Track-A pool")
    fact(f"leak|{a}|{b}", {"rows": int(r[C[2]]), "url": int(r[C[3]]), "input": int(r[C[4]]), "trunc": int(r[C[6]]), "dom_pct": dom, "nd_pct": None if not ndv else float(ndv)})
    rows.append(f"{an} & {b_} & {num(r[C[2]])} & {int(r[C[3]])} & {int(r[C[4]])} & {int(r[C[6]])} & {dom:.1f} & {ndv} \\\\")
write("leakage", tab("table", "tab:leakage", "Leakage audit after purging.", "llrrrrrr", [r"A & B & Rows B & URL & Input & Trunc. & Dom.\,\% & ND\,\% \\"], rows,
      r"URL / Input: rows of B whose normalized URL / model input occurs in A. Trunc.: rows whose truncated byte sequence equals that of a different input in A (not purged). Dom.\,\%: share of B rows whose registrable domain occurs in A. ND\,\%: share of B rows with a MinHash near-duplicate (estimated Jaccard $\ge$ 0.8) in development. Per-pair counts for all keys, including the canonical feature vector, are released (R5).", sep="2.5pt"))
for _, r in ND.iterrows():
    fact(f"nd|{r['evaluated set']}", {"rows": int(r["rows"]), "nd": int(r[ND.columns[2]]), "share": r["share"], "tmpl": int(r[ND.columns[4]])})
MW = rd("revision/R5_metrics_without_near_duplicates.csv")
fact("nd_metrics_max_abs_delta", None)
dd = []
for (ds, mdl), g in MW.groupby(["dataset", "model"]):
    if len(g.subset.unique()) >= 2:
        a = g[g.subset == g.subset.unique()[0]].roc_auc.values[0]; b = g[g.subset == g.subset.unique()[1]].roc_auc.values[0]; dd.append(abs(a - b))
fact("nd_metrics_subsets", MW.subset.unique().tolist()); fact("nd_metrics_max_abs_delta", max(dd) if dd else None)

# ---------------- shortcut ----------------
SA = rd("reports/shortcut_audit.csv"); SAll = rd("reports/shortcut_audit_all_corpora.csv"); ST = rd("revision/R8_shortcut_flag_stability.csv")
flagged = SA[SA.flag.ne("OK") & SA.flag.notna()]
rows = []
for _, r in flagged.iterrows():
    f = r.feature; nm = f.split("::")[-1].replace("_", r"\_")
    struct = f.startswith("struct::")
    a = SAll[SAll.feature == f].iloc[0]; s = ST[ST.feature == f]
    pinv = f"{s.iloc[0][ST.columns[5]]:.2f}" if len(s) else "--"
    flag = {"SIGN_INVERSION": "inversion", "STRONG_UNIVARIATE": "strong"}.get(r.flag, r.flag.lower())
    act = "audit only" if struct else ("removed" if "REMOVED" in str(r.action) else "kept")
    lab = "path segments (structural)$^\\dagger$" if struct else rf"\texttt{{{nm}}}" + ("$^\\ddagger$" if nm == r"path\_depth" else "")
    tc = " & ".join(["--"] * 4) if struct else " & ".join(f3(a[c]) for c in ["CompPhish", "Crawling2024", "D4v", "PhreshTest"])
    fact(f"shortcut|{f}", {"Mendeley": r.Mendeley, "PhreshTrain": r.PhreshTrain, "pinv": None if pinv == "--" else float(pinv), "action": act,
                            **({} if struct else {c: a[c] for c in ["CompPhish", "Crawling2024", "D4v", "PhreshTest"]})})
    rows.append(f"{lab} & {f3(r.Mendeley)} & {f3(r.PhreshTrain)} & {pinv} & {flag} & {act} & {tc} \\\\")
write("shortcut", tab("table*", "tab:shortcut", "Flagged canonical URL features and the audited structural variable.", "lrrrllrrrr",
      [r"Feature & Mendeley & PhreshPhish & $P_{\mathrm{inv}}$ & Flag & Action & CompPhish & Crawl.24 & D4v & PhreshTest \\"], rows,
      r"Univariate ROC-AUC on TRAIN rows (development) and on test rows (post-freeze diagnostic, no decision). $P_{\mathrm{inv}}$: share of 200 registrable-domain cluster resamples of TRAIN in which the sign inversion is reproduced. $^\ddagger$\,\texttt{path\_depth} was removed by a rule fixed before the audit and also meets the inversion criterion. $^\dagger$\,Structural path-segment count, audited but never a model input; it is listed once as a model feature and once as the audited variable, so six features were removed.", size=r"\footnotesize"))

# ---------------- sanity ----------------
rows = []
for m, rn in SANITY.items():
    v = [tm(rn, ds).roc_auc for ds in CORP]
    for ds, x in zip(CORP, v): fact(f"sanity|{m}|{ds}", x)
    rows.append(f"{m} & " + " & ".join(f3(x) for x in v) + r" \\")
write("sanity", tab("table", "tab:sanity", "One-variable sanity baselines (ROC-AUC).", "lrrrr", [r"Baseline & PhreshTest & Crawl.24 & CompPhish & D4v \\"], rows,
      r"Logistic regression on one or two URL variables, fitted on TRAIN; the direction of the score is learned, so values below 0.5 mean that the relation found in development is reversed on the test corpus."))

# ---------------- loco over all labelled corpora (light baselines) ----------------
R13 = rd("revision/R13_loco_all_labelled_corpora.csv"); R13 = R13[R13["threshold rule"] == "val_mcc"]
rows = []
for hc, g in R13.groupby("held-out corpus"):
    v = {m: g[g.model == m].roc_auc.values[0] for m in ["LightGBM (P)", "char 3-5-gram LR", "Sanity: url_len only (LR)"]}
    n = g["rows held out (after purge)"].values[0]
    for m, x in v.items(): fact(f"loco_all|{hc}|{m}", x)
    rows.append(f"{hc} & {num(n)} & {f3(v['LightGBM (P)'])} & {f3(v['char 3-5-gram LR'])} & {f3(v['Sanity: url_len only (LR)'])} \\\\")
write("loco_all", tab("table", "tab:loco_all", "Leave-one-corpus-out over all labelled corpora (ROC-AUC).", "lrrrr",
      [r"Held-out corpus & Rows & LightGBM (P) & Char-LR & URL length \\"], rows,
      r"Each corpus is held out and the light baselines are trained on the four others, with the same purge and threshold rules."))

# ---------------- corpora ----------------
DA = rd("reports/data_audit.csv").set_index("dataset"); DD = rd("reports/deduplication.csv").set_index("dataset"); TP = rd("reports/test_purge.csv").set_index("test")
MAN = {x["name"]: x for x in _json.loads((RES / "reports/dataset_manifest.json").read_text())}
def unseen(ds):
    sub = "unseen_domain (domain-disjoint)" if ds == "PhreshTest" else "unseen_domain"
    return int(tm(ENS["ADAPT-X (P)"], ds, subset=sub).n)
def nsplit(ds, sp):
    return int(TR.iloc[0:0].shape[0])
roles = {"Mendeley": "Development", "PhreshTrain": "Development", "PhreshTest": "IID test", "Crawling2024": "Secondary external test",
         "CompPhish": "External test + pool", "D4v": "External test (small)", "PhishTank": "Phishing-only stream"}
VN = VAL[(VAL.model == ENS["ADAPT-X (P)"])]
rows = []
for ds, role in roles.items():
    raw = DA.loc[ds, "rows"]; ded = DD.loc[ds, "rows_out"]; sha = str(MAN[ds].get("sha256", ""))[:12]
    if ds in ("Mendeley", "PhreshTrain"):
        vr = VAL[(VAL.model == ENS["ADAPT-X (P)"]) & (VAL.corpus == ("Mendeley" if ds == "Mendeley" else "PhreshPhish"))].iloc[0]
        final = f"VAL {num(vr.n)}"; pr = DA.loc[ds, "phishing"] / DA.loc[ds, "rows"]; un = "--"; fact(f"val_rows|{ds}", vr.n)
    elif ds == "PhishTank":
        final = num(TP.loc["PhishTank (stream)", "rows_after"]); pr = 1.0; un = "--"
    else:
        r = tm(ENS["ADAPT-X (P)"], ds); final = num(r.n) + (f" + {num(TP.loc['CompPhish -> Track-A pool', 'rows_moved'])} pool" if ds == "CompPhish" else "")
        pr = r.positives / r.n; un = num(unseen(ds))
    rows.append(f"{ds} & {role} & {num(raw)} & {num(ded)} & {final} & {pr:.3f} & {un} & \\texttt{{{sha}}} \\\\")
write("corpora", tab("table*", "tab:corpora", "Corpora, roles, row counts and file checksums.", "llrrrrrl",
      [r"Corpus & Role & Raw & After dedup. & Final & Phish. rate & Unseen dom. & SHA-256 (first 12) \\"], rows,
      r"Dedup.: exact duplicates, normalized-URL duplicates and rows of URLs carrying both labels removed within each file. Final: rows after the development purge; the development corpora are split by registrable domain into TRAIN (257,167 rows) and VALIDATION (53,878 rows; the column gives the VALIDATION rows per corpus), after removing 17 URLs present in both development files; the split manifest is released. Unseen: test rows whose registrable domain is absent from TRAIN. SHA-256 digests of the files as used are released in full."))
for ds in ["PhreshTest", "Crawling2024", "CompPhish", "D4v"]:
    fact(f"purge|{ds}", {k: TP.loc[ds, k] for k in TP.columns if not pd.isna(TP.loc[ds, k])})
fact("purge|PhishTank", {k: TP.loc["PhishTank (stream)", k] for k in TP.columns if not pd.isna(TP.loc["PhishTank (stream)", k])})
fact("trackA_pool_rows", TP.loc["CompPhish -> Track-A pool", "rows_moved"])

# ---------------- table8check ----------------
R2 = rd("revision/R2_table8_prediction_level_verification.csv")
rows = []
for ds in CORP:
    g = R2[R2.dataset == ds]
    a = g[g.comparison.str.startswith("v13 seed ensemble")].iloc[0]; b = g[g.comparison.str.startswith("v12 seed ensemble")].iloc[0]
    c = g[g.comparison.str.startswith("LightGBM")].iloc[0]; nat = "yes" if bool(a["native tokens in corpus"]) else "no"
    rows += [f"{ds} & {nat} & Seed ensemble, revised (mean raw logit, one temperature) & {f3(a[R2.columns[3]])} & {f3(a['ROC-AUC P'])} & {f3(a['ROC-AUC E'])} \\\\",
             f"{ds} & {nat} & Seed ensemble, original (each member scaled by its own view temperature) & {f3(b[R2.columns[3]])} & {f3(b['ROC-AUC P'])} & {f3(b['ROC-AUC E'])} \\\\",
             f"{ds} & {nat} & LightGBM (P) vs (E): two different models & -- & {f3(c['ROC-AUC P'])} & {f3(c['ROC-AUC E'])} \\\\"]
    fact(f"pe|{ds}|orig_maxdiff", b[R2.columns[3]]); fact(f"pe|{ds}|rev_maxdiff", a[R2.columns[3]])
write("table8check", tab("table*", "tab:p_e_check", "Prediction-level check of the (P) and (E) results.", "lllrrr",
      [r"Corpus & Natives & Comparison & max $|z_E - z_P|$ & ROC-AUC (P) & ROC-AUC (E) \\"], rows,
      r"$z$: logit before temperature scaling. ADAPT-Net (P) and (E) are two passes of the same network, with and without the native-feature tokens; LightGBM (P) and (E) are separate models. Per-seed rows, temperatures and SHA-256 digests of the prediction files are released (R2)."))
dump_facts()
print("ok")
