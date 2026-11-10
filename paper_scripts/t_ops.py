from lib import *
from t_more import tab

R4 = rd("revision/R4_operating_points_with_CIs.csv")
R4N = {"ADAPT-X (P)": "ADAPT-X (P)", "ADAPT-Net (P)": "ADAPT-PHISH (P)", "LightGBM (P)": "LightGBM (P)", "Char-LR": "char-LR",
       "URLNet-style CNN": "URLNet-style char+word CNN", "BiGRU": "BiGRU character model", "ADAPT-X (E)": "ADAPT-X (E)"}
def op(ds, m, rule):
    r = R4[(R4.dataset == ds) & (R4.model == m) & (R4["threshold rule"] == rule)]
    assert len(r) == 1, (ds, m, rule)
    return r.iloc[0]
rows = []
for rule, title in [("val_fpr@1%", "1"), ("val_fpr@0.1%", "0.1")]:
    if rows: rows.append(r"\midrule")
    rows.append(rf"\multicolumn{{5}}{{l}}{{\textit{{Threshold frozen for a validation macro FPR of {title}\%: TPR / FPR\,\% [exact 95\% CI of the FPR]}}}} \\")
    for nm, m in R4N.items():
        cells = []
        for ds in CORP:
            r = op(ds, m, rule); lo, hi = pci(r["FPR exact95"])
            cells.append(f"{r.TPR:.2f} / {100 * r.FPR:.2f} [{100 * lo:.2f}, {100 * hi:.2f}]")
            fact(f"op|{rule}|{ds}|{nm}", {"tpr": r.TPR, "fpr": r.FPR, "fpr_lo": lo, "fpr_hi": hi, "fp": int(r["FP count"]), "n_neg": int(r.n_neg),
                                          "prec@1%": r["precision @ prevalence 0.01"], "ratio": r["achieved FPR / target"]})
        rows.append(f"{nm} & " + " & ".join(cells) + r" \\")
nneg = int(op("D4v", "ADAPT-X (P)", "val_mcc").n_neg); fact("d4v_nneg", nneg)
write("deploy", tab("table*", "tab:deploy", "Operating points under domain shift.", "lrrrr", [r"Model & PhreshTest & Crawling2024 & CompPhish & D4v \\"], rows,
      rf"Clopper--Pearson intervals; registrable-domain cluster-bootstrap intervals, FP counts and the FPR resolution $1/n_{{\mathrm{{neg}}}}$ are released with the results (R4). With {nneg} D4v negatives one false positive is {100 / nneg:.2f}\%, so an FPR target of 0.1\% cannot be verified on D4v.", sep="2.5pt"))

# ---------------- recalibration ----------------
R6 = rd("revision/R6_target_domain_recalibration.csv"); R6 = R6[R6["target FPR"] == 0.01]
RN = {"ADAPT-X (P)": "ADAPT-X (P)", "ADAPT-Net (P)": "ADAPT-PHISH (P)", "Char-LR": "char-LR", "BiGRU, tuned": "BiGRU tuned", "ADAPT-X2 (P)": "ADAPT-X2 stack (P)"}
KS = [50, 200, 1000]
rows = []
for ds in ["Crawling2024", "CompPhish", "D4v"]:
    if rows: rows.append(r"\midrule")
    first = True
    for nm, m in RN.items():
        g = R6[(R6.dataset == ds) & (R6.model == m)]
        fz = g[g.threshold.str.startswith("frozen")].iloc[0]
        cells = [f"{100 * fz.FPR_mean:.2f} / {fz.TPR_mean:.2f}"]
        fact(f"recal|{ds}|{nm}|frozen", {"fpr": fz.FPR_mean, "tpr": fz.TPR_mean})
        for k in KS:
            x = g[(g.k == k) & g.threshold.str.startswith("recal")]
            if not len(x) or not x.iloc[0].feasible: cells.append("--"); continue
            x = x.iloc[0]; cells.append(f"{100 * x.FPR_mean:.2f} [{100 * x.FPR_p2_5:.2f}, {100 * x.FPR_p97_5:.2f}] / {x.TPR_mean:.2f}")
            fact(f"recal|{ds}|{nm}|k{k}", {"fpr": x.FPR_mean, "lo": x.FPR_p2_5, "hi": x.FPR_p97_5, "tpr": x.TPR_mean, "share_ok": x[R6.columns[-1]]})
        rows.append(f"{ds if first else ''} & {nm} & " + " & ".join(cells) + r" \\"); first = False
write("recal", tab("table*", "tab:recal", r"Recalibrating the 1\% FPR threshold on $k$ benign URLs of the target corpus.", "llrrrr",
      [r"Corpus & Model & Frozen & $k$ = 50 & $k$ = 200 & $k$ = 1000 \\"], rows,
      r"FPR\,\% [2.5th, 97.5th percentile over 50 random draws of the $k$ calibration URLs] / mean TPR, measured on the remaining target rows. The threshold is the conformal $(1-\alpha)$ quantile of the $k$ benign scores; --: infeasible ($k$ too small for the target, or fewer than $k$ benign rows left). Values for $k$ = 100, 500, 2000, for the 0.1\% target and for the other models are released (R6)."))
TT = rd("revision/R6_target_temperature_refit.csv")
fact("temp_refit", {f"{r.dataset}|{r.model}|{r.k}": {"ece0": r["ECE frozen"], "ece1": r["ECE re-fitted T"], "T": r["T'"]} for _, r in TT.iterrows()})

# ---------------- error strata ----------------
R7 = rd("revision/R7_fpr_tpr_by_url_stratum.csv"); R7 = R7[R7["threshold rule"] == "val_mcc"]
STR = ["root URL (no path)", "URL with a path", "has a query string", "contains a suspicious keyword", "risky TLD", "long URL (tercile 3)", "registrable domain unseen in TRAIN"]
SLAB = {"root URL (no path)": "root URL (no path)", "URL with a path": "URL with a path", "has a query string": "has a query string",
        "contains a suspicious keyword": "contains a suspicious keyword", "risky TLD": "risky TLD", "long URL (tercile 3)": "long URL (tercile 3)",
        "registrable domain unseen in TRAIN": "registrable domain unseen in TRAIN"}
rows = []
for s in STR:
    cells = []
    for ds in CORP:
        for m in ["ADAPT-X (P)", "BiGRU character model"]:
            r = R7[(R7.dataset == ds) & (R7.model == m) & (R7.stratum == s)]
            if not len(r) or r.iloc[0]["n benign"] < 10: cells.append("--"); continue
            r = r.iloc[0]; cells.append(f"{100 * r.FPR:.1f}")
            fact(f"err|{ds}|{m}|{s}", {"fpr": r.FPR, "n": int(r["n benign"])})
    rows.append(f"{SLAB[s]} & " + " & ".join(cells) + r" \\")
write("errors", tab("table*", "tab:errors", r"False-positive rate (\%) by URL stratum at the frozen weighted-MCC threshold.", "lrrrrrrrr",
      [r" & \multicolumn{2}{c}{PhreshTest} & \multicolumn{2}{c}{Crawling2024} & \multicolumn{2}{c}{CompPhish} & \multicolumn{2}{c}{D4v} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
       r"Benign stratum & AX & GRU & AX & GRU & AX & GRU & AX & GRU \\"], rows,
      r"AX: ADAPT-X (P); GRU: BiGRU. Strata are computed from the URL only and are not exclusive; --: fewer than 10 benign rows. Long: top tercile of TRAIN URL lengths. Counts per stratum, Wilson intervals, TPR and the remaining strata for all models are released (R7)."))
# shares used in the text
for ds in CORP:
    for m in ["ADAPT-X (P)"]:
        g = R7[(R7.dataset == ds) & (R7.model == m)]
        tot = g[g.stratum.isin(["root URL (no path)", "URL with a path"])]["n benign"].sum()
        root = g[g.stratum == "root URL (no path)"]["n benign"].sum()
        fact(f"rootshare|{ds}", root / tot if tot else None)

# ---------------- PhishTank ----------------
BOOT = rd("reports/bootstrap_results.csv")
PT = ["ADAPT-X (P)", "ADAPT-Net (P)", "LightGBM (P)", "Char-LR", "LightGBM (P) + $n$-grams", "URLNet-style CNN", "BiGRU", "Simple stack (no neural)",
      "ADAPT-X (E)", "ADAPT-Net (E)", "LightGBM (E)"]
rows = []
for m in PT:
    if m == "ADAPT-X (E)": rows.append(r"\midrule")
    cells = []
    for sub in ["PhishTank (purged)", "PhishTankNovel (primary)", "PhishTankNovel ∩ unseen domain"]:
        r = tm(ENS[m], "PhishTank", subset=sub); b = BOOT[(BOOT.model == ENS[m]) & (BOOT.dataset == "PhishTank") & (BOOT.subset == sub) & (BOOT.metric == "recall")]
        cells.append(f3(r.recall) + (f" {ci(b.iloc[0].ci95_low, b.iloc[0].ci95_high)}" if len(b) and "unseen" not in sub else ""))
        fact(f"pt|{sub}|{m}", r.recall)
    rows.append(f"{m} & " + " & ".join(cells) + r" \\")
npt = {s: int(TR[(TR.dataset == "PhishTank") & (TR.subset == s)].n_subset.dropna().iloc[0]) for s in ["PhishTank (purged)", "PhishTankNovel (primary)", "PhishTankNovel ∩ unseen domain"]}
fact("pt_rows", npt)
write("phishtank", tab("table", "tab:phishtank", "Recall on the PhishTank stream.", "lrrr", [r"Model & Purged & Novel & Novel $\cap$ unseen \\"], rows,
      rf"Frozen weighted-MCC threshold; brackets: 95\% registrable-domain cluster-bootstrap interval (original models). Rows: {num(npt['PhishTank (purged)'])} purged; {num(npt['PhishTankNovel (primary)'])} novel; {num(npt['PhishTankNovel ∩ unseen domain'])} novel from unseen registrable domains.", sep="2pt"))

# ---------------- statistics (Holm) ----------------
R11 = rd("revision/R11_paired_tests_holm.csv")
BN = {"ADAPT-PHISH (P) seed-ens": "ADAPT-Net (P)", "LightGBM (P) seed-ens": "LightGBM (P)", "char 3-5-gram LR": "Char-LR", "LightGBM (E) seed-ens": "LightGBM (E)"}
def pf(p): return r"$<$0.001" if p < 0.001 else f"{p:.3f}"
rows = []
for ds in CORP:
    if rows: rows.append(r"\midrule")
    first = True
    for b, nm in BN.items():
        cells = []
        for sub in ["ALL", "unseen"]:
            g = R11[(R11.dataset == ds) & (R11.B == b) & (R11.subset.str.startswith("unseen") if sub == "unseen" else (R11.subset == "ALL"))]
            r = g.iloc[0]; lo, hi = pci(r.delta_roc_auc_ci95)
            cells += [sg(r[R11.columns[4]], 4), f"[{sg(lo)}, {sg(hi)}]", pf(r["p Holm"])]
            fact(f"stats|{ds}|{sub}|{nm}", {"d": r[R11.columns[4]], "lo": lo, "hi": hi, "p_holm": r["p Holm"], "mcn": r[R11.columns[9]], "mcn_p": r.mcnemar_p})
        rows.append(f"{ds if first else ''} & {nm} & " + " & ".join(cells) + r" \\"); first = False
write("stats", tab("table*", "tab:stats", r"Paired ROC-AUC differences of ADAPT-X (P) to the pre-specified baselines (A $-$ B).", "llrrrrrr",
      [r" & & \multicolumn{3}{c}{All rows} & \multicolumn{3}{c}{Unseen registrable domains} \\ \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
       r"Corpus & Baseline B & $\Delta$ROC-AUC & 95\% CI & $p_{\mathrm{Holm}}$ & $\Delta$ROC-AUC & 95\% CI & $p_{\mathrm{Holm}}$ \\"], rows,
      rf"Paired registrable-domain cluster bootstrap ($B$ = 2000); two-sided $p$ adjusted with Holm's method over all {len(R11)} comparisons of this table. Contingency counts and McNemar tests at the frozen threshold, with Holm and Benjamini--Hochberg adjusted $p$, are released (R11). The post hoc models are compared in \tabref{{tab:v14dims}} with a separate Holm family."))

# ---------------- cost ----------------
R10 = rd("revision/R10_computational_cost.csv").set_index("model"); TT = rd("revision/R10_training_time.csv")
def trmin(prefix):
    g = TT[TT.model.str.startswith(prefix) & TT.model.str.contains(r" s\d+$") & ~TT.model.str.contains("2x")]
    return g["train seconds"].mean() / 60 if len(g) else np.nan
CN = [("ADAPT-X (P)", "ADAPT-X (P)", None), ("ADAPT-Net (P)", "ADAPT-PHISH (P)", "ADAPT-PHISH (P) s"), ("LightGBM (P)", "LightGBM (P)", None),
      ("Char-LR", "char-LR", None), ("ADAPT-X (E)", "ADAPT-X (E)", None), ("ADAPT-Net (E)", "ADAPT-PHISH (E)", None), ("LightGBM (E)", "LightGBM (E)", None),
      ("LightGBM (P) + $n$-grams", "LightGBM (P) + hashed char n-grams", "LightGBM (P) + hashed char n-grams s"),
      ("URLNet-style CNN", "URLNet-style char+word CNN", "URLNet-style char+word CNN s"), ("BiGRU", "BiGRU character model", "BiGRU character model s"),
      ("Simple stack (no neural)", "Simple stack (P): LightGBM + char-LR, no neural", None)]
rows = []
for nm, m, pre in CN:
    r = R10.loc[m]; tmn = trmin(pre) if pre else np.nan
    p = int(r[R10.columns[0]]); tr = r.trees
    fact(f"cost|{nm}", {"params": p, "trees": None if np.isnan(tr) else int(tr), "ms": r[R10.columns[2]], "ms1": r[R10.columns[3]], "train_min": tmn})
    rows.append(f"{nm} & {num(p) if p else '--'} & {'--' if np.isnan(tr) else num(tr)} & {r[R10.columns[2]]:.3f} & {r[R10.columns[3]]:.1f} & {'--' if np.isnan(tmn) else f'{tmn:.1f}'} \\\\")
write("cost", tab("table", "tab:cost", "Model size, inference latency and training time.", "lrrrrr",
      [r"Model & Param. & Trees & ms/URL & 1-URL ms & Train min \\"], rows,
      r"Parameters of all seed members together (five seeds; the (P) and (E) passes share one network); trees of all LightGBM members. ms/URL: batches of 2000 URLs; 1-URL: median latency of single requests, including feature extraction. Train min: mean wall-clock minutes per seed. Measured in the final run on one NVIDIA Quadro P6000 GPU (neural models) and an Intel Xeon E5-2609 v4 CPU (trees, linear models, feature extraction); absolute values depend on load, so only ratios are interpreted.", sep="2pt"))
dump_facts(); print("ok")
