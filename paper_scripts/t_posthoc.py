from lib import *
from t_more import tab

R15T = rd("revision/R15_v14_test_metrics.csv")
TN = [("ADAPT-X2 (P)", "ADAPT-X2 stack (P)"), ("Baseline stack (no ADAPT)", "Baseline stack (P): tuned baselines, no ADAPT"),
      ("ADAPT-Net v2, tuned (P)", "ADAPT-PHISH v2 (P) seed-ens"), ("ADAPT-Net v2-KD (P)", "ADAPT-PHISH v2-KD (P) seed-ens"),
      ("BiGRU, tuned", "BiGRU tuned seed-ens"), ("BiGRU-KD, tuned", "BiGRU-KD tuned seed-ens"), ("URLNet-style, tuned", "URLNet tuned seed-ens"),
      ("LightGBM + $n$-grams, tuned", "LightGBM + n-grams tuned seed-ens"), ("Char-LR, tuned", "char n-gram LR tuned"),
      ("LightGBM (P), tuned", "LightGBM (P) tuned seed-ens"), ("CANINE-s, fine-tuned", "Pretrained google/canine-s (fine-tuned)"),
      ("BERT-mini, 200k rows/epoch", "Pretrained google/bert_uncased_L-4_H-256_A-4 (fine-tuned)"), ("ADAPT-X (P)", "ADAPT-X stack (P)"),
      ("ADAPT-Net (P)", "ADAPT-PHISH (P) seed-ens")]

# ---------------- tuning ----------------
SC = rd("revision/R15_v14_selected_configs.csv"); TT = rd("reports/v14_tuning_trials.csv")
FN = {"ADAPT-PHISH v2": "ADAPT-Net v2", "BiGRU": "BiGRU", "URLNet-style": "URLNet-style CNN", "LightGBM (P)": "LightGBM (P)",
      "LightGBM (P) + hashed n-grams": "LightGBM + $n$-grams", "char n-gram LR": "Char-LR"}
rows = []
for _, r in SC.iterrows():
    nt = int((TT.family == r.family).sum())
    d = r["selected score"] - r["score of the v13 default (trial 0)"]
    fact(f"tune|{FN[r.family]}", {"trials": nt, "sel": int(r["selected trial"]), "default": r["score of the v13 default (trial 0)"], "selected": r["selected score"], "delta": d,
                                  "config": r["selected configuration"], "worst_auc": r[SC.columns[3]]})
    rows.append(f"{FN[r.family]} & {nt} & {int(r['selected trial'])} & {r['score of the v13 default (trial 0)']:.4f} & {r['selected score']:.4f} & {d:+.4f} \\\\")
fam_trials = TT.groupby("family").size().to_dict(); fact("tune_trials", fam_trials)
write("v14_tuning", tab("table", "tab:v14tuning", "Equal-budget hyper-parameter search (post hoc round): selection score on VALIDATION.", "lrrrrr",
      [r"Family & Trials & Selected & Default & Selected score & $\Delta$ \\"], rows,
      r"Score = worst development-corpus ROC-AUC on VALIDATION (+$10^{-3}\times$macro PR-AUC), identical for every family; trial 0 is the configuration of the pre-specified comparison. Neural trials: 8 epochs of 120,000 rows, seed 42; tree and linear trials are full fits. Search spaces and every trial are released (R15).", sep="2.5pt"))

# ---------------- test ROC/PR of the post hoc models ----------------
rows = []
for nm, m in TN:
    if nm == "ADAPT-X (P)": rows.append(r"\midrule")
    a = [tm(m, ds).roc_auc for ds in CORP]; p = [tm(m, ds).pr_auc for ds in CORP]; e2 = (a[2] + a[3]) / 2
    for ds, x, y in zip(CORP, a, p): fact(f"ph|{ds}|{nm}|roc", x); fact(f"ph|{ds}|{nm}|pr", y)
    fact(f"ph|ext2|{nm}", e2); fact(f"ph|ext3|{nm}", (a[1] + a[2] + a[3]) / 3)
    rows.append(f"{nm} & " + " & ".join(f3(x) for x in a) + f" & {f3(e2)} & " + " & ".join(f3(x) for x in p) + r" \\")
    if nm.startswith("BERT-mini, 200k"):
        R19 = rd("revision/R19_urltran_style_bert.csv"); g = R19[R19.model.str.startswith("URLTran-style") & (R19["threshold rule"] == "val_mcc")].set_index("dataset")
        a = [g.loc[ds, "roc_auc"] for ds in CORP]; p = [g.loc[ds, "pr_auc"] for ds in CORP]; e2 = (a[2] + a[3]) / 2
        for ds, x in zip(CORP, a): fact(f"ph|{ds}|BERT-mini URLTran (R19)|roc", x)
        fact("ph|ext2|BERT-mini URLTran (R19)", e2)
        rows.append(f"BERT-mini, URLTran-style (all TRAIN) & " + " & ".join(f3(x) for x in a) + f" & {f3(e2)} & " + " & ".join(f3(x) for x in p) + r" \\")
write("v14_test", tab("table*", "tab:v14test", "Post hoc round: ROC-AUC and PR-AUC of the tuned and pretrained models on the held-out corpora (all rows).", "lrrrrr|rrrr",
      [r"& \multicolumn{5}{c|}{ROC-AUC} & \multicolumn{4}{c}{PR-AUC} \\", r"Model & PhreshTest & Crawl. & CompPh. & D4v & Ext.$_2$ & PhreshTest & Crawl. & CompPh. & D4v \\"], rows,
      r"Ext.$_2$: mean over the two documented external corpora (CompPhish, D4v). Five-seed ensembles except char-LR (deterministic) and the pretrained models (one seed each): CANINE-s and BERT-mini fine-tuned for three epochs of 200,000 TRAIN rows, and the URLTran-style BERT-mini of \secref{sec:res_v14} fine-tuned for three epochs over all TRAIN rows (maximum length 128 for all three). KD: distilled from ADAPT-X2 on TRAIN (soft targets, weight 0.5), with the tuned BiGRU distilled by the same rule as control. ADAPT-X (P) and ADAPT-Net (P): pre-specified models, for reference."))

# ---------------- dimension counts ----------------
DC = rd("revision/R15_v14_dimension_counts.csv")
RN = {v: k for k, v in [(a, b.replace(" seed-ens", "")) for a, b in TN]}
RN.update({"ADAPT-PHISH v2 (P)": "ADAPT-Net v2, tuned (P)", "ADAPT-PHISH v2-KD (P)": "ADAPT-Net v2-KD (P)", "ADAPT-PHISH (P)": "ADAPT-Net (P)", "char-LR": "Char-LR",
           "LightGBM (P) + hashed char n-grams": "LightGBM (P) + $n$-grams", "URLNet-style char+word CNN": "URLNet-style CNN", "BiGRU character model": "BiGRU",
           "Simple stack (P): LightGBM + char-LR, no neural": "Simple stack (no neural)", "LightGBM (P)": "LightGBM (P)"})
rows = []
for _, r in DC.iterrows():
    a, b = RN.get(r.A, r.A), RN.get(r.B, r.B)
    fact(f"dims|{a}|{b}", {"better": int(r["A better"]), "worse": int(r["A worse"]), "tie": int(r.tie), "sig_plus": int(r["AUC cells significantly better (Holm)"]), "sig_minus": int(r["AUC cells significantly worse (Holm)"]), "cells": int(r["A better"] + r["A worse"] + r.tie)})
    if (r.A == "ADAPT-PHISH v2 (P)" and r.name == DC[DC.A == "ADAPT-PHISH v2 (P)"].index[0]) or (r.A == "ADAPT-PHISH v2-KD (P)" and r.name == DC[DC.A == "ADAPT-PHISH v2-KD (P)"].index[0]):
        rows.append(r"\midrule")
    rows.append(f"{a} & {b} & {int(r['A better'])} & {int(r['A worse'])} & {int(r.tie)} & {int(r["AUC cells significantly better (Holm)"])} & {int(r["AUC cells significantly worse (Holm)"])} \\\\")
P15 = rd("revision/R15_v14_paired_tests.csv"); fact("dims_holm_family", len(P15))
write("v14_dims", tab("table*", "tab:v14dims", "Post hoc round: number of comparison cells in which A is better or worse than B.", "llrrrrr",
      [r"A & B & Better & Worse & Tie & Sig.+ & Sig.$-$ \\"], rows,
      rf"Cells: 4 test corpora $\times$ 10 metric/threshold combinations (ROC-AUC, PR-AUC, MCC at two rules, FPR and recall at two 1\% rules, ECE, Brier) + PhishTankNovel recall at two rules = 42; tie: $|\Delta|\le10^{{-4}}$. Sig.: ROC-AUC and PR-AUC cells (8) whose paired domain-cluster bootstrap difference is significant after Holm adjustment over all {len(P15)} comparisons of the round.", sep="2.5pt"))

# ---------------- operating points of the post hoc models ----------------
rows = []
for nm, m in TN:
    if nm == "ADAPT-X (P)": rows.append(r"\midrule")
    cs = []
    for ds in CORP:
        r = tm(m, ds, rule="val_fpr@1%"); cs += [f"{100 * r.fpr:.1f}", f3(r.recall)]
        fact(f"phop|{ds}|{nm}", {"fpr": r.fpr, "rec": r.recall})
    pt = tm(m, "PhishTank", rule="val_fpr@1%", subset="PhishTankNovel (primary)").recall; fact(f"phop|PT|{nm}", pt)
    rows.append(f"{nm} & " + " & ".join(cs) + f" & {f3(pt)} \\\\")
write("v14_op", tab("table*", "tab:v14op", r"Post hoc round: FPR (\%) and recall at the threshold frozen for a 1\% macro VALIDATION FPR.", "lrrrrrrrr|r",
      [r"& \multicolumn{2}{c}{PhreshTest} & \multicolumn{2}{c}{Crawling2024} & \multicolumn{2}{c}{CompPhish} & \multicolumn{2}{c|}{D4v} & PhishTank \\",
       r"Model & FPR & Rec. & FPR & Rec. & FPR & Rec. & FPR & Rec. & Novel rec. \\"], rows,
      r"Thresholds from VALIDATION only, unchanged on every test set. PhishTank is phishing-only (recall on PhishTankNovel). Higher recall at a higher FPR is a different operating point, not a uniformly better detector; \tabref{tab:v14dims} compares rankings."))
# worst-corpus rules
for nm, m in TN[:6]:
    for ds in ["Crawling2024"]:
        fact(f"phop_rules|{ds}|{nm}", {rule: tm(m, ds, rule=rule).fpr for rule in ["val_fpr@1%", "val_maxfpr@1%"]})

# ---------------- robustness ----------------
R16 = rd("revision/R16_obfuscation_idn_robustness.csv"); R16 = R16[R16["threshold rule"] == "val_mcc"]
PERT = [("IDN", "IDN homoglyph host (punycode)"), (r"\%-enc.", "percent-encoded path characters"), ("Query", "extra query parameters"), ("Case", "random case in path"),
        ("@-prefix", "userinfo prefix (@ trick)"), ("Host only", "host only (path removed)"), ("Adaptive", "ADAPTIVE: per-model worst of 5 page-preserving rewrites (phishing only)")]
RM = [("ADAPT-X2 (P)", "ADAPT-X2 stack (P)"), ("Baseline stack (no ADAPT)", "Baseline stack (P): tuned baselines, no ADAPT"), ("ADAPT-Net v2, tuned (P)", "ADAPT-PHISH v2 (P)"),
      ("ADAPT-Net v2-KD (P)", "ADAPT-PHISH v2-KD (P)"), ("BiGRU, tuned", "BiGRU tuned"), ("URLNet-style, tuned", "URLNet tuned"), ("LightGBM + $n$-grams, tuned", "LightGBM + n-grams tuned"),
      ("Char-LR, tuned", "char n-gram LR tuned"), ("BERT-mini, 200k rows/epoch", "Pretrained google/bert_uncased_L-4_H-256_A-4 (fine-tuned)"), ("ADAPT-X (P)", "ADAPT-X (P)")]
rows = []
for nm, m in RM:
    cs = []
    for lab, p in PERT:
        g = R16[(R16.model == m) & (R16.perturbation == p)]
        v = g["Δroc_auc vs unperturbed"].mean(); cs.append(sg(v)); fact(f"rob|{nm}|{lab}", v)
    rows.append(f"{nm} & " + " & ".join(cs) + r" \\")
ctl = R16[R16.perturbation == "none (pipeline re-run)"]
fact("rob_control_max", ctl[~ctl.model.str.contains("canine")]["Δroc_auc vs unperturbed"].abs().max())
fact("rob_share_adaptive", R16[R16.perturbation.str.startswith("ADAPTIVE")]["share of URLs changed"].mean())
write("v14_robust", tab("table*", "tab:v14robust", "Robustness to URL rewrites: change of ROC-AUC with respect to the unperturbed URLs (mean over PhreshTest, CompPhish, Crawling2024).", "lrrrrrrr",
      [r"Model & IDN & \%-enc. & Query & Case & @-prefix & Host only & Adaptive \\"], rows,
      r"8,000 rows per corpus (balanced), rewritten deterministically and re-processed by the frozen pipeline. IDN: two letters of the registrable label replaced by Cyrillic homoglyphs (punycode); \%-enc.: 30\% of path characters percent-encoded; Query: three neutral parameters appended; Case: random case in the path; @-prefix: a well-known host as userinfo; Host only: path removed; these apply to both classes. Adaptive: for every phishing URL and every model separately, the rewrite among five page-preserving ones (all but host-only) that lowers that model's score most. Re-running the pipeline on unchanged URLs reproduces the stored scores within $2\times10^{-4}$ for every listed model; CANINE-s did not pass this control (differences up to 0.5) and is excluded. Changes of recall and FPR at the frozen thresholds are released (R16).", sep="2.5pt"))

# ---------------- LOCO with neural models ----------------
R17 = rd("revision/R17_loco_all_corpora_with_neural.csv"); R17 = R17[R17["threshold rule"] == "val_mcc"]
LM = [("ADAPT-Net v2 (P)", "ADAPT-PHISH v2 (P)"), ("ADAPT-Net, pre-specified config.", "ADAPT-PHISH v13 config (P)"), ("BiGRU, tuned", "BiGRU tuned"),
      ("URLNet-style, tuned", "URLNet tuned"), ("Neural stack", "neural stack (ADAPT v2 + BiGRU + URLNet)"), ("Char-LR", "char 3-5-gram LR"),
      ("LightGBM (P)", "LightGBM (P)"), ("URL length only", "Sanity: url_len only (LR)")]
HO = ["Mendeley", "PhreshPhish", "Crawling2024", "CompPhish", "D4v"]
rows = []
for nm, m in LM:
    v = [R17[(R17.model == m) & (R17["held-out corpus"] == h)].roc_auc.values[0] for h in HO]
    for h, x in zip(HO, v): fact(f"loco5|{nm}|{h}", x)
    fact(f"loco5|{nm}|mean", float(np.mean(v)))
    rows.append(f"{nm} & " + " & ".join(f3(x) for x in v) + f" & {f3(np.mean(v))} \\\\")
write("v14_loco", tab("table", "tab:v14loco", "Leave-one-corpus-out ROC-AUC over all labelled corpora, with the neural models.", "lrrrrrr",
      [r"Model & Mend. & Phresh. & Crawl. & CompPh. & D4v & Mean \\"], rows,
      r"Each column: trained on all other labelled corpora (purged of the held-out URLs), threshold and temperature from a domain-grouped internal validation split. One seed; neural models with the leave-one-out budget (25 epochs over all training rows, no early stopping). Neural stack: logistic meta-learner on the internal validation split.", sep="2pt"))

# ---------------- budget ----------------
rows = []
for nm, m in [("ADAPT-Net v2, 1$\\times$", "ADAPT-PHISH v2 (P) s42"), ("ADAPT-Net v2, 2$\\times$", "ADAPT-PHISH v2 (P) 2x budget s42"),
              ("BiGRU, tuned, 1$\\times$", "BiGRU tuned s42"), ("BiGRU, tuned, 2$\\times$", "BiGRU tuned 2x budget s42")]:
    v = [tm(m, ds).roc_auc for ds in CORP]
    for ds, x in zip(CORP, v): fact(f"budget|{nm}|{ds}", x)
    rows.append(f"{nm} & " + " & ".join(f3(x) for x in v) + r" \\")
TTm = rd("revision/R10_training_time.csv").set_index("model")
fact("budget_epochs", {k: TTm.loc[k, "epochs"] for k in ["ADAPT-PHISH v2 s42", "ADAPT-PHISH v2 (P) 2x budget s42", "BiGRU tuned s42", "BiGRU tuned 2x budget s42"]})
write("v14_budget", tab("table", "tab:v14budget", "Budget sensitivity (seed 42): ROC-AUC with all TRAIN rows versus twice as many rows per epoch.", "lrrrr",
      [r"Model & PhreshTest & Crawl. & CompPh. & D4v \\"], rows, r"Same configuration, selection rule and early stopping; only the rows per epoch differ (2$\times$: TRAIN sampled twice per epoch)."))

# ---------------- cost of the post hoc models ----------------
R10 = rd("revision/R10_computational_cost.csv").set_index("model")
CM = [("ADAPT-X2 (P)", "ADAPT-X2 stack (P)"), ("Baseline stack (no ADAPT)", "Baseline stack (P): tuned baselines, no ADAPT"), ("ADAPT-Net v2, tuned (P)", "ADAPT-PHISH v2 (P)"),
      ("BiGRU, tuned", "BiGRU tuned"), ("URLNet-style, tuned", "URLNet tuned"), ("LightGBM + $n$-grams, tuned", "LightGBM + n-grams tuned"), ("Char-LR, tuned", "char n-gram LR tuned"),
      ("CANINE-s, fine-tuned", "Pretrained google/canine-s (fine-tuned)"), ("BERT-mini, 200k rows/epoch", "Pretrained google/bert_uncased_L-4_H-256_A-4 (fine-tuned)")]
rows = []
for nm, m in CM:
    r = R10.loc[m]; p = int(r[R10.columns[0]]); tr = r.trees
    fact(f"cost2|{nm}", {"params": p, "trees": None if np.isnan(tr) else int(tr), "ms": r[R10.columns[2]], "ms1": r[R10.columns[3]]})
    rows.append(f"{nm} & {num(p) if p else '--'} & {'--' if np.isnan(tr) else num(tr)} & {r[R10.columns[2]]:.3f} & {r[R10.columns[3]]:.1f} \\\\")
for k in ["Pretrained google/canine-s (fine-tuned)", "Pretrained google/bert_uncased_L-4_H-256_A-4 (fine-tuned)", "ADAPT-PHISH v2 s42", "BiGRU tuned s42"]:
    fact(f"trainmin|{k}", TTm.loc[k, "train seconds"] / 60)
write("v14_cost", tab("table", "tab:v14cost", "Size and latency of the post hoc models.", "lrrrr", [r"Model & Param. & Trees & ms/URL & 1-URL ms \\"], rows,
      r"As \tabref{tab:cost}: all seed members together; same hardware.", sep="2pt"))

# ---------------- temporal evaluation: R18b and R22 ----------------
R18b = rd("revision/R18b_temporal_split_phreshphish.csv"); R18b = R18b[R18b["threshold rule"] == "val_mcc"]
R22 = rd("revision/R22_official_phreshphish_test_split.csv")
TM = ["ADAPT-Net (P)", "BiGRU, tuned", "URLNet, tuned", "LightGBM (P), tuned", "Char-LR, tuned", "Stack of all models"]
SETS = [("early period, held-out domains (random)", "Early, random"), ("later period, all rows", "Later, all"), ("later period, domains unseen before", "Later, unseen")]
rows = []
for m in TM:
    cs = []
    for key, _ in SETS:
        r = R18b[(R18b.model == m) & (R18b["test set"] == key)].iloc[0]; cs += [f3(r.roc_auc), f3(r.mcc)]
        fact(f"r18b|{key}|{m}", {"roc": r.roc_auc, "mcc": r.mcc, "rows": int(r.rows), "recall": r.recall, "fpr": r.fpr})
    rows.append(f"{m} & " + " & ".join(cs) + r" \\")
meta = R18b.iloc[0] if False else rd("revision/R18b_temporal_split_phreshphish.csv").iloc[0]
fact("r18b_meta", meta.to_dict())
write("temporal", tab("table*", "tab:temporal", "Forward-in-time evaluation on PhreshPhish: date-based re-split of the training release (ROC-AUC / MCC).", "lrrrrrr",
      [r" & \multicolumn{2}{c}{Early, random domains} & \multicolumn{2}{c}{Later, all rows} & \multicolumn{2}{c}{Later, unseen domains} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
       r"Model & ROC-AUC & MCC & ROC-AUC & MCC & ROC-AUC & MCC \\"], rows,
      r"PhreshPhish rows dated with the official release dates; models re-trained on the oldest 70\% of the dates (one seed, tuned configurations), thresholds from a domain-grouped early validation split; ``Later'': all rows after the cut-off date, purged of every training URL and model input; ``Early, random'': held-out domains from the training period, the usual random-domain test. Stack: logistic meta-learner on the early validation split. The neural models of this table are re-trained for this experiment and differ from the main models.", sep="3pt"))

rows = []
R22v = R22[(R22["threshold rule"] == "val_fpr@1%") & R22.roc_auc.notna()]
OM = [("ADAPT-X2 (P)", "ADAPT-X2 stack (P)"), ("ADAPT-X (P)", "ADAPT-X (P)"), ("ADAPT-Net v2, tuned (P)", "ADAPT-PHISH v2 (P)"), ("ADAPT-Net (P)", "ADAPT-PHISH (P)"),
      ("BiGRU, tuned", "BiGRU tuned"), ("URLNet-style, tuned", "URLNet tuned"), ("LightGBM + $n$-grams, tuned", "LightGBM + n-grams tuned"), ("LightGBM (P), tuned", "LightGBM (P) tuned"),
      ("Char-LR, tuned", "char n-gram LR tuned"), ("Baseline stack (no ADAPT)", "Baseline stack (P): tuned baselines, no ADAPT"),
      ("Simple stack (no neural)", "Simple stack (P): LightGBM + char-LR, no neural"), ("BERT-mini, 200k rows/epoch", "Pretrained google/bert_uncased_L-4_H-256_A-4 (fine-tuned)")]
print(sorted(R22v.model.unique()))
for nm, m in OM:
    a = R22v[(R22v.model == m) & R22v["test set"].str.endswith("all rows")]
    u = R22v[(R22v.model == m) & R22v["test set"].str.contains("unseen")]
    if not len(a): print("R22 missing", m); continue
    a = a.iloc[0]; lo, hi = pci(a["roc_auc ci95"]); u = u.iloc[0] if len(u) else None
    fact(f"r22|{nm}", {"roc": a.roc_auc, "lo": lo, "hi": hi, "d_roc": a["Δroc_auc vs PhreshTest"], "recall": a.recall, "fpr": a.fpr, "pt_recall": a["PhreshTest recall"],
                       "pt_fpr": a["PhreshTest fpr"], "roc_unseen": None if u is None else u.roc_auc})
    rows.append(f"{nm} & {f3(a.roc_auc)} {ci(lo, hi)} & {sg(a['Δroc_auc vs PhreshTest'])} & {f3(a.recall)} & {f3(a['PhreshTest recall'])} & {100 * a.fpr:.2f} & {'--' if u is None else f3(u.roc_auc)} \\\\")
m0 = R22[R22.roc_auc.isna()].iloc[0]; fact("r22_meta", m0.to_dict())
PROV = rd("revision/R22_phreshphish_split_provenance.csv"); fact("r22_prov", PROV.to_dict("records"))
CTL = rd("revision/R22_constructor_control.csv"); fact("r22_control_max", CTL[CTL.columns[1]].max())
write("official", tab("table*", "tab:official", "Official PhreshPhish test split (dated after every training row), frozen models, threshold for a 1\\% VALIDATION FPR.", "lrrrrrr",
      [r"Model & ROC-AUC [95\% CI] & $\Delta$ vs PhreshTest & Recall & Recall PhreshTest & FPR\,\% & ROC-AUC unseen dom. \\"], rows,
      r"Official test split of the pinned PhreshPhish release (commit \texttt{eca603e}), dated 8 September to 16 December 2025, purged of every URL and model input of TRAIN, VALIDATION and the evaluation corpora; URL and label only (the P view is computed from the URL). Brackets: registrable-domain cluster bootstrap. $\Delta$: ROC-AUC minus the PhreshTest value of the same frozen model. Re-building PhreshTest rows with the same code reproduces the stored scores of every model within $3\times10^{-7}$.", sep="2.5pt"))
dump_facts(); print("ok")
