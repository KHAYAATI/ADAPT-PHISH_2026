from lib import *

BOOT = rd("reports/bootstrap_results.csv")
def bci(model, ds, subset="ALL", metric="roc_auc"):
    r = BOOT[(BOOT.model == model) & (BOOT.dataset == ds) & (BOOT.subset == subset) & (BOOT.metric == metric)]
    return (r.iloc[0].ci95_low, r.iloc[0].ci95_high) if len(r) else None

# ---------------- Table: held-out corpora (pre-specified models) ----------------
P_ORDER = ["ADAPT-X (P)", "ADAPT-Net (P)", "LightGBM (P)", "Char-LR", "LightGBM (P) + $n$-grams", "URLNet-style CNN", "BiGRU",
           "Simple stack (no neural)", "ADAPT-X (E)", "ADAPT-Net (E)", "LightGBM (E)"]
L = [r"\begin{table*}[!t]", r"\centering", r"\caption{Held-out corpora at the frozen weighted-MCC threshold (pre-specified models).}", r"\label{tab:test}",
     r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrrrrrrr}", r"\toprule",
     r" & Model & ROC-AUC [95\% CI] & PR-AUC & F1 & MCC & Recall & FPR & ECE \\"]
for ds in CORP:
    n = int(tm(ENS["ADAPT-X (P)"], ds).n); fact(f"n_{ds}", n)
    L += [r"\midrule", rf"\multicolumn{{9}}{{l}}{{\textit{{{ds}}} ($n$ = {num(n)})}} \\"]
    for m in P_ORDER:
        r = tm(ENS[m], ds); c = bci(ENS[m], ds)
        auc = f3(r.roc_auc) + (" " + ci(*c) if c else "")
        for k in ["roc_auc", "pr_auc", "f1", "mcc", "recall", "fpr", "ece"]: fact(f"test|{ds}|{m}|{k}", r[k])
        if c: fact(f"test|{ds}|{m}|roc_ci", c)
        L.append(f" & {m} & {auc} & {f3(r.pr_auc)} & {f3(r.f1)} & {f3(r.mcc)} & {f3(r.recall)} & {f3(r.fpr)} & {f3(r.ece)} \\\\")
L += [r"\bottomrule", r"\end{tabular}",
      r"\par\smallskip\parbox{0.97\textwidth}{\scriptsize Thresholds, temperatures and stacking weights fixed on VALIDATION. Brackets: 95\% registrable-domain cluster-bootstrap interval ($B$ = 2000), computed for the seven models of the original comparison. On external corpora the native features are absent, so the (E) and (P) passes of ADAPT-Net receive the same inputs and give identical scores; their F1, MCC, recall and FPR differ only through the thresholds fitted on VALIDATION, where the (E) pass does use native features (Appendix~\ref{sec:appB}). The three one-variable sanity baselines are reported in Appendix~\ref{sec:appB}.}",
      r"\end{table*}"]
write("test", "\n".join(L) + "\n")

# ---------------- Table: macro external with / without Crawling2024 ----------------
R3 = rd("revision/R3_macro_auc_bootstrap_with_without_crawling2024.csv")
R3m = rd("revision/R3_external_macro_with_without_crawling2024.csv")
R3N = {"ADAPT-X (P)": "ADAPT-X (P)", "ADAPT-Net (P)": "ADAPT-PHISH (P)", "LightGBM (P)": "LightGBM (P)", "Char-LR": "char-LR",
       "LightGBM (P) + $n$-grams": "LightGBM (P) + hashed char n-grams", "URLNet-style CNN": "URLNet-style char+word CNN",
       "BiGRU": "BiGRU character model", "Simple stack (no neural)": "Simple stack (P): LightGBM + char-LR, no neural",
       "ADAPT-X (E)": "ADAPT-X (E)", "ADAPT-Net (E)": "ADAPT-PHISH (E)", "LightGBM (E)": "LightGBM (E)", "URL length only": "Sanity: url_len only (LR)"}
def pfmt(p):
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"
L = [r"\begin{table*}[!t]", r"\centering", r"\caption{Macro external results with and without Crawling2024.}", r"\label{tab:macro}",
     r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
     r" & \multicolumn{4}{c}{Crawling2024, CompPhish, D4v} & \multicolumn{4}{c}{CompPhish, D4v} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
     r"Model & ROC-AUC [95\% CI] & $\Delta$ ($p$) & MCC & FPR & ROC-AUC [95\% CI] & $\Delta$ ($p$) & MCC & FPR \\", r"\midrule"]
for i, m in enumerate(list(R3N)):
    if m in ("ADAPT-X (E)", "URL length only"): L.append(r"\midrule")
    cells = []
    for st, suf in [("3 ext.", "3 ext."), ("2 ext. (no Crawling2024)", "2 ext. (no Crawling2024)")]:
        r = R3[(R3.set == st) & (R3.model == R3N[m])].iloc[0]; lo, hi = pci(r.ci95)
        mm = R3m[(R3m.model == R3N[m]) & (R3m["threshold rule"] == "val_mcc")].iloc[0]
        d = "--" if np.isnan(r["Δ vs ADAPT-X (P)"]) else f"{sg(-r['Δ vs ADAPT-X (P)'])} ({pfmt(r['p (two-sided)'])})"
        cells += [f"{f3(r['macro ROC-AUC'])} {ci(lo, hi)}", d, f3(mm[f'mcc | {suf}']), f3(mm[f'fpr | {suf}'])]
        tag = "ext3" if st.startswith("3") else "ext2"
        fact(f"macro|{tag}|{m}|auc", r["macro ROC-AUC"]); fact(f"macro|{tag}|{m}|ci", (lo, hi))
        fact(f"macro|{tag}|{m}|delta_vs_AX", None if np.isnan(r['Δ vs ADAPT-X (P)']) else -r['Δ vs ADAPT-X (P)'])
        fact(f"macro|{tag}|{m}|p", None if np.isnan(r['p (two-sided)']) else r['p (two-sided)'])
        fact(f"macro|{tag}|{m}|mcc", mm[f'mcc | {suf}']); fact(f"macro|{tag}|{m}|fpr", mm[f'fpr | {suf}'])
    L.append(f"{m} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}",
      r"\par\smallskip\parbox{0.97\textwidth}{\scriptsize Unweighted means over corpora. Intervals: registrable-domain cluster bootstrap ($B$ = 2000) of the macro ROC-AUC; $\Delta$: model minus ADAPT-X (P), paired on the same resamples, with its two-sided bootstrap $p$ (unadjusted). MCC and FPR at the weighted-MCC threshold frozen on VALIDATION.}",
      r"\end{table*}"]
write("macro", "\n".join(L) + "\n")
dump_facts()
print(open(OUT / "macro.tex").read()[:3000])
