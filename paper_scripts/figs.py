"""Regenerates every result figure of the manuscript from the v15 CSV files (no number typed by hand)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lib import RES, ENS, TR, VAL, CORP, tm, pci
import os; OUT = os.environ.get("ADAPT_PHISH_FIGURES_OUT", "paper/figures") + "/"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
COL = {"ADAPT-X (P)": "#1f4e79", "ADAPT-Net (P)": "#c0392b", "LightGBM (P)": "#2e8b57", "Char-LR": "#d4880f", "BiGRU": "#6a3d9a", "URLNet-style CNN": "#7f7f7f"}
M6 = list(COL)
BOOT = pd.read_csv(RES / "reports/bootstrap_results.csv")

# ---- domain shift ----
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
sets = ["VALIDATION"] + CORP; w = 0.13
for i, m in enumerate(M6):
    v = [VAL[(VAL.model == ENS[m]) & (VAL.corpus == "MACRO (mean of corpora)")].roc_auc.iloc[0]] + [tm(ENS[m], d).roc_auc for d in CORP]
    err = [[0] * 5, [0] * 5]
    for j, d in enumerate(CORP):
        b = BOOT[(BOOT.model == ENS[m]) & (BOOT.dataset == d) & (BOOT.subset == "ALL") & (BOOT.metric == "roc_auc")]
        if len(b): err[0][j + 1] = v[j + 1] - b.ci95_low.iloc[0]; err[1][j + 1] = b.ci95_high.iloc[0] - v[j + 1]
    x = np.arange(5) + (i - 2.5) * w
    ax[0].bar(x, v, w, color=COL[m], label=m, yerr=err, capsize=1.5, error_kw={"lw": 0.6})
    mc = [VAL[(VAL.model == ENS[m]) & (VAL.corpus == "MACRO (mean of corpora)")].mcc.iloc[0]] + [tm(ENS[m], d).mcc for d in CORP]
    ax[1].bar(x, mc, w, color=COL[m])
for a, t, lo in [(ax[0], "(a) ROC-AUC", 0.8), (ax[1], "(b) MCC at the frozen threshold", 0.3)]:
    a.set_xticks(range(5)); a.set_xticklabels(sets, rotation=15); a.set_ylim(lo, 1.0); a.set_title(t); a.grid(axis="y", lw=0.3)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, ncol=6, fontsize=6, loc="lower center", frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT + "domain_shift.pdf"); plt.close(fig)

# ---- seen vs unseen ----
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.2))
for k, d in enumerate(CORP):
    us = "unseen_domain (domain-disjoint)" if d == "PhreshTest" else "unseen_domain"
    for m in M6:
        s, u = tm(ENS[m], d, subset="seen_domain"), tm(ENS[m], d, subset=us)
        ax[k].plot([0, 1], [s.roc_auc, u.roc_auc], "o-", color=COL[m], ms=3, lw=1, label=m)
    ax[k].set_xticks([0, 1]); ax[k].set_xticklabels([f"seen\n(n={int(s.n):,})", f"unseen\n(n={int(u.n):,})"]); ax[k].set_title(d); ax[k].grid(axis="y", lw=0.3)
ax[0].set_ylabel("ROC-AUC"); ax[0].legend(fontsize=5.5, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(OUT + "seen_vs_unseen.pdf"); plt.close(fig)

# ---- threshold transfer ----
R4 = pd.read_csv(RES / "revision/R4_operating_points_with_CIs.csv")
R4N = {"ADAPT-X (P)": "ADAPT-X (P)", "ADAPT-Net (P)": "ADAPT-PHISH (P)", "LightGBM (P)": "LightGBM (P)", "Char-LR": "char-LR", "BiGRU": "BiGRU character model", "URLNet-style CNN": "URLNet-style char+word CNN"}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
for a, rule, tgt in [(ax[0], "val_fpr@1%", 1.0), (ax[1], "val_fpr@0.1%", 0.1)]:
    for i, m in enumerate(M6):
        for j, d in enumerate(CORP):
            r = R4[(R4.dataset == d) & (R4.model == R4N[m]) & (R4["threshold rule"] == rule)].iloc[0]; lo, hi = pci(r["FPR exact95"])
            y = max(100 * r.FPR, 0.005); x = j + (i - 2.5) * 0.12
            a.errorbar(x, y, yerr=[[y - max(100 * lo, 0.005)], [100 * hi - y]], fmt="o", color=COL[m], ms=3, lw=0.7, label=m if j == 0 else None)
    a.axhline(tgt, ls="--", color="k", lw=0.7); a.set_yscale("log"); a.set_xticks(range(4)); a.set_xticklabels(CORP)
    a.set_ylabel("achieved FPR (%)"); a.set_title(f"threshold for a {tgt:g}% VALIDATION FPR"); a.grid(axis="y", lw=0.3, which="both")
ax[0].legend(fontsize=5.5, frameon=False, ncol=2)
fig.tight_layout(); fig.savefig(OUT + "threshold_transfer.pdf"); plt.close(fig)

# ---- recalibration ----
R6 = pd.read_csv(RES / "revision/R6_target_domain_recalibration.csv"); R6 = R6[R6["target FPR"] == 0.01]
RM = {"ADAPT-X (P)": "ADAPT-X (P)", "ADAPT-Net (P)": "ADAPT-PHISH (P)", "Char-LR": "char-LR", "BiGRU, tuned": "BiGRU tuned"}
RC = {"ADAPT-X (P)": COL["ADAPT-X (P)"], "ADAPT-Net (P)": COL["ADAPT-Net (P)"], "Char-LR": COL["Char-LR"], "BiGRU, tuned": COL["BiGRU"]}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
for k, d in enumerate(["Crawling2024", "CompPhish", "D4v"]):
    for nm, m in RM.items():
        g = R6[(R6.dataset == d) & (R6.model == m) & R6.threshold.str.startswith("recal") & (R6.feasible > 0)].sort_values("k")
        f = R6[(R6.dataset == d) & (R6.model == m) & R6.threshold.str.startswith("frozen")]
        ax[k].plot(g.k, 100 * g.FPR_mean, "o-", color=RC[nm], ms=2.5, lw=1, label=nm)
        ax[k].fill_between(g.k, 100 * g.FPR_p2_5, 100 * g.FPR_p97_5, color=RC[nm], alpha=0.12)
        if len(f): ax[k].axhline(100 * f.FPR_mean.iloc[0], ls=":", color=RC[nm], lw=0.8)
    ax[k].axhline(1.0, ls="--", color="k", lw=0.7); ax[k].set_xscale("log"); ax[k].set_title(d); ax[k].set_xlabel("$k$ legitimate target URLs"); ax[k].grid(lw=0.3)
ax[0].set_ylabel("test FPR (%)"); ax[0].legend(fontsize=5.5, frameon=False)
fig.tight_layout(); fig.savefig(OUT + "recalibration.pdf"); plt.close(fig)

# ---- corpus identifiability ----
CI = pd.read_csv(RES / "reports/corpus_identifiability.csv"); CI = CI[CI.features.isin(["all canonical features", "after shortcut filter"])]
cs = [c for c in ["Mendeley", "PhreshPhish", "Crawling2024", "CompPhish"] if c in set(CI.corpus)]
fig, a = plt.subplots(figsize=(3.4, 2.2)); x = np.arange(len(cs))
for i, (ft, col) in enumerate([("all canonical features", "#7f7f7f"), ("after shortcut filter", "#1f4e79")]):
    v = [CI[(CI.features == ft) & (CI.corpus == c)].one_vs_rest_AUC.iloc[0] for c in cs]
    a.bar(x + (i - 0.5) * 0.35, v, 0.35, color=col, label=ft)
a.axhline(0.6, ls="--", color="k", lw=0.7); a.axhline(0.5, ls=":", color="k", lw=0.6); a.set_ylim(0.4, 1.0); a.set_xticks(x); a.set_xticklabels(cs, fontsize=7)
a.set_ylabel("one-vs-rest ROC-AUC (benign)"); a.legend(fontsize=6, frameon=False, loc="upper left"); a.grid(axis="y", lw=0.3)
fig.tight_layout(); fig.savefig(OUT + "corpus_identifiability.pdf"); plt.close(fig)

# ---- ECRAC need statistic (A0 logs) ----
W = pd.read_csv(RES / "reports/ecrac_window_log.csv"); W0 = W[W.variant == "ECRAC (A0 full)"]
TRK = ["A: CompPhish pool", "B-mix: PhishTankNovel + benign", "B: PhishTankNovel", "C: controlled drift"]
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.0), sharex=True)
for k, t in enumerate(TRK):
    n = W0[W0.track == t].N.dropna()
    ax[k].hist(n, bins=np.linspace(0, 0.7, 29), color="#1f4e79"); ax[k].axvline(0.25, ls="--", color="#c0392b", lw=0.8)
    ax[k].set_title(f"{t.split(':')[0]} ({len(n)} windows)", fontsize=7); ax[k].set_xlabel("$N_t$")
ax[0].set_ylabel("windows")
fig.tight_layout(); fig.savefig(OUT + "ecrac_need.pdf"); plt.close(fig)

# ---- PhishTank timeline ----
P = pd.read_csv(RES / "reports/phishtank_recall_per_window.csv"); P = P[(P.subset == "PhishTankNovel (primary)") & (P.threshold_rule == "val_mcc")]
PN = {"ADAPT-X (P)": "ADAPT-X stack (P)", "ADAPT-Net (P)": "ADAPT-PHISH (P) seed-ens", "LightGBM (P)": "LightGBM (P) seed-ens", "Char-LR": "char 3-5-gram LR", "BiGRU": "BiGRU character model seed-ens"}
fig, a = plt.subplots(figsize=(3.4, 2.2))
for nm, m in PN.items():
    g = P[P.model == m].copy(); g["d"] = pd.to_datetime(g.window.str[:10]); g = g.sort_values("d"); g = g[g.n >= 50]
    a.plot(g.d, g.recall, lw=0.7, color=COL[nm], label=nm)
a.set_ylabel("recall (windows $\\geq$ 50 rows)"); a.legend(fontsize=5.5, frameon=False, loc="lower left"); a.grid(lw=0.3); a.tick_params(axis="x", labelsize=6, rotation=20)
fig.tight_layout(); fig.savefig(OUT + "phishtank_timeline.pdf"); plt.close(fig)

# ---- adaptation timeline (mean over seeds) ----
VV = {"Frozen (static)": ("Frozen", "#7f7f7f"), "Naive (periodic fine-tuning = A8)": ("Naive", "#c0392b"), "Drift + Replay": ("Drift + replay", "#d4880f"),
      "ECRAC (A0 full)": ("ECRAC (A0)", "#1f4e79"), "ECRAC-cal (tau_N calibrated on no-drift development windows)": ("ECRAC-cal", "#2e8b57")}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
for a, (t, col, lab) in zip(ax, [("A: CompPhish pool", "target_f1", "CompPhish F1 (Track A)"), ("B-mix: PhishTankNovel + benign", "preq_f1", "prequential F1 (B-mix)"),
                                ("B-mix: PhishTankNovel + benign", "retain_test_macro_f1", "independent retention macro-F1 (B-mix)")]):
    for v, (nm, c) in VV.items():
        g = W[(W.track == t) & (W.variant == v)].groupby("window")[col].mean()
        a.plot(g.index, g.values, lw=1, color=c, label=nm)
    a.set_title(lab, fontsize=7); a.set_xlabel("window"); a.grid(lw=0.3)
ax[0].legend(fontsize=5.5, frameon=False)
fig.tight_layout(); fig.savefig(OUT + "adaptation_timeline.pdf"); plt.close(fig)
print("figures ok")
