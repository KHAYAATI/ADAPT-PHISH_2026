from lib import *
from t_more import tab
from scipy import stats as st

A = rd("reports/ablation_results.csv"); A = A[A.view == "P"]
DS = [("VALIDATION (macro of corpora)", "VAL"), ("PhreshTest", "PhreshTest"), ("Crawling2024", "Crawl.24"), ("CompPhish", "CompPhish"), ("D4v", "D4v")]
ORDER = list(dict.fromkeys(A.ablation))
ref = "Reference (full model, ablation budget)"
def lab(x): return x.replace("−", "--").replace("α", r"$\alpha$").replace("is_https", r"is\_https")
rows = []
R = A[A.ablation == ref]
cs = []
for d, _ in DS:
    v = R[R.dataset == d].roc_auc; cs.append(f"{v.mean():.3f}$\\pm${v.std(ddof=1):.3f}"); fact(f"abl|ref|{d}", {"mean": v.mean(), "sd": v.std(ddof=1), "n": len(v)})
rows.append(f"Reference (full model, ablation budget) & {R.seed.nunique()} & " + " & ".join(cs) + r" \\")
for ab in ORDER:
    if ab == ref: continue
    if ab.startswith("v2:"): rows.append(r"\midrule")
    g = A[A.ablation == ab]; cs = []
    for d, _ in DS:
        x = g[g.dataset == d]["Δroc_auc vs reference"].dropna().values
        if not len(x): cs.append("--"); continue
        m = x.mean(); star = ""
        if len(x) >= 2:
            se = x.std(ddof=1) / np.sqrt(len(x)); h = st.t.ppf(0.975, len(x) - 1) * se
            if se > 0 and (m - h > 0 or m + h < 0): star = "$^{*}$"
        cs.append(sg(m) + star); fact(f"abl|{ab}|{d}", {"d": m, "n": len(x), "sig": bool(star)})
    rows.append(f"{lab(ab)} & {g.seed.nunique()} & " + " & ".join(cs) + r" \\")
write("ablation", tab("table*", "tab:ablation", "Model ablations (P view, identical ablation budget).", "lrrrrrr",
      [r"Variant & Seeds & VAL & PhreshTest & Crawl.24 & CompPhish & D4v \\"], rows,
      r"Twelve epochs over all TRAIN rows (patience as in the main models) for every variant and the reference. First row: ROC-AUC of the reference, mean $\pm$ SD over its seeds; other rows: paired difference to the reference, averaged over seeds. $^{*}$: the 95\% $t$-interval of the paired difference over seeds excludes zero. Lower block: the configuration selected by the post hoc search, the same without its recurrent branch, and the pre-specified configuration with only the recurrent branch added."))

# ---------------- need statistic, identifiability, probes ----------------
NS = rd("revision/R12_ecrac_need_statistic_from_logs.csv")
fact("need", NS.to_dict("records"))
CI = rd("reports/corpus_identifiability.csv"); fact("idf", CI.to_dict("records"))
CR = rd("reports/corpus_identifiability_representation.csv"); fact("idf_repr", CR.to_dict("records"))
V2I = rd("reports/v14_corpus_identifiability_representation.csv"); fact("idf_repr_v2", V2I.to_dict("records"))
PR = rd("revision/R8_probe_url_embedding.csv"); fact("probe", PR.to_dict("records"))
SI = rd("revision/R8_string_level_identifiability.csv"); fact("idf_string", SI.to_dict("records"))
TS = rd("revision/R8_shortcut_threshold_sensitivity.csv"); fact("shortcut_sens", TS.to_dict("records"))
P1 = rd("revision/R1_crawling2024_provenance.csv"); fact("crawl_prov", dict(zip(P1.item, P1.value)))
FPC = rd("revision/R7_false_positive_categories.csv"); fact("fp_categories", FPC.to_dict("records"))
R19 = rd("revision/R19_urltran_style_bert.csv"); fact("r19", R19.to_dict("records"))
R18w = rd("revision/R18_forward_in_time_trend.csv"); fact("r18_trend", R18w.to_dict("records"))
R18D = rd("revision/R18_trackD_ecrac_forward_in_time.csv"); fact("r18_trackD", R18D.to_dict("records"))
R18R = rd("revision/R18_phreshphish_date_ranges.csv"); fact("r18_ranges", R18R.to_dict("records"))
ST = rd("reports/statistical_tests.csv"); fact("wilcoxon", ST[ST.test.str.startswith("Wilcoxon")].to_dict("records"))
EL = rd("reports/ecrac_tauN_calibration_windows.csv"); fact("tauN_cal_rows", len(EL))
dump_facts(); print("ok")
