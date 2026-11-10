from lib import *
from t_more import tab

EV = rd("revision/R12_ecrac_every_run.csv"); ACC = rd("revision/R12_ecrac_update_accounting.csv")
TRK = [("A: CompPhish pool", "A"), ("B-mix: PhishTankNovel + benign", "B-mix"), ("B: PhishTankNovel", "B"), ("C: controlled drift", "C")]
VN = [("Frozen", "Frozen (static)"), ("Naive fine-tuning (A8)", "Naive (periodic fine-tuning = A8)"), ("Replay only", "Replay-only"),
      ("Drift-triggered", "Drift-triggered"), ("Drift-triggered + replay", "Drift + Replay"),
      ("ECRAC (A0, a priori $\\tau_N$)", "ECRAC (A0 full)"), ("ECRAC-cal (calibrated $\\tau_N$)", "ECRAC-cal (tau_N calibrated on no-drift development windows)"),
      ("A1 -- evidence consistency", "A1 − evidence consistency"), ("A1-cal", "A1-cal (− evidence consistency, calibrated tau_N)")]
ABL = [("ECRAC (A0 full)", "ECRAC (A0 full)"), ("A2 -- Need--Risk separation", "A2 − Need–Risk separation"), ("A3 -- gradient-conflict test", "A3 − gradient-conflict test"),
       ("A4 -- trust gate / quarantine", "A4 − trust gate / quarantine"), ("A5 -- adaptive budget", "A5 − adaptive budget"),
       ("A6 -- rare-class protection", "A6 − rare-class protection"), ("A7 -- adaptation certificate", "A7 − adaptation certificate")]

def cell(track, var):
    e = EV[(EV.track == track) & (EV.variant == var)]; a = ACC[(ACC.track == track) & (ACC.variant == var)].iloc[0]
    far_runs = e[e.attempted_updates > 0]
    far = far_runs.false_adaptation_rate.mean() if len(far_runs) else np.nan
    d = {"score": e.score.mean(), "forg": a.forgetting_independent, "committed": int(a.committed), "attempted": int(a.attempted), "far": far,
         "runs": int(a.runs), "windows": int(a.windows), "rejected": int(a.rejected), "forg_ctrl": a.forgetting_ctrl, "forg_ext": a.forgetting_ext,
         "rej_new": a.get("rejected_because::new"), "rej_conflict": a.get("rejected_because::conflict"), "rej_old": a.get("rejected_because::old"),
         "rej_rare": a.get("rejected_because::rare"), "score_source": e.score_source.iloc[0]}
    return d

rows = []
for nm, v in VN:
    if nm.startswith("ECRAC (A0") or nm.startswith("A1 --"): rows.append(r"\midrule")
    cs = []
    for tr, tag in TRK:
        d = cell(tr, v); fact(f"ecrac|{tag}|{nm}", d)
        cs += [f3(d["score"]), f3(d["forg"]), f"{d['committed']}/{d['attempted']}", "--" if np.isnan(d["far"]) else f"{d['far']:.2f}"]
    rows.append(f"{nm} & " + " & ".join(cs) + r" \\")
write("ecrac", tab("table*", "tab:ecrac", "Continual adaptation: scores, independent forgetting and update accounting.", "lrrrrrrrrrrrrrrrr",
      [r" & \multicolumn{4}{c}{A: CompPhish pool} & \multicolumn{4}{c}{B-mix: PhishTank + benign} & \multicolumn{4}{c}{B: PhishTankNovel} & \multicolumn{4}{c}{C: controlled drift} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}\cmidrule(lr){14-17}",
       r"Variant & Score & Forg. & C/A & FAR & Score & Forg. & C/A & FAR & Score & Forg. & C/A & FAR & Score & Forg. & C/A & FAR \\"], rows,
      r"Score: final CompPhish F1 (A; C for the five shift scenarios) or prequential F1 (B-mix; C for the no-drift and class-prior scenarios) or prequential recall (B), mean over runs. Forg.: drop of the macro-F1 on the independent retention set (never read by any controller) from its maximum to the end of the run. C/A: committed / attempted updates summed over runs (A, B, B-mix: 5 runs; C: 35 runs = 7 scenarios $\times$ 5 seeds). FAR: false-adaptation rate over runs with at least one attempted update; --: no attempt. Ablations A2--A7 and the external (PhreshTest) retention are in Appendix~\ref{sec:appB}.",
      sep="2pt", width=r"0.99\textwidth"))

rows = []
for nm, v in ABL:
    cs = []
    for tr, tag in TRK:
        d = cell(tr, v); fact(f"ecrac_abl|{tag}|{nm}", d)
        cs += [f3(d["score"]), f"{d['forg']:.4f}", f"{d['committed']}/{d['attempted']}"]
    rows.append(f"{nm} & " + " & ".join(cs) + r" \\")
write("ecrac_ablation", tab("table*", "tab:ecrac_ablation", r"Controller ablations A2--A7 (a-priori $\tau_N$).", "lrrrrrrrrrrrr",
      [r" & \multicolumn{3}{c}{A} & \multicolumn{3}{c}{B-mix} & \multicolumn{3}{c}{B} & \multicolumn{3}{c}{C} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
       r"Variant & Score & Forg. & C/A & Score & Forg. & C/A & Score & Forg. & C/A & Score & Forg. & C/A \\"], rows, r"Columns as in \tabref{tab:ecrac}.", sep="2pt"))

rows = []
for nm, v in VN:
    cs = []
    for tr, tag in TRK:
        d = cell(tr, v); cs += [f"{d['forg']:.4f}", f"{d['forg_ctrl']:.4f}", f"{d['forg_ext']:.4f}"]
    rows.append(f"{nm} & " + " & ".join(cs) + r" \\")
write("retention", tab("table*", "tab:retention", "Forgetting on the three retention sets.", "lrrrrrrrrrrrr",
      [r" & \multicolumn{3}{c}{A} & \multicolumn{3}{c}{B-mix} & \multicolumn{3}{c}{B} & \multicolumn{3}{c}{C} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
       r"Variant & Ind. & Ctrl. & Ext. & Ind. & Ctrl. & Ext. & Ind. & Ctrl. & Ext. & Ind. & Ctrl. & Ext. \\"], rows,
      r"Ind.: independent VALIDATION partition never read by a controller (reported in the main text). Ctrl.: partition read by the controller. Ext.: unseen-domain PhreshTest rows, external to development.", sep="2pt"))

# ---------------- corrupted labels (Track A-contam) ----------------
CS = [("Clean", None), ("Rand. 10\\%", "random flips 10%"), ("Rand. 30\\%", "random flips 30%"), ("Targeted 30\\%", "targeted: 30% of phishing labels -> benign"),
      ("Campaign 30\\%", "campaign: every phishing label of 30% of domains -> benign")]
PV = [("Frozen", "Frozen (static)"), ("Naive (A8)", "Naive (periodic fine-tuning = A8)"), ("Drift + replay", "Drift + Replay"), ("ECRAC (A0)", "ECRAC (A0 full)"),
      ("ECRAC-cal", "ECRAC-cal (tau_N calibrated on no-drift development windows)"), ("A4 -- trust gate", "A4 − trust gate / quarantine"), ("A7 -- certificate", "A7 − adaptation certificate")]
CL = rd("reports/continual_learning_results.csv")
rows = []
for nm, v in PV:
    cs = []
    for lab, sc in CS:
        g = CL[(CL.track == "A: CompPhish pool") & (CL.variant == v)] if sc is None else CL[(CL.track.str.startswith("A-contam")) & (CL.scenario == sc) & (CL.variant == v)]
        if not len(g): cs += ["--", "--"]; continue
        cs += [f3(g.final_target_f1.mean()), f"{g['update_count (commits)'].mean():.1f}"]
        fact(f"contam|{lab}|{nm}", {"f1": g.final_target_f1.mean(), "upd": g["update_count (commits)"].mean(), "quar": g.quarantine_rate.mean() if "quarantine_rate" in g else None})
    rows.append(f"{nm} & " + " & ".join(cs) + r" \\")
write("v14_contam", tab("table*", "tab:v14contam", "Track A with corrupted revealed labels: final CompPhish F1 and committed updates per run (mean of five stream seeds).", "lrrrrrrrrrr",
      [r"& \multicolumn{2}{c}{Clean} & \multicolumn{2}{c}{Rand. 10\%} & \multicolumn{2}{c}{Rand. 30\%} & \multicolumn{2}{c}{Targeted 30\%} & \multicolumn{2}{c}{Campaign 30\%} \\",
       r"Policy & F1 & Upd. & F1 & Upd. & F1 & Upd. & F1 & Upd. & F1 & Upd. \\"], rows,
      r"Labels revealed to the learner are corrupted after prediction (random flips; 30\% of phishing labels reported as benign; or every phishing label of a fixed 30\% of registrable domains reported as benign, a consistent campaign); prequential metrics and the final CompPhish target use the true labels. The certificate's held-out labels are drawn from the same corrupted window. A score-aware attacker is reported in \tabref{tab:r20}.", sep="2.5pt"))

# ---------------- R20 + R20b ----------------
R20 = rd("revision/R20_ecrac_retuned_on_development_streams.csv"); R20b = rd("revision/R20b_consistent_and_adaptive_poisoning.csv"); DEVS = rd("revision/R20_ecrac_development_search.csv")
sel = [v for v in R20.variant.unique() if v.startswith("ECRAC-dev")][0]
fact("r20_selected", sel)
J = DEVS.groupby(["base", "tau_N"]).J.mean().reset_index(); fact("r20_J", J.to_dict("records"))
V20 = [("Frozen", "Frozen (static)"), ("Naive (A8)", "Naive (periodic fine-tuning = A8)"), ("ECRAC (A0)", "ECRAC (A0 full)"),
       ("ECRAC-cal", "ECRAC-cal (tau_N calibrated on no-drift development windows)"), ("ECRAC-dev$^{\\ddagger}$", sel)]
SC = {"A: CompPhish pool": "final_target_f1", "B-mix: PhishTankNovel + benign": "prequential_mean_f1", "B: PhishTankNovel": "prequential_mean_recall", "C: controlled drift": "final_target_f1"}
rows = []
for nm, v in V20:
    cs = []
    for tr, _ in TRK:
        r = R20[(R20.track == tr) & (R20.variant == v)].iloc[0]
        cs += [f3(r[SC[tr]]), f"{r['update_count (commits)']:.1f}"]
        fact(f"r20|{tr}|{nm}", {"score": r[SC[tr]], "commits": r["update_count (commits)"], "forg": r[R20.columns[5]]})
    rows.append(f"{nm} & " + " & ".join(cs) + r" \\")
rows.append(r"\midrule")
rows.append(r"\multicolumn{9}{l}{\textit{Poisoned revealed labels, Track A: final CompPhish F1 / FPR}} \\")
rows.append(r"Policy & \multicolumn{2}{c}{Score-aware 30\%} & \multicolumn{2}{c}{Campaign 30\%} & \multicolumn{2}{c}{Targeted 30\%} & \multicolumn{2}{c}{} \\")
PS = [("adaptive", "Score-aware 30\\%"), ("campaign", "Campaign 30\\%"), ("targeted", "Targeted 30\\%")]
for nm, v in V20 + [("Drift + replay", "Drift + Replay"), ("A4 -- trust gate", "A4 − trust gate / quarantine"), ("A7 -- certificate", "A7 − adaptation certificate")]:
    cs = []
    for key, lab in PS:
        r = R20b[R20b.scenario.str.startswith(key) & (R20b.variant == v)].iloc[0]
        cs.append(f"\\multicolumn{{2}}{{c}}{{{f3(r.final_target_f1)} / {f3(r.final_target_fpr)}}}")
        fact(f"r20b|{key}|{nm}", {"f1": r.final_target_f1, "fpr": r.final_target_fpr, "recall": r.final_target_recall, "commits": r["update_count (commits)"],
                                  "camp_rec": r.final_target_campaign_recall, "camp_rec_frozen": r.frozen_target_campaign_recall})
    rows.append(f"{nm} & " + " & ".join(cs) + r" & \multicolumn{2}{c}{} \\")
hdr = [r" & \multicolumn{2}{c}{A} & \multicolumn{2}{c}{B-mix} & \multicolumn{2}{c}{B} & \multicolumn{2}{c}{C} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
       r"Policy & Score & Upd. & Score & Upd. & Score & Upd. & Score & Upd. \\"]
fixed = rows
note = (r"Upper block: ECRAC with $\tau_N$ and the evidence-consistency switch selected on development streams only (no drift, abrupt and gradual Mendeley$\to$PhreshPhish shift, class-prior shift; ten windows of 400 VALIDATION calibration rows; three seeds; criterion fixed in advance: forward transfer minus controller-side forgetting). "
        r"$^{\ddagger}$The search selected the a-priori configuration (A0, $\tau_N=0.25$). Score and Upd. (committed updates per run) as in \tabref{tab:ecrac}, five stream seeds. On B-mix the ECRAC-dev score (no update) was computed on independently sampled windows and equals the frozen detector on them; it is not paired with the other rows of that column. "
        r"Track C: final CompPhish F1 over the five shift scenarios. Lower block: final CompPhish F1 / FPR under a score-aware attacker (the 30\% phishing labels on which the deployed model is least confident reported as benign), a consistent campaign and the targeted flips of \tabref{tab:v14contam}; these runs were repeated in a separate cell, and GPU update steps are not bit-reproducible, so the campaign and targeted values differ from \tabref{tab:v14contam} by at most 0.009.")
write("r20", tab("table*", "tab:r20", "ECRAC re-tuned on development streams, and label poisoning by a consistent and a score-aware attacker.", "lrrrrrrrr", hdr, fixed, note, sep="2.5pt"))
dump_facts(); print("ok")
