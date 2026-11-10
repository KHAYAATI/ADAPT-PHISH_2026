from lib import *
M = ["ADAPT-X (P)", "ADAPT-Net (P)", "LightGBM (P)", "Char-LR", "LightGBM (P) + $n$-grams", "URLNet-style CNN", "BiGRU", "Simple stack (no neural)"]
for ds in CORP:
    us = "unseen_domain (domain-disjoint)" if ds == "PhreshTest" else "unseen_domain"
    for m in M:
        s = tm(ENS[m], ds, subset="seen_domain"); u = tm(ENS[m], ds, subset=us)
        fact(f"seen|{ds}|{m}", {"seen": s.roc_auc, "unseen": u.roc_auc, "drop": s.roc_auc - u.roc_auc, "n_seen": int(s.n), "n_unseen": int(u.n)})
for nm, m in [("ADAPT-X2 (P)", "ADAPT-X2 stack (P)"), ("BiGRU, tuned", "BiGRU tuned seed-ens"), ("ADAPT-Net v2, tuned (P)", "ADAPT-PHISH v2 (P) seed-ens")]:
    for ds in CORP:
        us = "unseen_domain (domain-disjoint)" if ds == "PhreshTest" else "unseen_domain"
        s = tm(m, ds, subset="seen_domain"); u = tm(m, ds, subset=us)
        fact(f"seen|{ds}|{nm}", {"seen": s.roc_auc, "unseen": u.roc_auc, "drop": s.roc_auc - u.roc_auc})
dump_facts()
