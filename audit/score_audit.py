"""Score the Crawling2024 manual label audit after it has been filled in.
Usage: python score_audit.py Crawling2024_label_audit.xlsx crawling2024_label_audit_key.csv [second_annotator.xlsx]
If the workbook has an AI_prescreen sheet, agreement between the AI pre-screen and the human labels is also reported.
Run it only after annotation is finished: it un-blinds the sample (false positive vs control)."""
import sys, math, pandas as pd
from scipy.stats import fisher_exact
def load(p):
    if p.endswith(".xlsx"):
        sheets=pd.ExcelFile(p).sheet_names
        d=pd.read_excel(p,sheet_name="Audit_aveugle" if "Audit_aveugle" in sheets else "Audit")
    else: d=pd.read_csv(p)
    lab=[c for c in d.columns if str(c).startswith("manual_label")][0]
    return d[["audit_id",lab]].rename(columns={lab:"manual"}).assign(manual=lambda x:x.manual.astype(str).str.strip().str.lower().replace({"nan":None}))
def wilson(k,n,z=1.959964):
    if n==0: return (float("nan"),)*2
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return c-h,c+h
a=load(sys.argv[1]); key=pd.read_csv(sys.argv[2])[["audit_id","sample"]]
m=key.merge(a,on="audit_id",how="left")
VALID={"phishing","legitimate","unsure","unreachable"}
bad=m[m.manual.notna() & ~m.manual.isin(VALID)]
miss=m.manual.isna().sum()
if len(bad): sys.exit(f"invalid labels in {len(bad)} rows: {bad.audit_id.tolist()[:10]}")
if miss: print(f"WARNING: {miss} of {len(m)} rows not annotated; results below are incomplete.\n")
m["grp"]=m["sample"].str.startswith("false positive").map({True:"FP",False:"control"})
print("| Sample | n | phishing | legitimate | unsure | unreachable | phishing share of decided [95% Wilson CI] |")
print("|---|---|---|---|---|---|---|")
res={}
for g in ["FP","control"]:
    x=m[m.grp==g]; c=x.manual.value_counts()
    ph,lg=int(c.get("phishing",0)),int(c.get("legitimate",0)); dec=ph+lg
    lo,hi=wilson(ph,dec); res[g]=(ph,lg)
    print(f"| {g} | {len(x)} | {ph} | {lg} | {int(c.get('unsure',0))} | {int(c.get('unreachable',0))} | "
          f"{(ph/dec if dec else float('nan')):.3f} [{lo:.3f}, {hi:.3f}] |")
(a1,b1),(a2,b2)=res["FP"],res["control"]
if (a1+b1) and (a2+b2):
    orr,p=fisher_exact([[a1,b1],[a2,b2]]); print(f"\nFisher exact test FP vs control (phishing vs legitimate): OR = {orr:.2f}, p = {p:.3g}")
if len(sys.argv)>3:
    b=load(sys.argv[3]).rename(columns={"manual":"m2"}); j=a.merge(b,on="audit_id").dropna()
    if len(j):
        cats=sorted(VALID); po=(j.manual==j.m2).mean()
        pe=sum((j.manual==c).mean()*(j.m2==c).mean() for c in cats)
        print(f"\nInter-annotator agreement on {len(j)} common rows: {po:.3f} raw, Cohen's kappa = {(po-pe)/(1-pe) if pe<1 else float('nan'):.3f}")

if sys.argv[1].endswith(".xlsx"):
    try:
        ai=pd.read_excel(sys.argv[1],sheet_name="AI_prescreen",header=1)[["audit_id","ai_suggestion"]]
        j=a.merge(ai,on="audit_id").dropna()
        if len(j):
            print(f"\nAI pre-screen vs human label on {len(j)} rows: agreement {(j.manual==j.ai_suggestion).mean():.3f}")
            print(pd.crosstab(j.ai_suggestion,j.manual).to_string())
    except ValueError: pass
