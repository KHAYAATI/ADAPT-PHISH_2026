# Crawling2024 label audit: protocol and record (to be completed by the human annotators)

This form covers steps 2 and 3. Step 1 (selection of the 300 rows, shuffled, without scores) was done by the pipeline.

## A. Protocol (fix before annotating, do not change afterwards)
- Decision rules for phishing / legitimate / unsure / unreachable: those of the "Instructions" sheet, or the amended rules written here:
- Evidence sources allowed (VirusTotal, Wayback Machine 2024 snapshots, urlscan, Google Safe Browsing, others):
- Safety rule: suspicious URLs are not opened directly; only archived copies or an isolated machine are used.
- Blinding: annotators do not see the model scores or the key `crawling2024_label_audit_key.csv` until every row is labelled.
- Double annotation: rows labelled by two annotators independently (recommended ≥ 50):
- AI pre-screen: an automated suggestion from the URL string only (sheet AI_prescreen) was consulted only after each human label was recorded: yes / no. If yes, the article must say so and report the AI–human agreement.
- Disagreement resolution (discussion, third annotator, or kept as "unsure"):

## B. Annotators
| Initials | Real role (e.g. first author, co-author, PhD student) | Security experience | Rows annotated | Dates |
|---|---|---|---|---|

## C. Record
- Start date / end date:
- Rows annotated / 300:
- Disagreements among double-annotated rows (count, how resolved):
- Problems met (unreachable pages, missing archives):

## D. Results (fill from `score_audit.py`, run only after C is complete)
- Output table of `score_audit.py` (pasted unchanged):
- Inter-annotator agreement (Cohen's kappa):
- Files kept: filled workbook(s), screenshots or links listed in the "evidence" column.
