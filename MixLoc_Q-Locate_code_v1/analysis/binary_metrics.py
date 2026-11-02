# -*- coding: utf-8 -*-
"""Binary protocol metrics (HUMAN vs AI-involved) from exported doc_logits.
Reproduces: AUROC 0.9570 / BACC 0.9113 / F1(AI) 0.9102 / Spec 0.9590 on the full
test split (n=202,762), BACC-tuned operating point. CPU-only.

Usage: point EVAL at the eval dir containing rank{0..3}.jsonl + rank{0..3}.f32,
and DATA at dataset_v2_test.jsonl.gz."""
import gzip, json
import numpy as np

EVAL = ".../eval_test"
DATA = ".../dataset_v2_test.jsonl.gz"

gold = {}
with gzip.open(DATA, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        gold[r["row_id_v2"]] = 0 if r["label"] == "HUMAN" else 1

scores, labels = [], []
mms = {r: np.memmap(f"{EVAL}/rank{r}.f32", dtype="<f4", mode="r") for r in range(4)}
for r in range(4):
    with open(f"{EVAL}/rank{r}.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sl = rec["arrays"]["doc_logits"]
            z = np.asarray(mms[r][sl["offset"]:sl["offset"] + 3], dtype="<f4")
            p = np.exp(z - z.max()); p = p / p.sum()
            scores.append(1.0 - p[0])          # P(AI-involved)
            labels.append(gold[rec["doc_id"]])

s = np.asarray(scores); y = np.asarray(labels)
P, N = int(y.sum()), len(y) - y.sum()

order = np.argsort(s, kind="mergesort")       # AUROC with tie-averaged ranks
sr = s[order]; ranks = np.empty(len(s)); i = 0
while i < len(s):
    j = i
    while j + 1 < len(s) and sr[j + 1] == sr[i]:
        j += 1
    ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
    i = j + 1
auroc = (ranks[y == 1].sum() - P * (P + 1) / 2) / (P * N)

idx = order[::-1]                              # BACC-tuned operating point
tps = np.cumsum(y[idx]); fps = np.cumsum(1 - y[idx])
bacc = tps / P + (1 - fps / N)
k = int(np.argmax(bacc))
TP, FP = int(tps[k]), int(fps[k]); TN = N - FP
prec = TP / max(1, TP + FP); rec = TP / P
print(json.dumps(dict(n=len(y), AUROC=round(float(auroc), 4),
                      BACC=round(float((TP / P + TN / N) / 2), 4),
                      F1_AI=round(float(2 * prec * rec / (prec + rec)), 4),
                      Spec=round(float(TN / N), 4),
                      threshold=round(float(s[idx[k]]), 6))))
