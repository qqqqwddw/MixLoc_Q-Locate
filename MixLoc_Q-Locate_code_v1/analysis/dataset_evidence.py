# -*- coding: utf-8 -*-
"""Dataset evidence package (CPU): A) per-language x split counts;
B) test PARTIAL per-operation: n, coverage stats, detection+localization metrics
   (using engine-stored decoded_spans = canonical); C) six-generator class composition."""
import gzip, json, os, sys, time
import numpy as np

BASE = "/OMFG/duomotai-backup-20260924"
RUN = f"{BASE}/tev4_storage/runs_q/full_Q_locate_fast_seed0_20261014"
EVAL = f"{RUN}/eval_test"
DATA = {s: f"{BASE}/dataset_v2/dataset_v2_{s}.jsonl.gz" for s in ("train", "val", "test")}
OUT = f"{BASE}/tev4_storage/analysis_dataset_evidence_20261023"
T0 = time.time()
os.makedirs(OUT, exist_ok=True)


def spans_of(spans, n):
    m = np.zeros(n, dtype=bool)
    for s, e in spans:
        m[s:e] = True
    return m


def prf(tp, fp, fn):
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return round(2 * p * r / max(1e-12, p + r), 4)


def span_metrics(pred, gold):
    exact = len(set(map(tuple, pred)) & set(map(tuple, gold)))
    used = set(); iou = 0; err = 0; ends = 0
    for ps, pe in pred:
        best, bi = 0.5, None
        for j, (gs, ge) in enumerate(gold):
            if j in used:
                continue
            inter = max(0, min(pe, ge) - max(ps, gs))
            union = (pe - ps) + (ge - gs) - inter
            v = inter / union if union else 0.0
            if v >= best:
                best, bi = v, j
        if bi is not None:
            used.add(bi); iou += 1
            gs, ge = gold[bi]
            err += abs(ps - gs) + abs(pe - ge); ends += 2
    return exact, iou, err, ends, len(pred), len(gold)


# ---- A) per-language x split counts ----
lang_counts = {}
for split in ("train", "val", "test"):
    with gzip.open(DATA[split], "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lg = r.get("language") or "UNKNOWN"
            lang_counts.setdefault(lg, {"train": 0, "val": 0, "test": 0})[split] += 1
print("[A] languages:", len(lang_counts), f"{(time.time()-T0)/60:.1f}min", flush=True)

# ---- B) per-operation metrics on test PARTIAL ----
gold = {}
with gzip.open(DATA["test"], "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["label"] != "PARTIAL_AI":
            continue
        spans = [(int(s), int(e)) for s, e in (r.get("machine_spans") or [])]
        n = len(r["text"])
        gold[r["row_id_v2"]] = dict(
            n=n, spans=spans, op=r.get("operation") or "(unrecorded)",
            cov=(sum(e - s for s, e in spans) / n) if n else 0.0,
            gen=r.get("generator") or "UNKNOWN")
mms = {r: np.memmap(f"{EVAL}/rank{r}.f32", dtype="<f4", mode="r") for r in range(4)}
ops = {}
for r in range(4):
    with open(f"{EVAL}/rank{r}.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            did = rec["doc_id"]
            g = gold.get(did)
            if not g:
                continue
            n = g["n"]
            dec = [tuple(x) for x in rec.get("decoded_spans") or []]
            pred = int(rec["prediction"])
            off = rec["arrays"]["alpha"]["offset"]
            al = np.asarray(mms[r][off:off + n], dtype="<f4").reshape(n)
            dm = spans_of(dec, n)
            gm = spans_of(g["spans"], n)
            tp = int((dm & gm).sum()); fp = int((dm & ~gm).sum()); fn = int((gm & ~dm).sum())
            ex, iou, err, ends, npred, ngold = span_metrics(dec, g["spans"])
            A = ops.setdefault(g["op"], dict(n=0, covs=[], det=0, tp=0, fp=0, fn=0,
                                             ex=0, iou=0, err=0, ends=0, npred=0, ngold=0,
                                             covb={}))
            A["n"] += 1; A["covs"].append(g["cov"]); A["det"] += int(pred == 1)
            A["tp"] += tp; A["fp"] += fp; A["fn"] += fn
            A["ex"] += ex; A["iou"] += iou; A["err"] += err; A["ends"] += ends
            A["npred"] += npred; A["ngold"] += ngold
            b = ("<5%" if g["cov"] < .05 else "5-20%" if g["cov"] < .2 else
                 "20-50%" if g["cov"] < .5 else "50-80%" if g["cov"] < .8 else "80-100%")
            A["covb"][b] = A["covb"].get(b, 0) + 1
print("[B] ops scanned:", f"{(time.time()-T0)/60:.1f}min", flush=True)

# ---- C) six-generator class composition on test ----
gencomp = {}
with gzip.open(DATA["test"], "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        gen = r.get("generator")
        if gen:
            gencomp.setdefault(gen, {})[r["label"]] = gencomp.setdefault(gen, {}).get(r["label"], 0) + 1

out = dict(
    per_language_counts=lang_counts,
    per_operation_test={op: dict(
        n=A["n"],
        cov_mean=round(float(np.mean(A["covs"])), 4),
        cov_median=round(float(np.median(A["covs"])), 4),
        cov_buckets=A["covb"],
        partial_recall=round(A["det"] / max(1, A["n"]), 4),
        char_f1=prf(A["tp"], A["fp"], A["fn"]),
        exact_f1=prf(A["ex"], A["npred"] - A["ex"], A["ngold"] - A["ex"]),
        iou50_f1=prf(A["iou"], A["npred"] - A["iou"], A["ngold"] - A["iou"]),
        mae=round(A["err"] / max(1, A["ends"]), 2),
        n_pred=A["npred"], n_gold=A["ngold"], n_matched=A["iou"])
        for op, A in ops.items()},
    generator_class_composition=gencomp,
    runtime_min=round((time.time() - T0) / 60, 1))
json.dump(out, open(f"{OUT}/dataset_evidence.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out["per_operation_test"], ensure_ascii=False, indent=1)[:800])
print("WROTE", f"{OUT}/dataset_evidence.json")
