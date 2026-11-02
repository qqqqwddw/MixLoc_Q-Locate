# -*- coding: utf-8 -*-
"""共享加载与度量（解码对照/分层共用）"""
import gzip, json, numpy as np

def load_split_gold_and_preds(split, evaldir, data_gz, need_text=False, need_alpha=False):
    gold, texts = {}, {}
    with gzip.open(data_gz, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            did = r["row_id_v2"]
            n = len(r["text"])
            spans = [(int(s), int(e)) for s, e in (r.get("machine_spans") or [])]
            cov = (sum(e - s for s, e in spans) / n) if (n and r["label"] == "PARTIAL_AI") else None
            gold[did] = {"label": {"HUMAN": 0, "PARTIAL_AI": 1, "FULL_AI": 2}[r["label"]],
                         "n": n, "spans": spans, "coverage": cov,
                         "generator": r.get("generator") or "UNKNOWN",
                         "language": r.get("language") or "UNKNOWN"}
            if need_text:
                texts[did] = r["text"]
    world = 4
    preds = {}
    arrays = {} if need_alpha else None
    for r in range(world):
        with open(f"{evaldir}/rank{r}.jsonl", encoding="utf-8") as f:
            recs = [json.loads(l) for l in f]
        mm = None
        if need_alpha:
            mm = np.memmap(f"{evaldir}/rank{r}.f32", dtype="<f4", mode="r")
        for rec in recs:
            did = rec["doc_id"]
            p = {"pred": int(rec["prediction"]),
                 "n_pred": 0, "tp": 0, "fp": 0, "fn": 0,
                 "exact_tp": 0, "iou_tp": 0, "err": 0, "ends": 0,
                 "n_pred_spans": len(rec.get("decoded_spans") or []),
                 "decoded_spans": [(int(a), int(b)) for a, b in (rec.get("decoded_spans") or [])],
                 "raw_spans": [(int(a), int(b)) for a, b in (rec.get("raw_spans") or [])]}
            preds[did] = p
            if need_alpha:
                arrays[did] = (r, rec["arrays"], mm)
    assert set(preds) == set(gold), f"pred={len(preds)} gold={len(gold)}"
    return (gold, preds, texts, arrays) if need_alpha else (gold, preds, texts)

def char_counts(pmask, gmask):
    tp = int((pmask & gmask).sum()); fp = int((pmask & ~gmask).sum()); fn = int((gmask & ~pmask).sum())
    return tp, fp, fn

def prf(tp, fp, fn):
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return {"precision": round(float(p), 4), "recall": round(float(r), 4),
            "f1": round(float(2 * p * r / max(1e-12, p + r)), 4)}

def iou_mae(pred_spans, gold_spans):
    pred = [tuple(x) for x in pred_spans]
    gold = [tuple(x) for x in gold_spans]
    exact = len(set(pred) & set(gold))
    used = set(); iou_tp = 0; err = 0; ends = 0
    for ps, pe in pred:
        best, bi = 0.5, None
        for j, (gs, ge) in enumerate(gold):
            if j in used: continue
            inter = max(0, min(pe, ge) - max(ps, gs))
            union = (pe - ps) + (ge - gs) - inter
            v = inter / union if union else 0.0
            if v >= best: best, bi = v, j
        if bi is not None:
            used.add(bi); iou_tp += 1
            gs, ge = gold[bi]
            err += abs(ps - gs) + abs(pe - ge); ends += 2
    return {"n_pred": len(pred), "n_gold": len(gold), "exact_tp": exact,
            "iou_tp": iou_tp, "err": err, "ends": ends}
