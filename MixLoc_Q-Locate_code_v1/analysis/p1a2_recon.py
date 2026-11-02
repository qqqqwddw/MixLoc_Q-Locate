# -*- coding: utf-8 -*-
"""P1-A2: reconcile offline dp-full vs engine decoded on exported arrays.

Sample-level proof:
  A) engine decode() on exported alpha/a  vs  stored decoded_spans  (export integrity)
  B) offline dp (faithful port of decodecmp.dp_batch, incl. its quirks) vs stored
  C) offline-fixed dp (float64, old-h in M transition, +beta*a_start(0) init) vs engine
  D) sample metrics (charF1/IoU50/exact/MAE) per variant vs gold
  E) dp-const-4.0 collapse diagnosis (val vs test): AI char ratio, spans/doc, all-H/all-M
  F) gated gain decomposition on PARTIAL docs by predicted doc label
"""
import gzip, json, sys, os
import numpy as np

ENGINE = "/OMFG/duomotai-backup-20260924/tev4_storage/code/forensic_Q_qwen_char_v1_1_20261014"
sys.path.insert(0, ENGINE)
from decode import decode, mask_spans  # engine ground truth

RUN = "/OMFG/duomotai-backup-20260924/tev4_storage/runs_q/full_Q_locate_fast_seed0_20261014"
TESTDIR = f"{RUN}/eval_test"
VALDIR = f"{RUN}/eval_step1433"
DATA = {
    "test": "/OMFG/duomotai-backup-20260924/dataset_v2/dataset_v2_test.jsonl.gz",
    "val": "/OMFG/duomotai-backup-20260924/dataset_v2/dataset_v2_val.jsonl.gz",
}
BETA = 0.25

# ---------- offline ports (single-doc, faithful to decodecmp.py) ----------

def dp_offline_asis(alpha, a):
    """Faithful port of decodecmp.dp_batch: float32, new-h bug, no a_start(0),
    extra free-H position n (terminal a_end(n) folded into h at pos n)."""
    n = len(alpha)
    al = np.asarray(alpha, dtype=np.float32)
    aa = np.asarray(a, dtype=np.float32)
    EM = (-np.logaddexp(0.0, -al)).astype(np.float32)
    EH = (-np.logaddexp(0.0, al)).astype(np.float32)
    AS = np.zeros(n + 1, np.float32); AE = np.zeros(n + 1, np.float32)
    AS[:] = aa[:n + 1, 0]; AE[:] = aa[:n + 1, 1]
    AS[n + 1:] = 0; AE[n + 1:] = 0
    h = EH[0].copy(); m = EM[0].copy()
    back = np.zeros((n + 1, 2), np.int8)
    for t in range(1, n + 1):
        eh = np.float32(0.0) if t == n else EH[t]   # EH[n]=0 padding
        em = -np.float32(1e18) if t == n else EM[t]
        cand = m + AE[t]
        b0 = (cand >= h)
        b1 = (m >= h + AS[t])
        h_new = eh + (cand if b0 else h)
        m = em + (m if b1 else h + AS[t])          # uses NEW h (the bug, kept)
        h = h_new
        back[t] = (b0, b1)
    st = int(m >= h)
    mask = np.zeros(n, dtype=bool)
    for t in range(n, -1, -1):
        if t < n:
            mask[t] = (st == 1)
        st = int(back[t, st]) if t > 0 else 0
    return mask


def dp_offline_fixed(alpha, a):
    """Engine-equivalent: float64, old-h in M transition, M0 = s_M(0)+beta*a_start(0)."""
    n = len(alpha)
    al = np.asarray(alpha, dtype=np.float64)
    aa = np.asarray(a, dtype=np.float64)
    EM = -np.logaddexp(0.0, -al)
    EH = -np.logaddexp(0.0, al)
    h, m = 0.0, -np.inf
    back = np.zeros((n, 2), np.int8)
    for t in range(n):
        from_m_h = (m + BETA * aa[t, 1] >= h) if t > 0 else False
        from_m_m = (m >= h + BETA * aa[t, 0]) if t > 0 else False
        if t == 0:
            nh = EH[0]
            nm = EM[0] + BETA * aa[0, 0]
        else:
            nh = EH[t] + (m + BETA * aa[t, 1] if from_m_h else h)
            nm = EM[t] + (m if from_m_m else h + BETA * aa[t, 0])
        back[t] = (int(from_m_h), int(from_m_m))
        h, m = nh, nm
    st = int(m + BETA * aa[n, 1] >= h)
    mask = np.zeros(n, dtype=bool)
    for t in range(n - 1, -1, -1):
        mask[t] = (st == 1)
        st = int(back[t, st])
    return mask


def dp_const(alpha, penalty):
    """Constant-penalty DP (no a field), engine-style, float64: a== -penalty everywhere."""
    n = len(alpha)
    al = np.asarray(alpha, dtype=np.float64)
    EM = -np.logaddexp(0.0, -al)
    EH = -np.logaddexp(0.0, al)
    h, m = 0.0, -np.inf
    back = np.zeros((n, 2), np.int8)
    for t in range(n):
        fmh = (m - penalty >= h) if t > 0 else False
        fmm = (m >= h - penalty) if t > 0 else False
        if t == 0:
            nh, nm = EH[0], EM[0] - penalty
        else:
            nh = EH[t] + (m - penalty if fmh else h)
            nm = EM[t] + (m if fmm else h - penalty)
        back[t] = (int(fmh), int(fmm))
        h, m = nh, nm
    st = int(m - penalty >= h)
    mask = np.zeros(n, dtype=bool)
    for t in range(n - 1, -1, -1):
        mask[t] = (st == 1)
        st = int(back[t, st])
    return mask

# ---------- metrics ----------

def span_metrics(pred_spans, gold_spans):
    pred = [tuple(x) for x in pred_spans]; gold = [tuple(x) for x in gold_spans]
    exact = len(set(pred) & set(gold))
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
    return dict(np_pred=len(pred), n_gold=len(gold), exact=exact, iou=iou, err=err, ends=ends)

def f1(tp, fp, fn):
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return round(2 * p * r / max(1e-12, p + r), 4)

# ---------- sampling ----------

def sample_docs(evaldir, per_rank=450, longest=50, need_pred=True):
    """First `per_rank` PARTIAL-gold docs per rank + `longest` longest docs overall.
    Returns {did: rec}, where rec has arrays info; PARTIAL-ness decided later via gold."""
    recs = {}
    longest_buf = []  # (n, did, rank, line_no)
    memmaps = {r: np.memmap(f"{evaldir}/rank{r}.f32", dtype="<f4", mode="r") for r in range(4)}
    for r in range(4):
        cnt = 0
        with open(f"{evaldir}/rank{r}.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                did = rec["doc_id"]
                n = rec["arrays"]["alpha"]["shape"][0]
                rec["rank"] = r; rec["n"] = n
                recs[did] = rec
                longest_buf.append((n, did))
                cnt += 1
                if cnt >= per_rank:
                    break
    longest_buf.sort(reverse=True)
    extra = [d for _, d in longest_buf[:longest]]
    return recs, set(extra), memmaps

def get_arrays(rec, mm):
    sl = rec["arrays"]
    out = {}
    for k in ("alpha", "a"):
        off = sl[k]["offset"]; sh = sl[k]["shape"]
        out[k] = np.array(mm[off:off + int(np.prod(sh))], dtype="<f4").reshape(sh)
    return out

def load_gold(split, ids):
    gold = {}
    with gzip.open(DATA[split], "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            did = r["row_id_v2"]
            if did in ids:
                gold[did] = dict(label=r["label"], n=len(r["text"]),
                                 spans=[(int(s), int(e)) for s, e in (r.get("machine_spans") or [])])
    return gold

# ---------- main ----------

def run_split(split, evaldir, per_rank, longest):
    recs, extra, mms = sample_docs(evaldir, per_rank, longest)
    gold = load_gold(split, set(recs))
    partial = [d for d in recs if gold.get(d, {}).get("label") == "PARTIAL_AI"]
    print(f"[{split}] sampled={len(recs)} partial={len(partial)}", flush=True)

    res = {k: dict(docs=0, span_eq=0, char=[0, 0, 0], sp=dict(np_pred=0, n_gold=0, exact=0, iou=0, err=0, ends=0),
                   mismatch_n=[], mismatch_start0=0, mismatch_endn=0)
           for k in ("stored", "engine", "asis", "fixed")}
    for did in partial:
        rec = recs[did]; g = gold[did]
        n = g["n"]
        if rec["n"] != n:
            print("LEN MISMATCH", did, rec["n"], n); continue
        arr = get_arrays(rec, mms[rec["rank"]])
        alpha, a = arr["alpha"].astype(np.float64), arr["a"].astype(np.float64)
        gm = np.zeros(n, dtype=bool)
        for s, e in g["spans"]:
            gm[s:e] = True
        stored_spans = [tuple(x) for x in rec["decoded_spans"]]
        eng_mask, eng_spans = decode(alpha, a)
        asis_mask = dp_offline_asis(alpha, a)
        fixed_mask = dp_offline_fixed(alpha, a)
        variants = {
            "stored": (None, stored_spans),
            "engine": (eng_mask, [tuple(x) for x in eng_spans]),
            "asis": (asis_mask, mask_spans(asis_mask)),
            "fixed": (fixed_mask, mask_spans(fixed_mask)),
        }
        for k, (mask, spans) in variants.items():
            R = res[k]
            R["docs"] += 1
            if set(spans) == set(stored_spans):
                R["span_eq"] += 1
            else:
                R["mismatch_n"].append(n)
                if g["spans"] and g["spans"][0][0] == 0:
                    R["mismatch_start0"] += 1
                if g["spans"] and g["spans"][-1][1] == n:
                    R["mismatch_endn"] += 1
            if mask is None:  # stored: rebuild mask from spans
                mask = np.zeros(n, dtype=bool)
                for s, e in spans:
                    mask[s:e] = True
            tp = int((mask & gm).sum()); fp = int((mask & ~gm).sum()); fn = int((gm & ~mask).sum())
            R["char"][0] += tp; R["char"][1] += fp; R["char"][2] += fn
            sm = span_metrics(spans, g["spans"])
            for kk in ("np_pred", "n_gold", "exact", "iou", "err", "ends"):
                R["sp"][kk] += sm[kk]
    out = {}
    for k, R in res.items():
        S = R["sp"]
        out[k] = dict(docs=R["docs"], span_eq_rate=round(R["span_eq"] / max(1, R["docs"]), 4),
                      charF1=f1(*R["char"]),
                      exactF1=f1(S["exact"], S["np_pred"] - S["exact"], S["n_gold"] - S["exact"]),
                      iou50F1=f1(S["iou"], S["np_pred"] - S["iou"], S["n_gold"] - S["iou"]),
                      MAE=round(S["err"] / max(1, S["ends"]), 2),
                      matched=S["iou"], pred_spans=S["np_pred"], gold_spans=S["n_gold"],
                      mismatch_mean_n=(round(float(np.mean(R["mismatch_n"])), 1) if R["mismatch_n"] else None),
                      mismatch_start0=R["mismatch_start0"], mismatch_endn=R["mismatch_endn"])
    return out, recs, gold, mms


def collapse_and_gated(split, evaldir, per_rank, mms_holder):
    """E) dp-const-4.0 collapse: val vs test; F) gated decomposition by pred label."""
    recs, extra, mms = sample_docs(evaldir, per_rank, 0)
    gold = load_gold(split, set(recs))
    partial = [d for d in recs if gold.get(d, {}).get("label") == "PARTIAL_AI"]
    stats = {v: dict(docs=0, ai_chars=0, tot=0, spans=0, allH=0, allM=0)
             for v in ("raw", "dp-full", "dp-const-4.0")}
    gated_grp = {0: [0, 0, 0, 0, 0, 0, 0], 1: [0, 0, 0, 0, 0, 0, 0], 2: [0, 0, 0, 0, 0, 0, 0]}
    # per pred-label: [docs, dec_tp, dec_fp, dec_fn, gat_tp, gat_fp, gat_fn]
    for did in partial:
        rec = recs[did]; g = gold[did]; n = g["n"]
        arr = get_arrays(rec, mms[rec["rank"]])
        alpha, a = arr["alpha"].astype(np.float64), arr["a"].astype(np.float64)
        gm = np.zeros(n, dtype=bool)
        for s, e in g["spans"]:
            gm[s:e] = True
        raw = alpha >= 0
        full = dp_offline_fixed(alpha, a)
        const = dp_const(alpha, 4.0)
        pred = int(rec["prediction"])
        for v, mask in (("raw", raw), ("dp-full", full), ("dp-const-4.0", const)):
            S = stats[v]
            S["docs"] += 1; S["tot"] += n
            S["ai_chars"] += int(mask.sum()); S["spans"] += len(mask_spans(mask))
            if not mask.any():
                S["allH"] += 1
            if mask.all():
                S["allM"] += 1
        gated = np.zeros(n, dtype=bool) if pred == 0 else np.ones(n, dtype=bool) if pred == 2 else full
        row = gated_grp[pred]
        row[0] += 1
        row[1] += int((full & gm).sum()); row[2] += int((full & ~gm).sum()); row[3] += int((gm & ~full).sum())
        row[4] += int((gated & gm).sum()); row[5] += int((gated & ~gm).sum()); row[6] += int((gm & ~gated).sum())
    coll = {v: dict(docs=S["docs"], ai_ratio=round(S["ai_chars"] / max(1, S["tot"]), 4),
                    spans_per_doc=round(S["spans"] / max(1, S["docs"]), 2),
                    allH=S["allH"], allM=S["allM"]) for v, S in stats.items()}
    gg = {str(k): dict(docs=r[0],
                       dec_charF1=f1(r[1], r[2], r[3]), gat_charF1=f1(r[4], r[5], r[6]),
                       dec_tp=r[1], dec_fp=r[2], dec_fn=r[3], gat_tp=r[4], gat_fp=r[5], gat_fn=r[6])
          for k, r in gated_grp.items()}
    gold_ai_ratio = None
    return dict(collapse=coll, gated_by_predlabel=gg)


def main():
    out = {}
    for split, evaldir, per_rank in (("test", TESTDIR, 450), ("val", VALDIR, 350)):
        print(f"===== {split} =====", flush=True)
        recon, recs, gold, mms = run_split(split, evaldir, per_rank, 50)
        out[f"{split}_reconciliation"] = recon
        cg = collapse_and_gated(split, evaldir, min(per_rank, 300), None)
        out[f"{split}_collapse_gated"] = cg
        print(json.dumps({k: recon[k] for k in ("stored", "engine", "asis", "fixed")}, indent=1), flush=True)
        print(json.dumps(cg, indent=1), flush=True)
    with open("/tmp/p1a2_recon.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("WROTE /tmp/p1a2_recon.json")

if __name__ == "__main__":
    main()
