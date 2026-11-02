# -*- coding: utf-8 -*-
"""P1-A 解码阶梯离线对照（numpy跨文档向量化两态DP）—— 流式内存安全版"""
import gzip, json, sys, re, bisect, os, gc
import numpy as np
sys.path.insert(0, "/tmp")
from decode_lib import load_split_gold_and_preds, char_counts, prf, iou_mae

split = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/decodecmp_{split}.json"
RUN = "/OMFG/duomotai-backup-20260924/tev4_storage/runs_q/full_Q_locate_fast_seed0_20261014"
EVALDIR = f"{RUN}/eval_test" if split == "test" else f"{RUN}/eval_step1433"
DATA = f"/OMFG/duomotai-backup-20260924/dataset_v2/dataset_v2_{split}.jsonl.gz"

CONST_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
SMOOTH_GRID = [4, 8, 16]
SENT = re.compile(r"[.!?。！？;；…]+[\s\"'”’）)»]*|[\n]+")
SMOOTH_K = int(os.environ.get("SMOOTH_K", "16"))
DP_CONST_C = float(os.environ.get("DP_CONST_C", "4.0"))

def sent_bounds(text):
    b = [0]
    for m in SENT.finditer(text):
        e = m.end()
        while e < len(text) and text[e] in " \t\"'”’）)»」』】":
            e += 1
        b.append(e)
    b.append(len(text))
    return sorted(set(b))

def snap_spans(spans, bounds, n):
    out = []
    for s, e in spans:
        i = bisect.bisect_right(bounds, s) - 1
        ns = bounds[max(0, i)]
        j = bisect.bisect_left(bounds, e)
        ne = bounds[j] if j < len(bounds) else n
        if ne <= ns:
            ns, ne = s, e
        out.append((ns, ne))
    return out

def runs_of(mask):
    return np.flatnonzero(np.diff(np.r_[0, mask.view(np.int8), 0]) != 0).reshape(-1, 2)

def smooth_mask(mask, k):
    if not mask.any():
        return mask
    keep = np.zeros_like(mask)
    for s, e in runs_of(mask):
        if e - s >= k:
            keep[s:e] = True
    holes = np.flatnonzero(np.diff(np.r_[0, (~keep).view(np.int8), 0]) != 0).reshape(-1, 2)
    out = keep.copy()
    for s, e in holes:
        if 0 < s and e < len(mask) and e - s < k:
            out[s:e] = True
    return out

def snap_mask_of(raw_spans_list, bounds, n):
    m = np.zeros(n, dtype=bool)
    for ns, ne in snap_spans(raw_spans_list, bounds, n):
        m[ns:ne] = True
    return m

def dp_batch(docs, use_a=True, penalty=None):
    B = len(docs); T = max(d["n"] + 1 for d in docs)
    EM = np.full((B, T), -1e18, dtype=np.float32)
    EH = np.full((B, T), -1e18, dtype=np.float32)
    AS = np.zeros((B, T), dtype=np.float32)
    AE = np.zeros((B, T), dtype=np.float32)
    for i, d in enumerate(docs):
        n = d["n"]; al = d["alpha"]
        EM[i, :n] = -np.logaddexp(0.0, -al)
        EH[i, :n] = -np.logaddexp(0.0, al)
        EM[i, n:] = -1e18; EH[i, n:] = 0.0
        if use_a:
            AE[i, :n + 1] = d["a"][:n + 1, 1]
            AS[i, :n + 1] = d["a"][:n + 1, 0]
            AE[i, n + 1:] = 0.0; AS[i, n + 1:] = 0.0
        else:
            AE[i, :] = -penalty; AS[i, :] = -penalty
    h = EH[:, 0].copy(); m = EM[:, 0].copy()
    back = np.zeros((B, T, 2), dtype=np.int8)
    for t in range(1, T):
        cand = m + AE[:, t]
        back[:, t, 0] = (cand >= h)
        back[:, t, 1] = (m >= h + AS[:, t])
        h = EH[:, t] + np.where(back[:, t, 0], cand, h)
        m = EM[:, t] + np.where(back[:, t, 1], m, h + AS[:, t])
    st = (m >= h).astype(np.int8)
    masks = np.zeros((B, T), dtype=bool)
    for t in range(T - 1, -1, -1):
        masks[:, t] = (st == 1)
        st = np.where(st == 1, back[:, t, 1], back[:, t, 0])
    return [masks[i, :docs[i]["n"]].copy() for i in range(B)]

def bucket_of(c):
    return "0-5%" if c < 0.05 else "5-20%" if c < 0.20 else "20-50%" if c < 0.50 else "50-80%" if c < 0.80 else "80-100%"

VARIANTS = ["raw", "snap", "smooth-4", "smooth-8", "smooth-16",
            "dp-const-0.0", "dp-const-0.25", "dp-const-0.5", "dp-const-1.0",
            "dp-const-2.0", "dp-const-4.0", "dp-full", "gated", "e2e"]
acc = {v: dict(P=[0, 0, 0], cov={b: [0, 0] for b in ("0-5%", "5-20%", "20-50%", "50-80%", "80-100%")},
               full=[0, 0, 0], hum=[0, 0, 0], e2e=[0, 0, 0],
               span=dict(pred=0, gold=0, exact=0, iou=0, err=0, ends=0)) for v in VARIANTS}

def flush_chunk(chunk, gold, preds, texts):
    if not chunk:
        return
    dp_res = {"dp-full": dp_batch(chunk, use_a=True)}
    for c in CONST_GRID:
        dp_res[f"dp-const-{c}"] = dp_batch(chunk, use_a=False, penalty=c)
    for i, d in enumerate(chunk):
        did = d["id"]; g = gold[did]; n = d["n"]; pred_label = d["pred"]
        gm = np.zeros(n, dtype=bool)
        for s, e in g["spans"]:
            gm[s:e] = True
        raw_mask = d["alpha"] >= 0
        allm = {"raw": raw_mask, "snap": snap_mask_of(d["raw_spans_list"], d["bounds"], n),
                "dp-full": dp_res["dp-full"][i]}
        for k in SMOOTH_GRID:
            allm[f"smooth-{k}"] = smooth_mask(raw_mask, k)
        for c in CONST_GRID:
            allm[f"dp-const-{c}"] = dp_res[f"dp-const-{c}"][i]
        gated = (np.zeros(n, dtype=bool) if pred_label == 0 else
                 np.ones(n, dtype=bool) if pred_label == 2 else allm["dp-full"])
        allm["gated"] = gated
        for v, mask in allm.items():
            A = acc[v]
            if g["label"] == 1:
                tp, fp, fn = char_counts(mask, gm)
                A["P"][0] += tp; A["P"][1] += fp; A["P"][2] += fn
                b = bucket_of(g["coverage"])
                A["cov"][b][0] += tp; A["cov"][b][1] += fn
                sp = iou_mae(sorted(runs_of(mask).tolist()), g["spans"])
                S = A["span"]
                S["pred"] += sp["n_pred"]; S["gold"] += sp["n_gold"]
                S["exact"] += sp["exact_tp"]; S["iou"] += sp["iou_tp"]
                S["err"] += sp["err"]; S["ends"] += sp["ends"]
            if g["label"] == 2:
                A["full"][0] += int(mask.sum()); A["full"][1] += n
            if g["label"] == 0:
                A["hum"][0] += int(mask.sum()); A["hum"][1] += n
        Ag = acc["gated"]
        if g["label"] == 1:
            Ag["e2e"][0] += int(gated.sum()); Ag["e2e"][1] += 0; Ag["e2e"][2] += n - int(gated.sum())
        elif g["label"] == 2:
            Ag["e2e"][0] += n; Ag["e2e"][1] += 0; Ag["e2e"][2] += 0
        else:
            Ag["e2e"][2] += n - int((~gated).sum()); Ag["e2e"][1] += int((~gated).sum()); Ag["e2e"][0] += 0

def main():
    print("streaming gold/preds...", flush=True)
    gold, preds, texts, arrays = load_split_gold_and_preds(
        split, EVALDIR, DATA, need_text=True, need_alpha=True)
    print(f"loaded {len(gold)} docs; precomputing bounds...", flush=True)
    for did in texts:
        texts[did] = sent_bounds(texts[did])
    gc.collect()
    CH_DOCS = 1000
    CH_CHARS = 1_500_000
    done = 0; chunk = []; chunk_chars = 0; chunk_max_n = 0
    rank_files = [(r, open(f"{EVALDIR}/rank{r}.jsonl", encoding="utf-8")) for r in range(4)]
    memmaps = {r: np.memmap(f"{EVALDIR}/rank{r}.f32", dtype="<f4", mode="r") for r in range(4)}

    def make_doc(did, rec):
        g = gold[did]; sl = rec["arrays"]; mm = memmaps[rec["rank_file"]]
        n = g["n"]
        off = sl["alpha"]["offset"]; sh = sl["alpha"]["shape"]
        alpha = np.asarray(mm[off:off + int(np.prod(sh))], dtype="<f4").reshape(sh).astype(np.float64)
        offa = sl["a"]["offset"]; sha = sl["a"]["shape"]
        aa = np.asarray(mm[offa:offa + int(np.prod(sha))], dtype="<f4").reshape(sha).astype(np.float64)
        m0 = alpha >= 0
        rspans = [tuple(x) for x in np.flatnonzero(np.diff(np.r_[0, m0.view(np.int8), 0]) != 0).reshape(-1, 2).tolist()]
        return dict(id=did, alpha=alpha, a=aa, n=n, label=g["label"], pred=rec["prediction"],
                    raw_spans_list=rspans, bounds=texts[did])

    for r, fh in rank_files:
        for line in fh:
            rec = json.loads(line)
            did = rec["doc_id"]
            n_new = gold[did]["n"]
            if chunk and (chunk_chars + n_new > CH_CHARS or n_new > 1_000_000
                          or len(chunk) * chunk_max_n * 4 > 60_000_000):
                make = [make_doc(did2, rec2) for did2, rec2 in chunk]
                flush_chunk(make, gold, preds, texts)
                del make
                done += len(chunk); chunk = []; chunk_chars = 0; chunk_max_n = 0
                print(f"progress {done}/{len(gold)}", flush=True)
            rec["rank_file"] = r
            chunk.append((did, rec))
            chunk_chars += n_new; chunk_max_n = max(chunk_max_n, n_new)
        if chunk:
            make = [make_doc(did2, rec2) for did2, rec2 in chunk]
            flush_chunk(make, gold, preds, texts)
            del make
            done += len(chunk); chunk = []; chunk_chars = 0; chunk_max_n = 0
            print(f"progress {done}/{len(gold)}", flush=True)
    for r, fh in rank_files:
        fh.close()

    out = {"scope": split, "tuned_on": "val", "frozen": {"snap": "union-out", "smooth_k": SMOOTH_K,
                                                          "dp_const_c": DP_CONST_C}}
    for v, A in acc.items():
        e = {"char_F1_PARTIAL": prf(*A["P"]),
             "PARTIAL_recall_by_coverage": {b: round(vv[0] / max(1, vv[0] + vv[1]), 4) for b, vv in A["cov"].items()},
             "FULL_char_recall": round(A["full"][0] / max(1, A["full"][1]), 4),
             "HUMAN_char_FP_rate": round(A["hum"][0] / max(1, A["hum"][1]), 4)}
        S = A["span"]
        if S["pred"] or S["gold"]:
            e["span_exact_F1"] = prf(S["exact"], S["pred"] - S["exact"], S["gold"] - S["exact"])
            e["span_iou50_F1"] = prf(S["iou"], S["pred"] - S["iou"], S["gold"] - S["iou"])
            e["boundary_MAE"] = round(S["err"] / max(1, S["ends"]), 2)
        if v == "gated":
            e["pipeline_char_F1_all_docs"] = prf(*A["e2e"])
        out[v] = e
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("WROTE", OUT)

if __name__ == "__main__":
    main()
