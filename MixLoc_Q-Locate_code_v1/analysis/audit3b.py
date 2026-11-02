# -*- coding: utf-8 -*-
"""Audit-3B addendum: (1) single-span decoder WITHOUT the a field (Kadane on
alpha only) — isolates the boundary field's contribution inside the best
decoder; (2) end-channel matched negatives at after-punctuation positions
(position-type-matched test for the end field). CPU only."""
import gzip, json, os, sys, time, re
import numpy as np

BASE = "/OMFG/duomotai-backup-20260924"
RUN = f"{BASE}/tev4_storage/runs_q/full_Q_locate_fast_seed0_20261014"
DATA = {s: f"{BASE}/dataset_v2/dataset_v2_{s}.jsonl.gz" for s in ("val", "test")}
EVAL = {"test": f"{RUN}/eval_test", "val": f"{RUN}/eval_step1433"}
OUT = f"{BASE}/tev4_storage/analysis_audit3_20261018"
BETA = 0.25
CONTROLLED = {"deepseek-v4-flash", "minimax-m2.5", "gemini-2.5-flash"}
SENT = re.compile(r"[.!?。！？;；…]+[\s\"'”’）)»]*|[\n]+")
PUNCT = re.compile(r"[.!?。！？;；…]+")
CH_CHARS = 1_200_000
T0 = time.time()


def spans_list(mask):
    ch = np.diff(np.r_[0, mask.view(np.int8), 0])
    return [(int(a), int(b)) for a, b in
            np.flatnonzero(ch != 0).reshape(-1, 2).tolist()]


def prf(tp, fp, fn):
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return round(float(2 * p * r / max(1e-12, p + r)), 4)


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
    return dict(npred=len(pred), ngold=len(gold), exact=exact, iou=iou, err=err, ends=ends)


def single_span_batch(docs, use_a=True, beta=BETA):
    B = len(docs)
    W = max(d[0] for d in docs) + 1
    D = np.full((B, W), -np.inf)
    ASx = np.zeros((B, W)) if not use_a else np.full((B, W), -np.inf)
    AEx = np.zeros((B, W)) if not use_a else np.full((B, W), -np.inf)
    for i, (n, al, aa) in enumerate(docs):
        D[i, :n] = (-np.logaddexp(0.0, -al)) - (-np.logaddexp(0.0, al))
        if use_a:
            ASx[i, :n] = beta * aa[:n, 0]
            AEx[i, 1:n + 1] = beta * aa[1:n + 1, 1]
    C = np.full(B, -np.inf)
    ptr = np.zeros((B, W), dtype=np.int32)
    best = np.zeros(B)
    best_e = np.zeros(B, dtype=np.int64)
    for t in range(1, W):
        cont = C + D[:, t - 1]
        rest = ASx[:, t - 1] + D[:, t - 1]
        restart = rest >= cont
        C = np.where(restart, rest, cont)
        ptr[:, t] = np.where(restart, t - 1, ptr[:, t - 1])
        cand = C + AEx[:, t]
        upd = cand > best
        best = np.where(upd, cand, best)
        best_e = np.where(upd, t, best_e)
    out = []
    for i, (n, al, aa) in enumerate(docs):
        m = np.zeros(n, dtype=bool)
        if best[i] > 0:
            s = int(ptr[i, best_e[i]]); e = int(best_e[i])
            if 0 <= s < e <= n:
                m[s:e] = True
        out.append(m)
    return out


def run_single(split):
    evaldir = EVAL[split]
    gold = {}
    with gzip.open(DATA[split], "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["label"] != "PARTIAL_AI":
                continue
            gold[r["row_id_v2"]] = dict(
                n=len(r["text"]),
                spans=[(int(s), int(e)) for s, e in (r.get("machine_spans") or [])])
    mms = {r: np.memmap(f"{evaldir}/rank{r}.f32", dtype="<f4", mode="r") for r in range(4)}
    acc = {v: dict(char=[0, 0, 0], span=dict(npred=0, ngold=0, exact=0, iou=0, err=0, ends=0))
           for v in ("single_a", "single_noa")}
    chunk = []; chunk_chars = 0; chunk_max = 0; ndocs = 0

    def flush():
        nonlocal chunk, chunk_chars, chunk_max, ndocs
        if not chunk:
            return
        docs = []
        for did, rec, r in chunk:
            g = gold[did]; n = g["n"]; sl = rec["arrays"]; mm = mms[r]
            al = np.asarray(mm[sl["alpha"]["offset"]:sl["alpha"]["offset"] + n],
                            dtype="<f4").reshape(n).astype(np.float64)
            aa = np.asarray(mm[sl["a"]["offset"]:sl["a"]["offset"] + (n + 1) * 2],
                            dtype="<f4").reshape(n + 1, 2).astype(np.float64)
            docs.append((did, al, aa))
        base = [(d[1].shape[0], d[1], d[2]) for d in docs]
        masks = {"single_a": single_span_batch(base, use_a=True),
                 "single_noa": single_span_batch(base, use_a=False)}
        for j, (did, al, aa) in enumerate(docs):
            g = gold[did]; n = g["n"]
            gm = np.zeros(n, dtype=bool)
            for s, e in g["spans"]:
                gm[s:e] = True
            for v, m2 in (("single_a", masks["single_a"][j]),
                          ("single_noa", masks["single_noa"][j])):
                A = acc[v]
                A["char"][0] += int((m2 & gm).sum())
                A["char"][1] += int((m2 & ~gm).sum())
                A["char"][2] += int((gm & ~m2).sum())
                M = span_metrics(spans_list(m2), g["spans"])
                for k in ("npred", "ngold", "exact", "iou", "err", "ends"):
                    A["span"][k] += M[k]
        ndocs += len(chunk)
        chunk = []; chunk_chars = 0; chunk_max = 0
        print(f"[single/{split}] {ndocs}/{len(gold)} {(time.time()-T0)/60:.1f}min", flush=True)

    for r in range(4):
        with open(f"{evaldir}/rank{r}.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                did = rec["doc_id"]
                if did not in gold:
                    continue
                n = gold[did]["n"]
                if chunk and (chunk_chars + n > CH_CHARS or n > 1_000_000
                              or (len(chunk) + 1) * max(chunk_max, n) * 40 > 500_000_000):
                    flush()
                chunk.append((did, rec, r))
                chunk_chars += n; chunk_max = max(chunk_max, n)
    flush()
    res = {}
    for v, A in acc.items():
        S = A["span"]
        res[v] = dict(char_f1=prf(*A["char"]),
                      iou50_f1=prf(S["iou"], S["npred"] - S["iou"], S["ngold"] - S["iou"]),
                      exact_f1=prf(S["exact"], S["npred"] - S["exact"], S["ngold"] - S["exact"]),
                      mae=round(S["err"] / max(1, S["ends"]), 2),
                      n_matched=S["iou"], n_pred=S["npred"])
    print(f"[single/{split}]", json.dumps(res), flush=True)
    return res


def end_channel_afterpunct():
    d = EVAL["val"]
    gold = {}
    with gzip.open(DATA["val"], "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["label"] != "PARTIAL_AI":
                continue
            gold[r["row_id_v2"]] = dict(
                n=len(r["text"]), text=r["text"],
                spans=[(int(s), int(e)) for s, e in (r.get("machine_spans") or [])])
    mms = {r: np.memmap(f"{d}/rank{r}.f32", dtype="<f4", mode="r") for r in range(4)}
    pos_all, neg_all = [], []
    pos_ctrl, neg_ctrl = [], []
    pos_ext, neg_ext = [], []
    n_docs = 0
    gen_of = {}
    # generator from data (same pass as gold load above missed it) — reload light
    with gzip.open(DATA["val"], "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["label"] == "PARTIAL_AI":
                gen_of[r["row_id_v2"]] = r.get("generator") or "UNKNOWN"
    for r in range(4):
        with open(f"{d}/rank{r}.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                did = rec["doc_id"]
                if did not in gold:
                    continue
                g = gold[did]; n = g["n"]; text = g["text"]
                sl = rec["arrays"]["a"]; off = sl["offset"]
                aa = np.asarray(mms[r][off:off + (n + 1) * 2], dtype="<f4").reshape(n + 1, 2).astype(np.float64)
                afterpunct = set()
                for m in PUNCT.finditer(text):
                    afterpunct.add(m.end())
                gold_e = set(e for s, e in g["spans"] if 0 < e <= n)
                ap_arr = np.array(sorted(x for x in afterpunct if x <= n), dtype=np.int64)
                if len(ap_arr) == 0:
                    continue
                pos_idx = np.array(sorted(gold_e & afterpunct), dtype=np.int64)
                neg_idx = ap_arr[~np.isin(ap_arr, sorted(gold_e))]
                ctrl = gen_of.get(did) in CONTROLLED
                if len(pos_idx):
                    pos_all.append(aa[pos_idx, 1])
                    (pos_ctrl if ctrl else pos_ext).append(aa[pos_idx, 1])
                if len(neg_idx):
                    neg_all.append(aa[neg_idx, 1])
                    (neg_ctrl if ctrl else neg_ext).append(aa[neg_idx, 1])
                n_docs += 1

    def auc(pos_l, neg_l):
        pos = np.concatenate(pos_l) if pos_l else np.array([])
        neg = np.concatenate(neg_l) if neg_l else np.array([])
        if len(pos) == 0 or len(neg) == 0:
            return None
        ns = np.sort(neg)
        hi = np.searchsorted(ns, pos); lo = np.searchsorted(ns, pos, side="left")
        return dict(n_pos=int(len(pos)), n_neg=int(len(neg)),
                    auc=round(float(((lo + 0.5 * (hi - lo)) / len(ns)).mean()), 4),
                    pos_mean=round(float(pos.mean()), 3), neg_mean=round(float(neg.mean()), 3))

    res = dict(docs=n_docs, all=auc(pos_all, neg_all),
               controlled=auc(pos_ctrl, neg_ctrl), external=auc(pos_ext, neg_ext))
    print("[endpunct]", json.dumps(res), flush=True)
    return res


def main():
    single = {s: run_single(s) for s in ("val", "test")}
    ep = end_channel_afterpunct()
    json.dump(dict(single_span_ablation=single, end_channel_afterpunct=ep),
              open(f"{OUT}/audit3b_addendum.json", "w"), ensure_ascii=False, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
