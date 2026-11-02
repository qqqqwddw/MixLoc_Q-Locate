# -*- coding: utf-8 -*-
"""Audit-3 (CPU only, per GPT investigation directives 2026-10-18):

A) Span-count audit: train/val/test x {0,1,2,3,>=4 spans}, stratified by
   generator (controlled 3 vs external) and language. Verifies whether Test
   PARTIAL is (nearly) all single-span.
B) Sentence-boundary-matched hard negatives (Val, step1433 exports):
   a-field AUC at gold boundaries vs OTHER sentence boundaries of the same
   doc (start/end channels), + gold<->sentence coincidence rates,
   controlled vs external breakdown.
C) Boundary-field perturbation + single-span decoder (Val & Test, PARTIAL):
   ref(dp-full) / a_zero / a_shuffle / a_shift32 / start_only / end_only /
   single_span(Kadane O(n)) — metrics stratified by gold span count {1, >=2}.
Outputs -> /OMFG/.../analysis_audit3_20261018/
"""
import gzip, json, os, sys, time, hashlib, socket, re
import numpy as np

SMOKE = bool(int(os.environ.get("SMOKE", "0")))
BASE = "/OMFG/duomotai-backup-20260924"
RUN = f"{BASE}/tev4_storage/runs_q/full_Q_locate_fast_seed0_20261014"
DATA = {s: f"{BASE}/dataset_v2/dataset_v2_{s}.jsonl.gz" for s in ("train", "val", "test")}
EVAL = {"test": f"{RUN}/eval_test", "val": f"{RUN}/eval_step1433"}
OUT = f"{BASE}/tev4_storage/analysis_audit3_20261018"
BETA = 0.25
CONTROLLED = {"deepseek-v4-flash", "minimax-m2.5", "gemini-2.5-flash"}
SENT = re.compile(r"[.!?。！？;；…]+[\s\"'”’）)»]*|[\n]+")
CH_CHARS = 1_200_000
MEM_BUDGET = 500_000_000
T0 = time.time()
STATS = {"exit_status": None}


def sha256_file(path, block=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sent_bounds(text):
    b = [0]
    for m in SENT.finditer(text):
        e = m.end()
        while e < len(text) and text[e] in " \t\"'”’）)»」』】":
            e += 1
        b.append(e)
    b.append(len(text))
    return np.array(sorted(set(b)), dtype=np.int64)


def runs_of(mask):
    return np.flatnonzero(np.diff(np.r_[0, mask.view(np.int8), 0]) != 0).reshape(-1, 2)


def spans_list(mask):
    return [(int(a), int(b)) for a, b in runs_of(mask).tolist()]


def prf(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return dict(p=round(float(p), 4), r=round(float(r), 4),
                f1=round(float(2 * p * r / max(1e-12, p + r)), 4))


def span_metrics(pred_spans, gold_spans):
    pred = [tuple(x) for x in pred_spans]
    gold = [tuple(x) for x in gold_spans]
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


def auc_vs_neg(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    neg_s = np.sort(neg)
    hi = np.searchsorted(neg_s, pos)
    lo = np.searchsorted(neg_s, pos, side="left")
    return round(float(((lo + 0.5 * (hi - lo)) / len(neg_s)).mean()), 4)


# ---------------- A: span-count audit ----------------

def bucket_k(k):
    return "0" if k == 0 else "1" if k == 1 else "2" if k == 2 else "3" if k == 3 else ">=4"


def part_a():
    out = {}
    for split in ("train", "val", "test"):
        overall = {b: 0 for b in ("0", "1", "2", "3", ">=4")}
        by_gen = {}
        by_ctrl = {"controlled": {b: 0 for b in overall}, "external": {b: 0 for b in overall}}
        by_lang_multi = {}
        n_partial = 0; total_spans = 0
        with gzip.open(DATA[split], "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["label"] != "PARTIAL_AI":
                    continue
                gen = r.get("generator") or "UNKNOWN"
                lang = r.get("language") or "UNKNOWN"
                k = len(r.get("machine_spans") or [])
                b = bucket_k(k)
                n_partial += 1; total_spans += k
                overall[b] += 1
                by_gen.setdefault(gen, {x: 0 for x in overall})[b] += 1
                by_ctrl["controlled" if gen in CONTROLLED else "external"][b] += 1
                if k >= 2:
                    by_lang_multi[lang] = by_lang_multi.get(lang, 0) + 1
                if SMOKE and n_partial >= 3000:
                    break
        out[split] = dict(
            n_partial=n_partial, total_spans=total_spans,
            avg_spans_per_doc=round(total_spans / max(1, n_partial), 4),
            overall=overall,
            single_span_frac=round(overall["1"] / max(1, n_partial), 4),
            by_controlled=by_ctrl,
            top_generators=sorted(by_gen.items(), key=lambda kv: -sum(kv[1].values()))[:12],
            multi_span_by_language_top10=sorted(by_lang_multi.items(), key=lambda kv: -kv[1])[:10])
        print(f"[A/{split}] partial={n_partial} spans={total_spans} "
              f"avg={out[split]['avg_spans_per_doc']} single_frac={out[split]['single_span_frac']}",
              flush=True)
    json.dump(out, open(f"{OUT}/span_count_audit.json", "w"), ensure_ascii=False, indent=1)
    return out


# ---------------- B: sentence-matched negatives (val) ----------------

def part_b():
    gold = {}
    with gzip.open(DATA["val"], "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["label"] != "PARTIAL_AI":
                continue
            gold[r["row_id_v2"]] = dict(
                n=len(r["text"]), bounds=sent_bounds(r["text"]),
                spans=[(int(s), int(e)) for s, e in (r.get("machine_spans") or [])],
                gen=r.get("generator") or "UNKNOWN")
            if SMOKE and len(gold) >= 300:
                break
    acc = {grp: {ch: {p: [] for p in ("pos", "neg")} for ch in ("start", "end")}
           for grp in ("all", "controlled", "external")}
    coinc = {ch: [0, 0] for ch in ("start", "end")}
    d = EVAL["val"]
    mms = {r: np.memmap(f"{d}/rank{r}.f32", dtype="<f4", mode="r") for r in range(4)}
    ndocs = 0
    for r in range(4):
        with open(f"{d}/rank{r}.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                did = rec["doc_id"]
                if did not in gold:
                    continue
                g = gold[did]; n = g["n"]
                sl = rec["arrays"]["a"]; off = sl["offset"]
                aa = np.asarray(mms[r][off:off + (n + 1) * 2], dtype="<f4").reshape(n + 1, 2).astype(np.float64)
                sent_starts = set(int(b) for b in g["bounds"] if b < n)
                sent_ends = set(int(b) for b in g["bounds"] if b > 0)
                gold_s = set(s for s, e in g["spans"] if 0 <= s < n)
                gold_e = set(e for s, e in g["spans"] if 0 < e <= n)
                coinc["start"][0] += len(gold_s & sent_starts); coinc["start"][1] += len(gold_s)
                coinc["end"][0] += len(gold_e & sent_ends); coinc["end"][1] += len(gold_e)
                grp = "controlled" if g["gen"] in CONTROLLED else "external"
                for target in ("all", grp):
                    A = acc[target]
                    sstart_arr = np.array(sorted(sent_starts), dtype=np.int64)
                    send_arr = np.array(sorted(sent_ends), dtype=np.int64)
                    pos_s = np.array(sorted(gold_s & sent_starts), dtype=np.int64)
                    neg_s = sstart_arr[~np.isin(sstart_arr, sorted(gold_s))]
                    pos_e = np.array(sorted(gold_e & sent_ends), dtype=np.int64)
                    neg_e = send_arr[~np.isin(send_arr, sorted(gold_e))]
                    if len(pos_s):
                        A["start"]["pos"].append(aa[pos_s, 0])
                    if len(neg_s):
                        A["start"]["neg"].append(aa[neg_s, 0])
                    if len(pos_e):
                        A["end"]["pos"].append(aa[pos_e, 1])
                    if len(neg_e):
                        A["end"]["neg"].append(aa[neg_e, 1])
                ndocs += 1
    res = dict(docs=ndocs,
               coincidence={ch: dict(at_sentence_boundary=coinc[ch][0], total=coinc[ch][1],
                                     frac=round(coinc[ch][0] / max(1, coinc[ch][1]), 4))
                            for ch in ("start", "end")})
    for grp, A in acc.items():
        res[grp] = {}
        for ch in ("start", "end"):
            pos = np.concatenate(A[ch]["pos"]) if A[ch]["pos"] else np.array([])
            neg = np.concatenate(A[ch]["neg"]) if A[ch]["neg"] else np.array([])
            res[grp][ch] = dict(n_pos=int(len(pos)), n_neg=int(len(neg)),
                                auc_sent_matched=auc_vs_neg(pos, neg),
                                pos_mean=round(float(pos.mean()), 3) if len(pos) else None,
                                neg_mean=round(float(neg.mean()), 3) if len(neg) else None)
    print("[B]", json.dumps(res, indent=1), flush=True)
    json.dump(res, open(f"{OUT}/sentence_matched_negatives.json", "w"), ensure_ascii=False, indent=1)
    return res


# ---------------- C: perturbation + single-span (val & test) ----------------

def dp_batch(docs, const=None, beta=BETA):
    B = len(docs)
    W = max(d[0] for d in docs) + 1
    EH = np.zeros((B, W)); EM = np.zeros((B, W))
    AS = np.zeros((B, W)); AE = np.zeros((B, W))
    for i, (n, al, aa) in enumerate(docs):
        EM[i, :n] = -np.logaddexp(0.0, -al); EM[i, n:] = -np.inf
        EH[i, :n] = -np.logaddexp(0.0, al); EH[i, n:] = 0.0
        if const is None:
            AS[i, :n] = beta * aa[:n, 0]
            AE[i, :n] = beta * aa[:n, 1]
            AE[i, n] = beta * aa[n, 1]
        else:
            AS[i, :n + 1] = -const
            AE[i, :n + 1] = -const
        AE[i, n + 1:] = -np.inf
    h = EH[:, 0].copy()
    m = EM[:, 0] + AS[:, 0]
    back = np.zeros((B, W, 2), dtype=np.int8)
    idx = np.arange(B)
    with np.errstate(invalid="ignore"):
        for t in range(1, W):
            cand_mh = m + AE[:, t]
            cand_hm = h + AS[:, t]
            b0 = cand_mh >= h
            b1 = m >= cand_hm
            h = np.where(b0, cand_mh, h) + EH[:, t]
            m = np.where(b1, m, cand_hm) + EM[:, t]
            back[:, t, 0] = b0
            back[:, t, 1] = b1
        st = np.zeros(B, dtype=np.int64)
        masks = np.zeros((B, W), dtype=bool)
        for t in range(W - 1, -1, -1):
            masks[:, t] = st == 1
            st = back[idx, t, st].astype(np.int64)
    return [masks[i, :docs[i][0]].copy() for i in range(B)]


def single_span_batch(docs, beta=BETA):
    """O(n) Kadane: best single half-open span [s,e) under engine score
    semantics: beta*a[s,0] + sum_{i in [s,e)}(sM-sH) + beta*a[e,1], vs no-span."""
    B = len(docs)
    W = max(d[0] for d in docs) + 1
    D = np.full((B, W), -np.inf)
    ASx = np.full((B, W), -np.inf)
    AEx = np.full((B, W), -np.inf)
    for i, (n, al, aa) in enumerate(docs):
        sM = -np.logaddexp(0.0, -al)
        sH = -np.logaddexp(0.0, al)
        D[i, :n] = sM - sH
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


VARIANTS = ["ref", "a_zero", "a_shuffle", "a_shift32", "start_only", "end_only", "single_span"]


def part_c(split):
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
            if SMOKE and len(gold) >= 400:
                break
    rng = np.random.default_rng(0)
    vars_ = {v: dict(char=[0, 0, 0], span=dict(npred=0, ngold=0, exact=0, iou=0, err=0, ends=0))
             for v in VARIANTS}
    strat = {sb: {v: dict(char=[0, 0, 0], span=dict(npred=0, ngold=0, exact=0, iou=0, err=0, ends=0))
                  for v in VARIANTS} for sb in ("sp1", "sp2plus")}
    mms = {r: np.memmap(f"{evaldir}/rank{r}.f32", dtype="<f4", mode="r") for r in range(4)}
    ndocs = 0
    chunk = []; chunk_chars = 0; chunk_max = 0

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
        masks = {}
        masks["ref"] = dp_batch(base, const=None)
        masks["a_zero"] = dp_batch(base, const=0.0)
        shuf = [(n, al, aa[rng.permutation(n + 1)]) for (n, al, aa) in base]
        masks["a_shuffle"] = dp_batch(shuf, const=None)
        shifted = [(n, al, np.roll(aa, 32, axis=0)) for (n, al, aa) in base]
        masks["a_shift32"] = dp_batch(shifted, const=None)
        st_only = [(n, al, np.stack([aa[:, 0], np.zeros(n + 1)], axis=1)) for (n, al, aa) in base]
        masks["start_only"] = dp_batch(st_only, const=None)
        ed_only = [(n, al, np.stack([np.zeros(n + 1), aa[:, 1]], axis=1)) for (n, al, aa) in base]
        masks["end_only"] = dp_batch(ed_only, const=None)
        masks["single_span"] = single_span_batch(base)
        for j, (did, al, aa) in enumerate(docs):
            g = gold[did]; n = g["n"]
            gm = np.zeros(n, dtype=bool)
            for s, e in g["spans"]:
                gm[s:e] = True
            sb = "sp1" if len(g["spans"]) == 1 else "sp2plus"
            for v in VARIANTS:
                m2 = masks[v][j]
                for T in (vars_[v], strat[sb][v]):
                    T["char"][0] += int((m2 & gm).sum())
                    T["char"][1] += int((m2 & ~gm).sum())
                    T["char"][2] += int((gm & ~m2).sum())
                    S = T["span"]; M = span_metrics(spans_list(m2), g["spans"])
                    for k in ("npred", "ngold", "exact", "iou", "err", "ends"):
                        S[k] += M[k]
        ndocs += len(chunk)
        chunk = []; chunk_chars = 0; chunk_max = 0
        print(f"[C/{split}] {ndocs}/{len(gold)} docs {(time.time()-T0)/60:.1f}min", flush=True)

    for r in range(4):
        with open(f"{evaldir}/rank{r}.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                did = rec["doc_id"]
                if did not in gold:
                    continue
                n = gold[did]["n"]
                if chunk and (chunk_chars + n > CH_CHARS or n > 1_000_000
                              or (len(chunk) + 1) * max(chunk_max, n) * 40 > MEM_BUDGET):
                    flush()
                chunk.append((did, rec, r))
                chunk_chars += n; chunk_max = max(chunk_max, n)
                if SMOKE and ndocs + len(chunk) >= 400:
                    break
    flush()

    def fin(T):
        S = T["span"]
        return dict(char_f1=prf(*T["char"])["f1"],
                    iou50_f1=prf(S["iou"], S["npred"] - S["iou"], S["ngold"] - S["iou"])["f1"],
                    exact_f1=prf(S["exact"], S["npred"] - S["exact"], S["ngold"] - S["exact"])["f1"],
                    mae=round(S["err"] / max(1, S["ends"]), 2),
                    n_matched=S["iou"], n_pred=S["npred"], n_gold=S["ngold"])

    res = dict(split=split, docs=ndocs,
               overall={v: fin(T) for v, T in vars_.items()},
               by_span_count={sb: {v: fin(T) for v, T in st.items()} for sb, st in strat.items()})
    print(f"[C/{split}]", json.dumps(res["overall"], indent=1), flush=True)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    a = part_a()
    b = part_b()
    c = {s: part_c(s) for s in ("val", "test")}
    json.dump(dict(perturbation_and_single_span=c),
              open(f"{OUT}/perturbation_single_span.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    try:
        main()
        STATS["exit_status"] = 0
    except Exception:
        import traceback
        traceback.print_exc()
        STATS["exit_status"] = 1
    finally:
        try:
            json.dump(dict(exit_status=STATS["exit_status"],
                           runtime_min=round((time.time() - T0) / 60, 1),
                           host=socket.gethostname(), outdir=OUT, smoke=SMOKE,
                           script_sha256=sha256_file(os.path.abspath(__file__)),
                           checkpoint_sha256=sha256_file(f"{RUN}/checkpoint.pt")),
                      open(f"{OUT}/run_manifest.json", "w"), indent=1)
        except Exception as e2:
            print("manifest failed:", e2)
        print("MANIFEST WRITTEN")
    sys.exit(STATS["exit_status"] or 0)
