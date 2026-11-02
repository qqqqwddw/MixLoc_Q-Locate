# -*- coding: utf-8 -*-
"""P1-A2 FULL: full-scale decoding reconciliation + corrected ladder (CPU, 42234).

Protocol (user-approved 2026-10-18):
  P0  official DP full replay vs stored decoded_spans (val+test, every doc)
  P0  fixed-DP full consistency (span-set equality count)
  P0  const-penalty DP grid: tuned on VAL ONLY, frozen, applied to TEST
  P0  ladder re-eval: raw / snap / smooth{4,8,16} / dp-const / dp-full / gated / e2e
  P1  coverage + generator + language stratification (PARTIAL, new defs)
  P1  diagnostics: AI char ratio, spans/doc, all-H/all-M; MAE always with
      (N_matched, N_pred, N_gold)
Constraints: independent output dir (no canonical overwrite); OMP threads capped;
             smoke mode first (SMOKE=1); manifest with hashes/config/exit status;
             negative results preserved.
"""
import gzip, json, os, sys, time, hashlib, platform, re, socket
import numpy as np

SMOKE = bool(int(os.environ.get("SMOKE", "0")))
RUN = "/OMFG/duomotai-backup-20260924/tev4_storage/runs_q/full_Q_locate_fast_seed0_20261014"
EVAL = {"test": f"{RUN}/eval_test", "val": f"{RUN}/eval_step1433"}
DATA = {"test": "/OMFG/duomotai-backup-20260924/dataset_v2/dataset_v2_test.jsonl.gz",
        "val": "/OMFG/duomotai-backup-20260924/dataset_v2/dataset_v2_val.jsonl.gz"}
OUTDIR = "/OMFG/duomotai-backup-20260924/tev4_storage/analysis_p1a2_full_20261018"
BETA = 0.25
CONST_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
SMOOTH_GRID = [4, 8, 16]
MAXDOC_SMOKE = 600
CH_CHARS = 1_200_000
MEM_BUDGET = 500_000_000
LAB = {"HUMAN": 0, "PARTIAL_AI": 1, "FULL_AI": 2}
SENT = re.compile(r"[.!?。！？;；…]+[\s\"'”’）)»]*|[\n]+")

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


def snap_spans(spans, bounds, n):
    out = []
    for s, e in spans:
        i = int(np.searchsorted(bounds, s, side="right")) - 1
        ns = int(bounds[max(0, i)])
        j = int(np.searchsorted(bounds, e, side="left"))
        ne = int(bounds[j]) if j < len(bounds) else n
        if ne <= ns:
            ns, ne = s, e
        out.append((ns, ne))
    return out


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


def mask_of(spans, n):
    m = np.zeros(n, dtype=bool)
    for s, e in spans:
        m[s:e] = True
    return m


def dp_batch(docs, const=None, beta=BETA):
    """Fixed batch DP, engine-equivalent semantics, float64.
    docs: list of (n, alpha[f8], a[f8 (n+1,2)] or None when const).
    M0 = s_M(0)+beta*a[0,0]; M->H at t uses beta*a[t,1]; H->M at t uses OLD h
    + beta*a[t,0]; terminal beta*a[n,1] (or -const); machine-preferred ties."""
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
            cand_hm = h + AS[:, t]           # OLD h — the fix
            b0 = cand_mh >= h
            b1 = m >= cand_hm
            h = np.where(b0, cand_mh, h) + EH[:, t]
            m = np.where(b1, m, cand_hm) + EM[:, t]
            back[:, t, 0] = b0
            back[:, t, 1] = b1
        st = np.zeros(B, dtype=np.int64)     # virtual terminal position is H
        masks = np.zeros((B, W), dtype=bool)
        for t in range(W - 1, -1, -1):
            masks[:, t] = st == 1
            st = back[idx, t, st].astype(np.int64)
    return [masks[i, :docs[i][0]].copy() for i in range(B)]


def new_var():
    return dict(char=[0, 0, 0],
                span=dict(npred=0, ngold=0, exact=0, iou=0, err=0, ends=0),
                diag=dict(docs=0, ai=0, tot=0, nsp=0, allH=0, allM=0),
                full=[0, 0], hum=[0, 0])


def bucket_of(c):
    return ("0-5%" if c < 0.05 else "5-20%" if c < 0.20 else "20-50%" if c < 0.50
            else "50-80%" if c < 0.80 else "80-100%")


def acc_variant(V, mask, spans, gm, gspans, n):
    V["char"][0] += int((mask & gm).sum())
    V["char"][1] += int((mask & ~gm).sum())
    V["char"][2] += int((gm & ~mask).sum())
    S = V["span"]; M = span_metrics(spans, gspans)
    for k in ("npred", "ngold", "exact", "iou", "err", "ends"):
        S[k] += M[k]
    D = V["diag"]; D["docs"] += 1; D["tot"] += n
    D["ai"] += int(mask.sum()); D["nsp"] += len(spans)
    D["allH"] += int(not mask.any()); D["allM"] += int(mask.all())


def strat_add(store, key, var, mask, spans, gm, gspans):
    A = store.setdefault(key, {}).setdefault(
        var, dict(char=[0, 0, 0], span=dict(npred=0, ngold=0, exact=0, iou=0, err=0, ends=0)))
    A["char"][0] += int((mask & gm).sum())
    A["char"][1] += int((mask & ~gm).sum())
    A["char"][2] += int((gm & ~mask).sum())
    S = A["span"]; M = span_metrics(spans, gspans)
    for k in ("npred", "ngold", "exact", "iou", "err", "ends"):
        S[k] += M[k]


def finalize_var(V):
    S = V["span"]; D = V["diag"]
    out = dict(char=prf(*V["char"]),
               span=dict(exact_f1=prf(S["exact"], S["npred"] - S["exact"], S["ngold"] - S["exact"])["f1"],
                         iou50_f1=prf(S["iou"], S["npred"] - S["iou"], S["ngold"] - S["iou"])["f1"],
                         mae=round(S["err"] / max(1, S["ends"]), 2),
                         n_matched=S["iou"], n_pred=S["npred"], n_gold=S["ngold"]),
               diag=dict(ai_ratio=round(D["ai"] / max(1, D["tot"]), 4),
                         spans_per_doc=round(D["nsp"] / max(1, D["docs"]), 2),
                         allH=D["allH"], allM=D["allM"]))
    if V["full"][1]:
        out["FULL_char_recall"] = round(V["full"][0] / max(1, V["full"][1]), 4)
    if V["hum"][1]:
        out["HUMAN_char_FP_rate"] = round(V["hum"][0] / max(1, V["hum"][1]), 4)
    return out


def finalize_strat(sk):
    return {g2: {v: dict(char=prf(*A["char"]),
                         span=dict(exact_f1=prf(A["span"]["exact"], A["span"]["npred"] - A["span"]["exact"],
                                                 A["span"]["ngold"] - A["span"]["exact"])["f1"],
                                   iou50_f1=prf(A["span"]["iou"], A["span"]["npred"] - A["span"]["iou"],
                                                A["span"]["ngold"] - A["span"]["iou"])["f1"],
                                   mae=round(A["span"]["err"] / max(1, A["span"]["ends"]), 2),
                                   n_matched=A["span"]["iou"], n_pred=A["span"]["npred"],
                                   n_gold=A["span"]["ngold"]))
                 for v, A in gv.items()} for g2, gv in sk.items()}


def process_split(split, frozen_const=None):
    evaldir = EVAL[split]
    gold = {}
    with gzip.open(DATA[split], "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            did = r["row_id_v2"]
            label = r["label"]; n = len(r["text"])
            spans = ([(int(s), int(e)) for s, e in (r.get("machine_spans") or [])]
                     if label == "PARTIAL_AI" else [])
            g = dict(label=LAB[label], n=n, spans=spans,
                     cov=(sum(e - s for s, e in spans) / n) if (label == "PARTIAL_AI" and n) else None,
                     gen=r.get("generator") or "UNKNOWN", lang=r.get("language") or "UNKNOWN")
            if label == "PARTIAL_AI":
                g["bounds"] = sent_bounds(r["text"])
            gold[did] = g
    print(f"[{split}] gold loaded {len(gold)} docs {(time.time()-T0):.0f}s", flush=True)

    const_names = ([f"dp-const-{c}" for c in CONST_GRID] if frozen_const is None
                   else [f"dp-const-{frozen_const}"])
    vars_ = {v: new_var() for v in (["raw", "snap"] + [f"smooth-{k}" for k in SMOOTH_GRID]
                                    + ["dp-full", "gated"] + const_names)}
    strat = {"coverage": {}, "generator": {}, "language": {}}
    recon = dict(docs=0, span_eq=0, mismatches=[])
    e2e = [0, 0, 0]
    gated_grp = {0: [0] * 7, 1: [0] * 7, 2: [0] * 7}

    mms = {r: np.memmap(f"{evaldir}/rank{r}.f32", dtype="<f4", mode="r") for r in range(4)}
    fhs = [(r, open(f"{evaldir}/rank{r}.jsonl", encoding="utf-8")) for r in range(4)]
    ndocs = 0; nchars = 0
    chunk = []; chunk_chars = 0; chunk_max = 0
    stop = False

    def flush():
        nonlocal chunk, chunk_chars, chunk_max, ndocs, nchars
        if not chunk:
            return
        docs = []
        for did, rec, r in chunk:
            g = gold[did]; n = g["n"]; sl = rec["arrays"]; mm = mms[r]
            off = sl["alpha"]["offset"]
            al = np.asarray(mm[off:off + n], dtype="<f4").reshape(n).astype(np.float64)
            offa = sl["a"]["offset"]
            aa = np.asarray(mm[offa:offa + (n + 1) * 2], dtype="<f4").reshape(n + 1, 2).astype(np.float64)
            docs.append((did, rec, al, aa))
        dpfull = dp_batch([(d[2].shape[0], d[2], d[3]) for d in docs], const=None)
        partial_idx = [i for i, d in enumerate(docs) if gold[d[0]]["label"] == 1]
        cmaps = {}
        if partial_idx:
            cdocs = [(docs[i][2].shape[0], docs[i][2], None) for i in partial_idx]
            for cname in const_names:
                cv = float(cname.split("-")[-1])
                cm = dp_batch(cdocs, const=cv)
                cmaps[cname] = {i: m for i, m in zip(partial_idx, cm)}
        for j, (did, rec, al, aa) in enumerate(docs):
            g = gold[did]; n = g["n"]; pred = int(rec["prediction"])
            mask = dpfull[j]
            mine = spans_list(mask)
            stored = [tuple(x) for x in rec["decoded_spans"]]
            recon["docs"] += 1
            if set(mine) == set(stored):
                recon["span_eq"] += 1
            elif len(recon["mismatches"]) < 100:
                recon["mismatches"].append(dict(did=did, n=n, stored=len(stored), mine=len(mine)))
            gated = (np.zeros(n, dtype=bool) if pred == 0 else
                     np.ones(n, dtype=bool) if pred == 2 else mask)
            if g["label"] == 1:
                gm = mask_of(g["spans"], n)
                gsp = g["spans"]
                raw = al >= 0
                rspans = spans_list(raw)
                snap_m = mask_of(snap_spans(rspans, g["bounds"], n), n)
                vm = {"raw": (raw, rspans), "snap": (snap_m, spans_list(snap_m))}
                for k in SMOOTH_GRID:
                    sm = smooth_mask(raw, k)
                    vm[f"smooth-{k}"] = (sm, spans_list(sm))
                vm["dp-full"] = (mask, mine)
                vm["gated"] = (gated, spans_list(gated))
                for cname, cmap in cmaps.items():
                    cm2 = cmap[j]
                    vm[cname] = (cm2, spans_list(cm2))
                for v, (m2, sp2) in vm.items():
                    acc_variant(vars_[v], m2, sp2, gm, gsp, n)
                for v in ("raw", "dp-full", "gated"):
                    m2, sp2 = vm[v]
                    strat_add(strat["coverage"], bucket_of(g["cov"]), v, m2, sp2, gm, gsp)
                    strat_add(strat["generator"], g["gen"], v, m2, sp2, gm, gsp)
                    strat_add(strat["language"], g["lang"], v, m2, sp2, gm, gsp)
                row = gated_grp[pred]; row[0] += 1
                row[1] += int((mask & gm).sum()); row[2] += int((mask & ~gm).sum())
                row[3] += int((gm & ~mask).sum())
                row[4] += int((gated & gm).sum()); row[5] += int((gated & ~gm).sum())
                row[6] += int((gm & ~gated).sum())
                e2e[0] += int((gated & gm).sum())
                e2e[1] += int((gated & ~gm).sum())
                e2e[2] += int((gm & ~gated).sum())
            elif g["label"] == 2:
                vars_["raw"]["full"][0] += int((al >= 0).sum())
                vars_["dp-full"]["full"][0] += int(mask.sum())
                vars_["gated"]["full"][0] += int(gated.sum())
                for V in (vars_["raw"], vars_["dp-full"], vars_["gated"]):
                    V["full"][1] += n
                e2e[0] += int(gated.sum()); e2e[2] += n - int(gated.sum())
            else:
                vars_["raw"]["hum"][0] += int((al >= 0).sum())
                vars_["dp-full"]["hum"][0] += int(mask.sum())
                vars_["gated"]["hum"][0] += int(gated.sum())
                for V in (vars_["raw"], vars_["dp-full"], vars_["gated"]):
                    V["hum"][1] += n
                e2e[1] += int(gated.sum())
        ndocs += len(chunk); nchars += chunk_chars
        chunk = []; chunk_chars = 0; chunk_max = 0
        print(f"[{split}] {ndocs} docs {nchars/1e6:.1f}M chars {(time.time()-T0)/60:.1f}min "
              f"recon_eq={recon['span_eq']}/{recon['docs']}", flush=True)

    for r, fh in fhs:
        if stop:
            break
        for line in fh:
            rec = json.loads(line)
            did = rec["doc_id"]
            n = gold[did]["n"]
            if chunk and (chunk_chars + n > CH_CHARS or n > 1_000_000
                          or (len(chunk) + 1) * max(chunk_max, n) * 40 > MEM_BUDGET):
                flush()
            chunk.append((did, rec, r))
            chunk_chars += n; chunk_max = max(chunk_max, n)
            if SMOKE and ndocs + len(chunk) >= MAXDOC_SMOKE:
                stop = True
                break
    flush()
    for r, fh in fhs:
        fh.close()

    return dict(split=split,
                reconciliation=recon,
                variants={v: finalize_var(V) for v, V in vars_.items()},
                strat={k: finalize_strat(sk) for k, sk in strat.items()},
                e2e=dict(prf(*e2e), tp=e2e[0], fp=e2e[1], fn=e2e[2]),
                gated_by_predlabel={str(k): dict(docs=row[0],
                                                 dec_charF1=prf(row[1], row[2], row[3])["f1"],
                                                 gat_charF1=prf(row[4], row[5], row[6])["f1"],
                                                 dec_tp=row[1], dec_fp=row[2], dec_fn=row[3],
                                                 gat_tp=row[4], gat_fp=row[5], gat_fn=row[6])
                                    for k, row in gated_grp.items() if row[0]})


def engine_check(split, mine):
    try:
        eng = json.load(open(f"{EVAL[split]}/result.json"))["metrics"]
        got = mine["variants"]["dp-full"]
        return dict(engine_charF1=round(eng["character"]["decoded"]["f1"], 4),
                    mine_charF1=got["char"]["f1"],
                    engine_exact=round(eng["spans"]["decoded"]["exact"]["f1"], 4),
                    mine_exact=got["span"]["exact_f1"],
                    engine_iou50=round(eng["spans"]["decoded"]["iou50"]["f1"], 4),
                    mine_iou50=got["span"]["iou50_f1"],
                    engine_mae=round(eng["spans"]["decoded"]["boundary_mae"], 2),
                    mine_mae=got["span"]["mae"],
                    engine_matched=eng["spans"]["decoded"]["matched_spans"],
                    mine_matched=got["span"]["n_matched"],
                    engine_e2e_charF1=round(eng["character"]["e2e"]["f1"], 4),
                    mine_e2e_charF1=mine["e2e"]["f1"])
    except Exception as exc:
        return dict(error=f"{type(exc).__name__}: {exc}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if SMOKE:
        val = process_split("val", frozen_const=None)
        print(json.dumps(dict(recon=val["reconciliation"],
                              dp_full=val["variants"]["dp-full"],
                              engine=engine_check("val", val)), indent=1), flush=True)
        json.dump(dict(smoke_val=val), open(f"{OUTDIR}/smoke.json", "w"),
                  ensure_ascii=False, indent=1)
        print("SMOKE DONE")
        return
    val = process_split("val", frozen_const=None)
    grid = {f"dp-const-{c}": val["variants"][f"dp-const-{c}"] for c in CONST_GRID}
    best = max(grid, key=lambda k: (grid[k]["char"]["f1"], grid[k]["span"]["iou50_f1"]))
    frozen_c = float(best.split("-")[-1])
    print(f"[frozen] const c*={frozen_c} chosen on VAL by charF1 (tie->IoU50)", flush=True)
    test = process_split("test", frozen_const=frozen_c)
    results = dict(val=val, frozen_const=frozen_c, val_grid=grid, test=test,
                   engine_crosscheck=dict(val=engine_check("val", val),
                                          test=engine_check("test", test)))
    json.dump(results, open(f"{OUTDIR}/P1_Decoding_Canonical_Final.json", "w"),
              ensure_ascii=False, indent=1)
    json.dump(dict(val=dict(docs=val["reconciliation"]["docs"],
                            span_eq=val["reconciliation"]["span_eq"],
                            mismatches=val["reconciliation"]["mismatches"]),
                   test=dict(docs=test["reconciliation"]["docs"],
                             span_eq=test["reconciliation"]["span_eq"],
                             mismatches=test["reconciliation"]["mismatches"]),
                   engine_crosscheck=results["engine_crosscheck"]),
              open(f"{OUTDIR}/P1_A2_Full_Reconciliation.json", "w"),
              ensure_ascii=False, indent=1)
    print("FULL DONE")


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
            manifest = dict(start=T0, end=time.time(), runtime_min=round((time.time() - T0) / 60, 1),
                            exit_status=STATS["exit_status"], smoke=SMOKE,
                            host=socket.gethostname(), python=sys.version.split()[0],
                            numpy=np.__version__, outdir=OUTDIR, beta=BETA,
                            const_grid=CONST_GRID, smooth_grid=SMOOTH_GRID,
                            chunk_chars=CH_CHARS, mem_budget=MEM_BUDGET, run=RUN,
                            script_sha256=sha256_file(os.path.abspath(__file__)),
                            engine_decode_sha256=sha256_file(
                                "/OMFG/duomotai-backup-20260924/tev4_storage/code/"
                                "forensic_Q_qwen_char_v1_1_20261014/decode.py"),
                            checkpoint_sha256=sha256_file(f"{RUN}/checkpoint.pt"))
            json.dump(manifest, open(f"{OUTDIR}/run_manifest.json", "w"), indent=1)
            print("MANIFEST", json.dumps({k: v for k, v in manifest.items()
                                          if k != "checkpoint_sha256"}))
        except Exception as exc2:
            print("manifest failed:", exc2)
    sys.exit(STATS["exit_status"] or 0)
