# -*- coding: utf-8 -*-
"""P1 log-mining (CPU, no GPU): two evidence tasks from existing artifacts.

Task 1  Training trajectories: per-step loss components (doc/loc/dice/
        boundary_bce/boundary_rank) + gradient groups + clip rate, averaged
        across ranks, downsampled for curves.
Task 2  Boundary-field calibration at step500/1000/1433 (Val PARTIAL):
        does the learned `a` field discriminate gold span boundaries?
          - AUC vs all negatives (per channel start/end)
          - AUC vs near-boundary negatives (+-20 chars, the hard test)
          - precision@k (k = #gold boundaries, global top-k)
        Evolution across checkpoints pairs with exact-F1 trajectory.
Outputs -> /OMFG/.../analysis_logmine_20261018/*.json
"""
import json, os, sys, time, hashlib, socket
import numpy as np

RUN = "/OMFG/duomotai-backup-20260924/tev4_storage/runs_q/full_Q_locate_fast_seed0_20261014"
DATA_VAL = "/OMFG/duomotai-backup-20260924/dataset_v2/dataset_v2_val.jsonl.gz"
OUT = "/OMFG/duomotai-backup-20260924/tev4_storage/analysis_logmine_20261018"
CKPTS = ["eval_step500", "eval_step1000", "eval_step1433"]
NEAR = 20
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


# ---------------- Task 1: trajectories ----------------

def task1():
    steps = {}
    for r in range(4):
        p = f"{RUN}/train_rank{r}.jsonl"
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                s = d["step"]
                a = steps.setdefault(s, dict(n=0, loss={}, grad={}, clip=0, lr=0.0))
                for k, v in d["loss"].items():
                    a["loss"][k] = a["loss"].get(k, 0.0) + v
                for k, v in d["gradient_groups"].items():
                    a["grad"][k] = a["grad"].get(k, 0.0) + v
                a["clip"] += int(d["gradient_norm_before_clip"] > 1.0)
                a["lr"] = d["lr"]
                a["n"] += 1
    curve = {}
    for s in sorted(steps):
        a = steps[s]
        curve[s] = dict(loss={k: round(v / a["n"], 5) for k, v in a["loss"].items()},
                        grad={k: round(v / a["n"], 5) for k, v in a["grad"].items()},
                        clip_frac=round(a["clip"] / a["n"], 3), lr=a["lr"])
    # milestones: first step where boundary_bce < thresholds, and endpoints
    def first_below(key, th):
        for s in sorted(curve):
            if curve[s]["loss"].get(key, 1e9) < th:
                return s
        return None
    milestones = dict(
        boundary_bce_lt_0_1=first_below("boundary_bce", 0.1),
        boundary_bce_lt_0_05=first_below("boundary_bce", 0.05),
        boundary_rank_lt_0_05=first_below("boundary_rank", 0.05),
        loc_lt_0_15=first_below("loc", 0.15),
        doc_lt_0_10=first_below("doc", 0.10),
        start=curve[min(curve)], end=curve[max(curve)],
        n_steps=len(curve), clip_total=sum(curve[s]["clip_frac"] for s in curve))
    # downsample every 10 steps + always keep last
    ds = {s: curve[s] for s in sorted(curve)[::10]}
    ds[max(curve)] = curve[max(curve)]
    json.dump(dict(milestones=milestones, curve_downsampled=ds),
              open(f"{OUT}/training_trajectories.json", "w"), indent=1)
    return milestones


# ---------------- Task 2: boundary-field calibration ----------------

def auc_vs_neg(pos, neg):
    """AUC of pos vs neg via sorted-neg searchsorted (ties half-counted)."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    neg_s = np.sort(neg)
    idx = np.searchsorted(neg_s, pos)
    idx_lo = np.searchsorted(neg_s, pos, side="left")
    less = idx_lo.astype(np.float64)
    ties = (idx - idx_lo).astype(np.float64)
    return round(float(((less + 0.5 * ties) / len(neg_s)).mean()), 4)


def prec_at_k(pos, neg):
    if len(pos) == 0:
        return None
    k = len(pos)
    allv = np.concatenate([pos, neg]) if len(neg) else pos
    lab = np.concatenate([np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)])
    if k >= len(allv):
        return round(float(lab.mean()), 4)
    top = np.argpartition(-allv, k - 1)[:k]
    return round(float(lab[top].mean()), 4)


def task2():
    gold = {}
    with gzip_open(DATA_VAL) as f:
        for line in f:
            r = json.loads(line)
            if r["label"] != "PARTIAL_AI":
                continue
            gold[r["row_id_v2"]] = dict(
                n=len(r["text"]),
                spans=[(int(s), int(e)) for s, e in (r.get("machine_spans") or [])])
    out = {}
    for ck in CKPTS:
        d = f"{RUN}/{ck}"
        f32 = [f"{d}/rank{r}.f32" for r in range(4)]
        if not all(os.path.exists(p) for p in f32):
            out[ck] = dict(error="missing array exports")
            continue
        acc = {f"{ch}_{pop}": [] for ch in ("start", "end")
               for pop in ("pos", "neg", "nearneg")}
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
                    sl = rec["arrays"]["a"]
                    off = sl["offset"]
                    aa = np.asarray(mms[r][off:off + (n + 1) * 2],
                                    dtype="<f4").reshape(n + 1, 2).astype(np.float64)
                    is_s = np.zeros(n, bool); is_e = np.zeros(n + 1, bool)
                    for s, e in g["spans"]:
                        if 0 <= s < n:
                            is_s[s] = True
                        if 0 <= e <= n:
                            is_e[e] = True
                    near_s = np.zeros(n, bool); near_e = np.zeros(n + 1, bool)
                    for s, e in g["spans"]:
                        near_s[max(0, s - NEAR):min(n, s + NEAR + 1)] = True
                        near_e[max(0, e - NEAR):min(n + 1, e + NEAR + 1)] = True
                    astart = aa[:n, 0]; aend = aa[:, 1]
                    acc["start_pos"].append(astart[is_s])
                    acc["start_neg"].append(astart[~is_s & ~near_s])
                    acc["start_nearneg"].append(astart[~is_s & near_s])
                    acc["end_pos"].append(aend[is_e])
                    acc["end_neg"].append(aend[~is_e & ~near_e])
                    acc["end_nearneg"].append(aend[~is_e & near_e])
                    ndocs += 1
        res = dict(docs=ndocs)
        for ch in ("start", "end"):
            pos = np.concatenate(acc[f"{ch}_pos"]) if acc[f"{ch}_pos"] else np.array([])
            neg = np.concatenate(acc[f"{ch}_neg"]) if acc[f"{ch}_neg"] else np.array([])
            nneg = np.concatenate(acc[f"{ch}_nearneg"]) if acc[f"{ch}_nearneg"] else np.array([])
            res[ch] = dict(
                n_pos=int(len(pos)), n_neg=int(len(neg)), n_nearneg=int(len(nneg)),
                auc_all=auc_vs_neg(pos, neg),
                auc_near20=auc_vs_neg(pos, nneg),
                precision_at_k=prec_at_k(pos, neg),
                pos_mean=round(float(pos.mean()), 3) if len(pos) else None,
                neg_mean=round(float(neg.mean()), 3) if len(neg) else None,
                nearneg_mean=round(float(nneg.mean()), 3) if len(nneg) else None)
        try:
            m = json.load(open(f"{d}/result.json"))["metrics"]
            res["exact_f1_decoded"] = round(m["spans"]["decoded"]["exact"]["f1"], 4)
            res["char_f1_raw"] = round(m["character"]["raw"]["f1"], 4)
            res["doc_macro_f1"] = round(m.get("document", m).get("macro_f1",
                                   m.get("doc", {}).get("macro_f1", -1)), 4)
        except Exception as exc:
            res["result_json"] = f"{type(exc).__name__}: {exc}"
        out[ck] = res
        print(f"[calib] {ck}: {json.dumps({k: v for k, v in res.items() if k != 'result_json'})}",
              flush=True)
    json.dump(out, open(f"{OUT}/boundary_field_calibration.json", "w"), indent=1)
    return out


def gzip_open(path):
    import gzip
    return gzip.open(path, "rt", encoding="utf-8")


def main():
    os.makedirs(OUT, exist_ok=True)
    mil = task1()
    print("[traj]", json.dumps(mil), flush=True)
    cal = task2()
    json.dump(dict(milestones=mil), open(f"{OUT}/_summary.json", "w"), indent=1)


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
            json.dump(dict(exit_status=STATS["exit_status"], runtime_min=round((time.time() - T0) / 60, 1),
                           host=socket.gethostname(), outdir=OUT,
                           script_sha256=sha256_file(os.path.abspath(__file__)),
                           checkpoint_sha256=sha256_file(f"{RUN}/checkpoint.pt")),
                      open(f"{OUT}/run_manifest.json", "w"), indent=1)
        except Exception as e2:
            print("manifest failed:", e2)
        print("MANIFEST WRITTEN")
    sys.exit(STATS["exit_status"] or 0)
