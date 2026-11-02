"""Stratified offline analysis v2: gold straight from test JSONL (no sqlite)."""
import collections, gzip, json, time

RUN = "/OMFG/duomotai-backup-20260924/tev4_storage/runs_q/full_Q_locate_fast_seed0_20261014"
TEST_JSONL = "/OMFG/duomotai-backup-20260924/dataset_v2/dataset_v2_test.jsonl.gz"
OUT = RUN + "/eval_test/stratified_analysis.json"
WORLD = 4

def prf(tp, fp, fn):
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * tp / max(1, 2 * tp + fp + fn), 4)}

def char_counts(pred_ivs, gold_spans, n):
    pm = bytearray(n)
    for s, e in pred_ivs:
        s = max(0, int(s)); e = min(n, int(e))
        for c in range(s, e): pm[c] = 1
    gm = bytearray(n)
    for s, e in gold_spans:
        s = max(0, int(s)); e = min(n, int(e))
        for c in range(s, e): gm[c] = 1
    tp = fp = fn = 0
    for c in range(n):
        if pm[c] and gm[c]: tp += 1
        elif pm[c]: fp += 1
        elif gm[c]: fn += 1
    return tp, fp, fn

def macro3(cm):
    f1s = []
    for c in range(3):
        tp = cm[c][c]; fp = sum(cm[r][c] for r in range(3)) - tp; fn = sum(cm[c]) - tp
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        f1s.append(2 * p * r / max(1e-12, p + r))
    return sum(f1s) / 3

def bucket_of(cov):
    if cov < 0.05: return "0-5%"
    if cov < 0.20: return "5-20%"
    if cov < 0.50: return "20-50%"
    if cov < 0.80: return "50-80%"
    return "80-100%"

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

log("loading gold+meta from test jsonl...")
gold = {}; gen_of = {}; lang_of = {}
LAB = {"HUMAN": 0, "PARTIAL_AI": 1, "FULL_AI": 2}
with gzip.open(TEST_JSONL, "rt", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        did = row["row_id_v2"]
        text = row["text"]; n = len(text)
        spans = [(int(s), int(e)) for s, e in (row.get("gold_machine_spans") or [])]
        cov = (sum(e - s for s, e in spans) / n) if n else 0.0
        gold[did] = {"label": LAB[row["label"]], "n": n, "spans": spans, "coverage": cov}
        gen_of[did] = row.get("generator") or "UNKNOWN"
        lang_of[did] = row.get("language") or "UNKNOWN"
log(f"gold docs: {len(gold)}")

log("loading predictions...")
preds = {}
for r in range(WORLD):
    with open(f"{RUN}/eval_test/rank{r}.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            preds[rec["doc_id"]] = {"pred": int(rec["prediction"]),
                                    "dec": rec.get("decoded_spans") or []}
log(f"predictions: {len(preds)}")
assert set(preds) == set(gold), f"mismatch: pred={len(preds)} gold={len(gold)}"

BUCKETS = ["HUMAN(all)", "0-5%", "5-20%", "20-50%", "50-80%", "80-100%"]
log("coverage stratification (char scan)...")
cov_stats = {b: {"cm": [[0]*3 for _ in range(3)], "dec": [0, 0, 0], "n": 0} for b in BUCKETS}
for did, g in gold.items():
    p = preds[did]
    b = "HUMAN(all)" if g["label"] == 0 else bucket_of(g["coverage"])
    s = cov_stats[b]
    s["cm"][g["label"]][p["pred"]] += 1; s["n"] += 1
    if g["label"] == 1:
        tp, fp, fn = char_counts(p["dec"], g["spans"], g["n"])
        s["dec"][0] += tp; s["dec"][1] += fp; s["dec"][2] += fn
cov_result = {}
for b in BUCKETS:
    s = cov_stats[b]
    e = {"n": s["n"], "macro_f1": round(macro3(s["cm"]), 4), "confusion": s["cm"]}
    if any(s["dec"]): e["char_decoded_F1"] = prf(*s["dec"])
    cov_result[b] = e
log("coverage done")

log("per-generator (char scan)...")
gen_stats = collections.defaultdict(lambda: {"cm": [[0]*3 for _ in range(3)], "dec": [0, 0, 0], "n": 0})
for did, g in gold.items():
    p = preds[did]
    s = gen_stats[gen_of.get(did, "UNKNOWN")]
    s["cm"][g["label"]][p["pred"]] += 1; s["n"] += 1
    if g["label"] == 1:
        tp, fp, fn = char_counts(p["dec"], g["spans"], g["n"])
        s["dec"][0] += tp; s["dec"][1] += fp; s["dec"][2] += fn
gen_result = {}
for gen, s in sorted(gen_stats.items(), key=lambda kv: -kv[1]["n"]):
    e = {"n": s["n"], "macro_f1": round(macro3(s["cm"]), 4), "confusion": s["cm"]}
    if any(s["dec"]): e["char_decoded_F1"] = prf(*s["dec"])
    gen_result[gen] = e
log("generator done")

log("per-language...")
lang_stats = collections.defaultdict(lambda: {"cm": [[0]*3 for _ in range(3)], "n": 0})
for did, g in gold.items():
    s = lang_stats[lang_of.get(did, "UNKNOWN")]
    s["cm"][g["label"]][preds[did]["pred"]] += 1; s["n"] += 1
lang_result = {lang: {"n": s["n"], "macro_f1": round(macro3(s["cm"]), 4), "confusion": s["cm"]}
               for lang, s in sorted(lang_stats.items(), key=lambda kv: -kv[1]["n"])}
log("language done")

result = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
          "scope": "test split (202,762 docs); decoded prediction path",
          "coverage_stratified": cov_result, "per_generator": gen_result,
          "per_language": lang_result}
open(OUT, "w").write(json.dumps(result, indent=2))
log(f"written {OUT}")

print("=== COVERAGE ===")
for b in BUCKETS:
    e = cov_result[b]
    line = f"{b:10s} n={e['n']:7d} macroF1={e['macro_f1']:.4f}"
    if "char_decoded_F1" in e: line += f"  charF1={e['char_decoded_F1']['f1']:.4f}"
    print(line)
print("=== GENERATOR ===")
for gen, e in gen_result.items():
    line = f"{gen:22s} n={e['n']:7d} macroF1={e['macro_f1']:.4f}"
    if "char_decoded_F1" in e: line += f"  charF1={e['char_decoded_F1']['f1']:.4f}"
    print(line)
print("=== LANGUAGE (top 10) ===")
for lang, e in list(lang_result.items())[:10]:
    print(f"{lang:6s} n={e['n']:7d} macroF1={e['macro_f1']:.4f}")
