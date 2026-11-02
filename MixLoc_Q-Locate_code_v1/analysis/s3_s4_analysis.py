# -*- coding: utf-8 -*-
"""S3: generated-vs-human structural diversity (paired, within controlled docs);
S4: language-mismatch distribution (heuristic re-detection) + doc-level sensitivity.
CPU-only, streaming. Output -> analysis_dataset_evidence_20261023/"""
import gzip, json, os, re, sys, time, unicodedata
import numpy as np

BASE = "/OMFG/duomotai-backup-20260924"
RUN = f"{BASE}/tev4_storage/runs_q/full_Q_locate_fast_seed0_20261014"
EVAL = f"{RUN}/eval_test"
OUT = f"{BASE}/tev4_storage/analysis_dataset_evidence_20261023"
SENT = re.compile(r"[^.!?。！？;；…]+[.!?。！？;；…]+|\n+")

STOP = {
 "es": {"el","la","que","de","en","un","una","por","con","para","los","las"},
 "pt": {"que","de","em","um","uma","para","com","não","os","as","por","mais"},
 "fr": {"le","la","les","de","des","et","que","dans","un","une","pour","avec"},
 "it": {"il","lo","la","di","che","un","una","per","con","non","gli","le"},
 "nl": {"de","het","een","en","van","dat","niet","voor","met","zijn","aan"},
 "en": {"the","of","and","to","in","is","that","for","it","was","are","on"},
 "de": {"der","die","das","und","ist","nicht","ein","eine","den","mit","für"},
 "tr": {"ve","bir","bu","için","ile","olarak","değil","daha","çok","olan"},
 "vi": {"của","và","các","được","một","cho","không","trong","với","những"},
}
LATIN = set(STOP)


def detect_language(text):
    t = unicodedata.normalize("NFC", text)[:3000]
    def ratio(rng):
        cnt = tot = 0
        for ch in t:
            o = ord(ch)
            if o > 32:
                tot += 1
                if rng[0] <= o <= rng[1]:
                    cnt += 1
        return cnt / max(1, tot)
    if ratio((0x3040, 0x30FF)) > 0.05:
        return "ja"
    if ratio((0xAC00, 0xD7AF)) > 0.2:
        return "ko"
    if ratio((0x0E00, 0x0E7F)) > 0.2:
        return "th"
    if ratio((0x0900, 0x097F)) > 0.2:
        return "hi"
    if ratio((0x0600, 0x06FF)) > 0.2:
        return "ar"
    if ratio((0x0400, 0x04FF)) > 0.3:
        return "ru"
    if ratio((0x4E00, 0x9FFF)) > 0.2:
        return "zh"
    toks = re.findall(r"[A-Za-zÀ-ɏĊ-₺]+", t.lower())[:80]
    if not toks:
        return "?"
    best, bl = "?", -1
    for lg, ws in STOP.items():
        hit = sum(1 for w in toks if w in ws)
        if hit > bl:
            best, bl = lg, hit
    return best if bl >= 2 else "?"


# ---------- S3: paired structural stats on controlled PARTIAL (test) ----------
op_stats = {}
def add_op(op, span_len, mach_sent_lens, hum_sent_lens, mach_ttr, hum_ttr):
    A = op_stats.setdefault(op, dict(n=0, sl=[], ms=[], hs=[], mt=[], ht=[]))
    A["n"] += 1
    A["sl"].append(span_len)
    A["ms"].extend(mach_sent_lens[:50]); A["hs"].extend(hum_sent_lens[:50])
    A["mt"].append(mach_ttr); A["ht"].append(hum_ttr)

n_docs = 0
with gzip.open(f"{BASE}/dataset_v2/dataset_v2_test.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["label"] != "PARTIAL_AI":
            continue
        text = unicodedata.normalize("NFC", r["text"])
        spans = r.get("machine_spans") or []
        op = r.get("operation") or "?"
        n = len(text)
        mach = "".join(text[s:e] for s, e in spans)
        hum_mask = np.ones(n, dtype=bool)
        for s, e in spans:
            hum_mask[s:e] = False
        hum = "".join(c for c, m in zip(text, hum_mask) if m)
        msl = [len(x) for x in SENT.findall(mach) if len(x.strip()) > 0]
        hsl = [len(x) for x in SENT.findall(hum) if len(x.strip()) > 0]
        mw = mach.split(); hw = hum.split()
        mttr = len(set(mw)) / max(1, len(mw)) if len(mw) >= 10 else np.nan
        httr = len(set(hw)) / max(1, len(hw)) if len(hw) >= 10 else np.nan
        add_op(op, len(mach), msl, hsl, mttr, httr)
        n_docs += 1
print(f"[S3] docs={n_docs} ops={list(op_stats)} {time.time()%1000:.0f}", flush=True)

S3 = {}
for op, A in op_stats.items():
    mt = [x for x in A["mt"] if x == x]
    ht = [x for x in A["ht"] if x == x]
    S3[op] = dict(
        n=A["n"],
        span_chars_mean=round(float(np.mean(A["sl"])), 1),
        machine_sent_len_mean=round(float(np.mean(A["ms"])), 1),
        human_sent_len_mean=round(float(np.mean(A["hs"])), 1),
        machine_sent_len_p90=round(float(np.percentile(A["ms"], 90)), 1),
        human_sent_len_p90=round(float(np.percentile(A["hs"], 90)), 1),
        machine_ttr_mean=round(float(np.mean(mt)), 3) if mt else None,
        human_ttr_mean=round(float(np.mean(ht)), 3) if ht else None)

# ---------- S4: mismatch distribution + sensitivity ----------
lang_total = {}
lang_mis = {}
mis_ids = set()
preds = {}
with gzip.open(f"{BASE}/dataset_v2/dataset_v2_test.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        did = r["row_id_v2"]
        declared = r.get("language") or "?"
        lab = r["label"]
        # re-detect only for AI-involved docs (mismatch record concerns generated content);
        # use full text for HUMAN (human pool language was declared by source)
        det = detect_language(r["text"])
        lang_total[declared] = lang_total.get(declared, 0) + 1
        if det != declared and det != "?":
            lang_mis[declared] = lang_mis.get(declared, 0) + 1
            if lab != "HUMAN":
                mis_ids.add(did)
print(f"[S4] langs={len(lang_total)} mismatched(docs all labels)={sum(lang_mis.values())}", flush=True)

LAB = {"HUMAN": 0, "PARTIAL_AI": 1, "FULL_AI": 2}
gold = {}
with gzip.open(f"{BASE}/dataset_v2/dataset_v2_test.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        gold[r["row_id_v2"]] = LAB[r["label"]]
for rr in range(4):
    with open(f"{EVAL}/rank{rr}.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            preds[rec["doc_id"]] = int(rec["prediction"])

def macro_f1(ids):
    cm = np.zeros((3, 3), dtype=np.int64)
    for d in ids:
        cm[gold[d], preds[d]] += 1
    f1s = []
    for c in range(3):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    acc = np.trace(cm) / max(1, cm.sum())
    return round(acc, 4), round(float(np.mean(f1s)), 4)

all_ids = list(preds)
keep = [d for d in all_ids if d not in mis_ids]
acc_all, mac_all = macro_f1(all_ids)
acc_keep, mac_keep = macro_f1(keep)
S4 = dict(
    declared_language_totals=lang_total,
    mismatch_by_declared_language=lang_mis,
    mismatch_rate_all=round(sum(lang_mis.values()) / max(1, sum(lang_total.values())), 4),
    mismatch_rate_ai_involved=round(len(mis_ids) / max(1, 110897), 4),
    sensitivity=dict(
        full_test=dict(acc=acc_all, macro_f1=mac_all, n=len(all_ids)),
        exclude_ai_mismatched=dict(acc=acc_keep, macro_f1=mac_keep, n=len(keep))),
    detector="NFC + script-ratio + Latin stopword fingerprint (heuristic re-detection)")

out = dict(s3_diversity=S3, s4_language_mismatch=S4)
json.dump(out, open(f"{OUT}/s3_s4_analysis.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(S3, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in S4.items() if k != "declared_language_totals"},
                 ensure_ascii=False, indent=1))
print("WROTE", f"{OUT}/s3_s4_analysis.json")
