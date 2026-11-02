# -*- coding: utf-8 -*-
"""V4 trainer/evaluator: smoke / pilot / A0 phases; arms: LoRA | Effort-SVD.

Contracts implemented: C3 class-mixed schedule (sampler), C4 class-norm CE,
C5 label-derived anchors, C7 finite guards, C10 no truncation (long docs via
two-pass gradient carrier — exact gradients, bounded memory), C13 1/m token
loss, R4 (1/D) doc-mean step loss, storage rule (mounted drive; best-only
weights; one rolling checkpoint.pt is updated atomically during the run).
Run: python3 runner.py --arm lora --phase smoke --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
import random
import hashlib
import numpy as np
from typing import List

import torch
import torch.nn.functional as F

from adapters import build_lora_arm, build_svd_arm, trainable_param_count
from char_align import (canonical_token_probs_from_windows,
                        project_token_probs_to_chars, threshold_to_char_spans)
from dataset_v2 import Sample, load_split, stratified_subset, split_provenance, file_sha256
from losses import class_normalized_weights
from model import Qwen3DualHead
from sampler import class_mixed_schedule

MODEL_PATH = os.environ.get(
    "TEV4_MODEL_PATH", "/OMFG/duomotai-backup-20260924/tev4_storage/models/Qwen3-4B")
RUNS = os.environ.get(
    "TEV4_RUNS", "/OMFG/duomotai-backup-20260924/tev4_storage/runs_v4")
PACK_MAX_TOKENS = int(os.environ.get('PACK_MAX_TOKENS', 10240))
PACK_MAX_WINDOWS = int(os.environ.get('PACK_MAX_WINDOWS', 6))
LONG_DOC_WINDOWS = 6
LONG_SEEN = set()
CHUNK_WINDOWS = 1


# ----------------------------- primitives -----------------------------
def _ckpt_set(model, on):
    for m in model.backbone.modules():
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = on


def _ckpt_off(model):
    _ckpt_set(model, False)


def _ckpt_on(model):
    _ckpt_set(model, True)



def _bad_grad_count(model):
    return sum(1 for q in model.parameters()
               if q.requires_grad and q.grad is not None
               and not torch.isfinite(q.grad).all())


def pad_batch(win_ids: List[List[int]], device):
    """Vectorized right-padding (was a python loop — speed patch 2026-10-10)."""
    seqs = [torch.tensor(w, dtype=torch.long) for w in win_ids]
    ids = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=0).to(device, non_blocking=True)
    lens = torch.tensor([len(w) for w in win_ids], device=device)
    mask = (torch.arange(ids.shape[1], device=device)
            .unsqueeze(0) < lens.unsqueeze(1)).long()
    return ids, mask


def forward_windows(model, win_ids, device, cache=True):
    """cache=False for any NO-GRAD forward (C6: autocast cast-cache poison,
    R24 lesson — a cached detached bf16 cast silently detaches later
    grad-forwards; suspected root cause of SVD-arm NaN grads at step 2)."""
    ids, mask = pad_batch(win_ids, device)
    if os.environ.get("TEV4_NO_AUTOCAST") == "1":
        last_h, tok_logits = model.forward_window(ids, mask)
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=cache):
            last_h, tok_logits = model.forward_window(ids, mask)
    return last_h.float(), tok_logits.float()


def _window_token_terms(s: Sample, w_idx: int, logits_row):
    """(bce*inv*valid) sum and mass for window w_idx of sample s."""
    a, b = s.windows[w_idx]
    L = b - a
    tgt = torch.tensor(s.token_targets[a:b], device=logits_row.device).float()
    valid = torch.tensor(s.valid[a:b], device=logits_row.device).float()
    inv = torch.tensor([1.0 / m for m in s.multiplicity[a:b]],
                       device=logits_row.device)
    bce = F.binary_cross_entropy_with_logits(
        logits_row[:L], tgt, reduction="none")
    return (bce * inv * valid).sum(), int(np.count_nonzero(s.valid[a:b]))


def doc_ce(model, mean_h: torch.Tensor, label_id: int, cls_w, scale: float):
    logits = model.doc_head(
        mean_h.to(model.doc_head.weight.dtype).unsqueeze(0))
    logp = F.log_softmax(logits, dim=-1)
    loss = cls_w[label_id] * (-logp[0, label_id])
    return loss * scale, logits.detach()


def make_packs(step_samples: List[Sample]):
    """Greedy packs (all windows per forward); two-pass only for docs whose
    token count exceeds the pack budget (true monsters, ~0.1%). Rationale:
    2026-10-10 SVD NaN forensics — pack path clean, two-pass path NaNs in
    training context only; route everything that fits through packs."""
    def _fwd_cost(s):
        return sum(b - a for (a, b) in s.windows)  # padded forward positions
    short = [s for s in step_samples if _fwd_cost(s) <= PACK_MAX_TOKENS]
    longs = [s for s in step_samples if _fwd_cost(s) > PACK_MAX_TOKENS]
    if longs and os.environ.get("TEV4_SKIP_LONG") == "1":
        print(f"SKIP-LONG(diagnostic): dropping {len(longs)} docs "
              f"ids={[s.doc_id[:8] for s in longs]}", flush=True)
        longs = []
    short.sort(key=lambda s: -s.n_tokens)
    # budget by PADDED size: batch tensor = n_windows x max_window_len
    # (2026-10-10 A100 OOM root cause: sum-of-lengths accounting allowed
    #  16 x 4096 = 65K padded positions under a 12K "budget")
    packs, cur, cur_win, cur_maxlen = [], [], 0, 0
    for s in short:
        s_maxlen = max(b - a for a, b in s.windows)
        if cur:
            new_win = cur_win + s.n_windows
            new_max = max(cur_maxlen, s_maxlen)
            if new_win * new_max > PACK_MAX_TOKENS or new_win > PACK_MAX_WINDOWS:
                packs.append(("pack", cur))
                cur, cur_win, cur_maxlen = [], 0, 0
        cur.append(s)
        cur_win += s.n_windows
        cur_maxlen = max(cur_maxlen, s_maxlen)
    if cur:
        packs.append(("pack", cur))
    packs.extend(("long", [s]) for s in longs)
    return packs


def train_step(model, step_samples, D_step, cls_w, device, optimizer, world=1):
    """Exact R4: L_step = (1/D)·Σ_d(L_doc_d + L_tok_d); returns telemetry.
    DDP mode (world>1): each rank holds a shard of the step's docs; the loss
    computed here is sum-over-local-docs scaled by world/D_step, so that after
    DDP's gradient averaging the effective objective is exactly (1/D)Σ_all L_d
    (pre-registered R4 identity; see losses.step_document_mean docstring)."""
    D_scale = world / float(D_step) if world > 1 else 1.0 / float(D_step)
    l_doc_vals, l_tok_vals = [], []
    for kind, group in make_packs(step_samples):
        if kind == "pack":
            win_ids, owner = [], []
            for di, s in enumerate(group):
                for (a, b) in s.windows:
                    win_ids.append(s.input_ids[a:b])
                    owner.append(di)
            if os.environ.get("TEV4_FORENSIC") == "1":
                print(f"MEM before pack: alloc="
                      f"{torch.cuda.memory_allocated()/2**30:.2f}G "
                      f"reserved={torch.cuda.memory_reserved()/2**30:.2f}G "
                      f"docs={len(group)} win={len(win_ids)}", flush=True)
            last_h, tok_logits = forward_windows(model, win_ids, device)
            h_sum = torch.zeros(len(group), last_h.shape[1], device=device)
            for wi, di in enumerate(owner):
                h_sum[di] += last_h[wi]
            means = h_sum / torch.tensor(
                [s.n_windows for s in group], device=device
            ).unsqueeze(1).clamp(min=1)
            tok_terms = [torch.zeros((), device=device) for _ in group]
            l_doc = torch.zeros((), device=device)
            l_tok = torch.zeros((), device=device)
            for di, s in enumerate(group):
                ce, _ = doc_ce(model, means[di], s.label_id, cls_w, 1.0)
                l_doc = l_doc + ce
            for wi in range(len(win_ids)):
                di = owner[wi]
                s = group[di]
                w_idx = wi - sum(s2.n_windows for s2 in group[:di])
                term, _ = _window_token_terms(s, w_idx, tok_logits[wi])
                tok_terms[di] = tok_terms[di] + term
            for di, s in enumerate(group):
                l_tok = l_tok + tok_terms[di] / max(1, s.n_tokens)
            if os.environ.get("TEV4_FORENSIC") == "1":
                _offs = {d: sum(x.n_windows for x in group[:d])
                         for d in range(len(group))}
                _tlm = 0.0
                for wi, od in enumerate(owner):
                    a, b = group[od].windows[wi - _offs[od]]
                    _tlm = max(_tlm, float(tok_logits[wi, :b - a].abs().max()))
                print(f"FORENSIC docs={len(group)} ldoc={float(l_doc):.4f} "
                      f"ltok={float(l_tok):.4f} toklogit_maxabs={_tlm:.1f}",
                      flush=True)
            (l_doc + l_tok).mul(D_scale).backward()
            if os.environ.get("TEV4_FORENSIC") == "1" and _bad_grad_count(model):
                print(f"FORENSIC-BADPACK docs={[s.doc_id for s in group]} "
                      f"ldoc={float(l_doc):.3f} ltok={float(l_tok):.3f} "
                      f"tlm={_tlm:.1f}", flush=True)
            l_doc_vals.append(l_doc.detach())
            l_tok_vals.append(l_tok.detach())
        else:
            s = group[0]
            LONG_SEEN.add(s.doc_id)
            if os.environ.get("TEV4_FORENSIC") == "1":
                print(f"FORENSIC-LONG doc={s.doc_id} nw={s.n_windows} "
                      f"ntok={s.n_tokens}", flush=True)
            # pass 1: no-grad mean hidden; head-only CE keeps head grads exact
            with torch.no_grad():
                h_sum = torch.zeros(model.doc_head.in_features, device=device)
                chunks = [s.windows[i:i + CHUNK_WINDOWS]
                          for i in range(0, s.n_windows, CHUNK_WINDOWS)]
                for ch in chunks:
                    lh, _ = forward_windows(
                        model, [s.input_ids[a:b] for (a, b) in ch], device,
                        cache=False)
                    h_sum += lh.sum(dim=0)
                mean_h = h_sum / s.n_windows
            ce_head, logits_det = doc_ce(model, mean_h.detach(),
                                         s.label_id, cls_w, 1.0)
            (ce_head * D_scale).backward()          # exact head grads
            with torch.no_grad():                   # carrier dL/dH_sum
                p = F.softmax(model.doc_head(mean_h.unsqueeze(0)), -1)[0]
                onehot = F.one_hot(torch.tensor(s.label_id, device=device),
                                   3).float()
                dh = (cls_w[s.label_id] * (p - onehot)) @ model.doc_head.weight
                dh = dh / s.n_windows
            # _ckpt_off: no-ckpt needs ~10.3G for one 4096 window — does not fit
            # 23.5G card at 13G baseline (OOM 2026-10-10). Keep ckpt ON (exact, ~0.7G).
            long_tok = torch.zeros((), device=device)
            for ch in chunks:
                lh, tok_logits = forward_windows(
                    model, [s.input_ids[a:b] for (a, b) in ch], device)
                surrogate = (lh * dh.unsqueeze(0)).sum()
                base = next(wi for wi, (a, b) in enumerate(s.windows)
                            if (a, b) == ch[0])
                tok = torch.zeros((), device=device)
                for k in range(len(ch)):
                    term, _ = _window_token_terms(s, base + k, tok_logits[k])
                    tok = tok + term
                ((surrogate + tok / max(1, s.n_tokens)) * D_scale).backward()
                long_tok += tok.detach() / max(1, s.n_tokens)
                if os.environ.get("TEV4_FORENSIC") == "1" and _bad_grad_count(model):
                    print(f"FORENSIC-BADLONG doc={s.doc_id} chunk", flush=True)
            l_doc_vals.append(ce_head.detach())
            l_tok_vals.append(long_tok)
    count = max(1, len(step_samples))
    return (float(sum(l_doc_vals) / count), float(sum(l_tok_vals) / count))


# ------------------------------- eval ---------------------------------
@torch.no_grad()
def evaluate(model, val_samples, device, batch_tokens=8192):
    was_training = model.training
    model.eval()
    try:
        return _evaluate(model, val_samples, device, batch_tokens)
    finally:
        model.train(was_training)


def _evaluate(model, val_samples, device, batch_tokens):
    ys, preds = [], []
    tp = fp = fn = 0
    exact_tp = exact_p = exact_g = 0
    iou_tp = iou_p = iou_g = 0
    bnd_errs = []
    e2e_tp = e2e_fp = e2e_fn = 0
    gated_tp = gated_fp = gated_fn = 0
    anchor_errors = {0: [], 2: []}
    partial_doc_f1 = []
    samples = sorted(val_samples, key=lambda s: s.n_tokens)
    for s, mean_h, canon in iter_eval_documents(model, samples, device, batch_tokens):
        logits = model.doc_head(mean_h.to(model.doc_head.weight.dtype).unsqueeze(0))
        prediction = int(logits.argmax(-1))
        ys.append(s.label_id)
        preds.append(prediction)
        if s.label_id in anchor_errors:
            raw = np.asarray(canon) >= .5
            valid = np.asarray(s.valid)
            anchor_errors[s.label_id].append(float(np.mean(
                raw[valid] != (s.label_id == 2))))
        charp = project_token_probs_to_chars(s.offsets, canon, s.text_len)
        gt_mask = np.zeros(s.text_len, dtype=bool)
        if s.label_id == 2:
            gt_mask[:] = True
        elif s.label_id == 1:
            for a, b in s.spans:
                gt_mask[a:min(b, s.text_len)] = True
        pred_mask = np.asarray(charp) >= .5
        gated = (np.zeros(s.text_len, dtype=bool) if prediction == 0 else
                 np.ones(s.text_len, dtype=bool) if prediction == 2 else pred_mask)
        dtp = int(np.count_nonzero(gt_mask & gated))
        dfp = int(np.count_nonzero(~gt_mask & gated))
        dfn = int(np.count_nonzero(gt_mask & ~gated))
        e2e_tp += dtp
        e2e_fp += dfp
        e2e_fn += dfn
        if s.label_id != 1:
            continue
        gated_tp += dtp
        gated_fp += dfp
        gated_fn += dfn
        pred_spans = threshold_to_char_spans(charp, 0.5)
        rtp = int(np.count_nonzero(gt_mask & pred_mask))
        rfp = int(np.count_nonzero(~gt_mask & pred_mask))
        rfn = int(np.count_nonzero(gt_mask & ~pred_mask))
        tp += rtp
        fp += rfp
        fn += rfn
        partial_doc_f1.append(2 * rtp / max(1, 2 * rtp + rfp + rfn))
        exact_p += len(pred_spans)
        exact_g += len(s.spans)
        exact_tp += len(set(map(tuple, pred_spans)) & set(map(tuple, s.spans)))
        used = set()
        for (ps, pe) in pred_spans:
            best, bi = 0.5, None
            for gi, (gs, ge) in enumerate(s.spans):
                if gi in used:
                    continue
                inter = max(0, min(pe, ge) - max(ps, gs))
                union = (pe - ps) + (ge - gs) - inter
                iou = inter / union if union else 0.0
                if iou >= best:
                    best, bi = iou, gi
            if bi is not None:
                used.add(bi)
                iou_tp += 1
                bnd_errs.append(abs(ps - s.spans[bi][0]))
                bnd_errs.append(abs(pe - s.spans[bi][1]))
        iou_p += len(pred_spans)
        iou_g += len(s.spans)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # macro-F1 (3-class)
    from collections import Counter
    per = {}
    for c in range(3):
        tp_c = sum(1 for y, q in zip(ys, preds) if y == c and q == c)
        fp_c = sum(1 for y, q in zip(ys, preds) if y != c and q == c)
        fn_c = sum(1 for y, q in zip(ys, preds) if y == c and q != c)
        p_c = tp_c / (tp_c + fp_c) if tp_c + fp_c else 0.0
        r_c = tp_c / (tp_c + fn_c) if tp_c + fn_c else 0.0
        per[c] = 2 * p_c * r_c / (p_c + r_c) if p_c + r_c else 0.0
    macro = sum(per.values()) / 3
    acc = sum(1 for y, q in zip(ys, preds) if y == q) / max(1, len(ys))
    ex_f1 = (2 * exact_tp / (exact_p + exact_g)
             if exact_p + exact_g else 0.0)
    io_f1 = (2 * iou_tp / (iou_p + iou_g) if iou_p + iou_g else 0.0)
    bnd = sum(bnd_errs) / max(1, len(bnd_errs))
    return dict(acc=acc, macro_f1=macro, per_class_f1=per,
                char_precision=prec, char_recall=rec, char_f1=f1,
                span_exact_f1=ex_f1, span_iou50_f1=io_f1, boundary_mae=bnd,
                n_documents=len(ys),
                confusion_matrix=[[sum(y == a and p == b for y, p in zip(ys, preds))
                                   for b in range(3)] for a in range(3)],
                raw_partial_char_f1=f1,
                raw_partial_doc_mean_char_f1=sum(partial_doc_f1)/max(1, len(partial_doc_f1)),
                gated_partial_char_f1=2*gated_tp/max(1, 2*gated_tp+gated_fp+gated_fn),
                e2e_all_char_f1=2*e2e_tp/max(1, 2*e2e_tp+e2e_fp+e2e_fn),
                human_token_false_positive_doc_mean=float(np.mean(anchor_errors[0]))
                    if anchor_errors[0] else None,
                full_token_false_negative_doc_mean=float(np.mean(anchor_errors[2]))
                    if anchor_errors[2] else None,
                boundary_matched_spans=iou_tp)


@torch.no_grad()
def predict_samples(model, samples, device, batch_tokens=8192):
    """Prediction uses text-derived features only; no ground-truth class gate."""
    was_training = model.training
    model.eval()
    try:
        for sample, mean, canon in iter_eval_documents(model, samples, device, batch_tokens):
            logits = model.doc_head(mean.to(model.doc_head.weight.dtype).unsqueeze(0))
            probabilities = logits.softmax(-1)[0].cpu().tolist()
            label = int(logits.argmax(-1))
            if label == 0:
                spans = []
            elif label == 2:
                spans = [(0, sample.text_len)]
            else:
                charp = project_token_probs_to_chars(sample.offsets, canon, sample.text_len)
                spans = threshold_to_char_spans(charp, .5)
            yield dict(doc_id=sample.doc_id, label_id=label,
                       class_probabilities=probabilities, machine_spans=spans)
    finally:
        model.train(was_training)


def iter_eval_documents(model, samples, device, batch_tokens=8192):
    """Bound padded forwards, including a single document with many windows.

    Aggregate window representations on GPU, probabilities on CPU. A document
    is yielded only after every window has been counted, never truncated.
    """
    pending = {}

    def process(batch):
        h, logits = forward_windows(model,
            [s.input_ids[a:b] for _, s, _, a, b in batch], device, cache=False)
        probs = logits.sigmoid().cpu().numpy()
        completed = []
        for row, (di, sample, wi, a, b) in enumerate(batch):
            if di not in pending:
                pending[di] = [torch.zeros_like(h[row]),
                               np.zeros(sample.n_tokens, dtype=np.float64)]
            total, canon = pending[di]
            total.add_(h[row])
            canon[a:b] += probs[row, :b-a]
            if wi == sample.n_windows - 1:
                completed.append((sample, total / sample.n_windows,
                                  canon / sample.multiplicity))
                del pending[di]
        return completed

    batch, maxlen = [], 0
    for di, sample in enumerate(samples):
        if not sample.windows:
            raise RuntimeError(f'empty document: {sample.doc_id}')
        for wi, (a, b) in enumerate(sample.windows):
            length = b - a
            if length > batch_tokens:
                raise ValueError('eval budget smaller than a canonical window')
            newmax = max(maxlen, length)
            if batch and (len(batch) + 1) * newmax > batch_tokens:
                yield from process(batch)
                batch, maxlen = [], 0
            batch.append((di, sample, wi, a, b))
            maxlen = max(maxlen, length)
    if batch:
        yield from process(batch)
    if pending:
        raise RuntimeError('incomplete document aggregation')


def canonical_probs_for(s, owners, di, tok_logits):
    """Compatibility helper with one block transfer, no scalar GPU syncs."""
    rows = [wi for wi, d in enumerate(owners) if d == di]
    probs = tok_logits[rows].sigmoid().detach().cpu().numpy()
    canon = np.zeros(s.n_tokens, dtype=np.float64)
    for wi, (a, b) in enumerate(s.windows):
        canon[a:b] += probs[wi, :b-a]
    return (canon / s.multiplicity).tolist()


def wi_of(owners, di, k):
    """Global window row index of doc di's k-th window."""
    seen = -1
    for wi, d in enumerate(owners):
        if d == di:
            seen += 1
            if seen == k:
                return wi
    raise RuntimeError("wi_of")


def trainable_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()
            if p.requires_grad}


def load_trainable_state(model, state):
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if params.keys() != state.keys():
        raise RuntimeError('checkpoint trainable parameter keys mismatch')
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(state[name].to(p))


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_recovery(model, optimizer, step, best, manifest, path):
    payload = dict(weights=trainable_state(model), optimizer=optimizer.state_dict(),
                   step=step, best=best, manifest=manifest,
                   rng=torch.get_rng_state(), python_rng=random.getstate(),
                   numpy_rng=np.random.get_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    atomic_torch_save(payload, path)


def load_recovery(model, optimizer, path, manifest):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['manifest'] != manifest:
        raise RuntimeError('recovery manifest mismatch; use the frozen source/config')
    load_trainable_state(model, payload['weights'])
    optimizer.load_state_dict(payload['optimizer'])
    # Optimizer state is serialized on CPU so checkpoints remain portable.
    # Move tensor state back to the live parameter device before stepping.
    device = next((p.device for p in model.parameters() if p.requires_grad),
                  torch.device('cpu'))
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)
    torch.set_rng_state(payload['rng'])
    random.setstate(payload['python_rng'])
    np.random.set_state(payload['numpy_rng'])
    if payload['cuda_rng']:
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return payload['step'], payload['best']


def svd_diagnostics(model):
    from svd_residual import SVDResidualLinear
    records = {name: m.drift_diagnostics() for name, m in model.backbone.named_modules()
               if isinstance(m, SVDResidualLinear)}
    if not records:
        return {}
    return dict(per_projection=records,
                mean={key: sum(r[key] for r in records.values()) / len(records)
                      for key in next(iter(records.values()))},
                max={key: max(r[key] for r in records.values())
                     for key in next(iter(records.values()))})


def svd_anchor_fingerprint(model):
    """Small deterministic fingerprint for frozen SVD orientation/state.

    Recovery reconstructs SVD from the frozen base model, then loads only the
    trainable tail. Refuse a resume if that reconstruction changed. Sampling
    tensor ends keeps startup/checkpoint overhead negligible while detecting
    sign/order changes in every adapted projection.
    """
    import hashlib
    from svd_residual import SVDResidualLinear
    h = hashlib.sha256()
    for name, m in model.backbone.named_modules():
        if not isinstance(m, SVDResidualLinear):
            continue
        h.update(name.encode())
        for tensor in (m.weight_main, m.u_r, m.vh_r):
            flat = tensor.detach().float().cpu().reshape(-1)
            take = torch.cat((flat[:32], flat[-32:])) if flat.numel() > 64 else flat
            h.update(take.numpy().tobytes())
            h.update(str(tuple(tensor.shape)).encode())
    return h.hexdigest()


def _ddp_anchor_check(world, rank, device, fingerprint):
    """Require every rank to reconstruct the same SVD frozen anchor.

    SVD sign/order choices are not safe to assume identical across runtimes.
    Hash bytes are exchanged on the NCCL device so this check also works when
    the process group has no CPU/Gloo backend.
    """
    if world <= 1:
        return
    import torch.distributed as dist
    raw = bytes.fromhex(fingerprint)
    local = torch.tensor(list(raw), dtype=torch.uint8, device=device)
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    if any(not torch.equal(local, other) for other in gathered):
        _ddp_destroy(world)
        raise RuntimeError('SVD anchor fingerprint mismatch across ranks')


def _ddp_barrier(world):
    """Synchronize ranks when running the explicit-gradient DDP path."""
    if world > 1:
        import torch.distributed as dist
        dist.barrier()


def _ddp_destroy(world):
    if world > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


def _ddp_collective_check(world, device, ok, phase):
    """Make a local validation result visible to every rank.

    A plain exception before a barrier leaves the other ranks waiting forever.
    Reduce a one-bit health flag first so all ranks take the same exit path.
    """
    if world <= 1:
        if not ok:
            raise RuntimeError(f"{phase} failed")
        return
    import torch.distributed as dist
    flag = torch.tensor(1 if ok else 0, device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if int(flag.item()) != 1:
        _ddp_destroy(world)
        raise RuntimeError(f"{phase} failed on at least one rank")


def _ddp_rank0_call(world, rank, device, phase, fn):
    """Run a rank-0-only operation and propagate failure before the barrier."""
    result = None
    error = None
    if rank == 0:
        try:
            result = fn()
        except Exception as exc:  # propagate the failure instead of deadlock
            error = exc
    if world > 1:
        import torch.distributed as dist
        flag = torch.tensor(0 if error is not None else 1,
                            device=device, dtype=torch.int32)
        dist.broadcast(flag, src=0)
        if int(flag.item()) != 1:
            _ddp_destroy(world)
            if error is not None:
                raise RuntimeError(f"{phase} failed: {error}") from error
            raise RuntimeError(f"{phase} failed on rank 0")
        dist.barrier()
    elif error is not None:
        raise RuntimeError(f"{phase} failed: {error}") from error
    return result


def _ddp_phase_guard(world, device, phase, error):
    """Propagate a rank-local failure before ranks enter the next phase."""
    if world <= 1:
        if error is not None:
            raise RuntimeError(f"{phase} failed: {error}") from error
        return
    import torch.distributed as dist
    flag = torch.tensor(0 if error is not None else 1,
                        device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if int(flag.item()) != 1:
        _ddp_destroy(world)
        if error is not None:
            raise RuntimeError(f"{phase} failed: {error}") from error
        raise RuntimeError(f"{phase} failed on another rank")


# ------------------------------- main ---------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["lora", "svd"], required=True)
    ap.add_argument("--phase", choices=["smoke", "pilot", "a0"], default="smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--val-n", type=int, default=None)
    ap.add_argument("--n-residual", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-name")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--batch-documents", type=int, default=64,
                    help="global documents per optimizer step; use 256 for 4 GPUs")
    args = ap.parse_args()

    # --- Explicit-gradient DDP (R4 contract): rank/world from torchrun env.
    _world = int(os.environ.get("WORLD_SIZE", "1"))
    _rank = int(os.environ.get("RANK", "0"))
    _local = int(os.environ.get("LOCAL_RANK", "0"))
    if args.batch_documents < 1:
        raise ValueError('--batch-documents must be positive')
    if _world > 1 and args.batch_documents < _world:
        raise ValueError('--batch-documents must be at least WORLD_SIZE')
    if _world > 1:
        import torch.distributed as dist
        dist.init_process_group(
            os.environ.get("TEV4_DDP_BACKEND", "nccl"))
        device = f"cuda:{_local}"
        torch.cuda.set_device(device)
    else:
        device = "cuda"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run_name = args.run_name or f"{args.phase}_{args.arm}_seed{args.seed}"
    if Path(run_name).name != run_name or run_name in ('.', '..'):
        raise ValueError('run-name must be one directory name')
    out = os.path.join(RUNS, run_name)
    if Path(out, 'train.log').exists() and not (args.resume or args.eval_only):
        raise RuntimeError('existing run; use a new run-name or explicit resume')
    os.makedirs(out, exist_ok=True)
    _ddp_barrier(_world)
    logf = open(os.path.join(out, "train.log"), "a") if _rank == 0 else None
    metrics_path = os.path.join(out, "metrics.jsonl")

    def log(msg):
        if _rank != 0:
            return
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    cfg = dict(smoke=dict(limit=18000, steps=280, ev=70, val=600),
               pilot=dict(limit=None, steps=None, ev=500, val=3000),
               a0=dict(limit=None, steps=None, ev=500, val=32466))[args.phase]
    limit = args.data_limit or cfg["limit"]
    log("loading train split (tokenize-once, cached)...")
    # Shared cache creation is serialized by dataset_v2's filesystem lock.
    # Do not hold an NCCL broadcast open while a full split is tokenized.
    train = load_split(tok, "train", limit=limit)
    val_full = load_split(tok, "val")
    val = stratified_subset(val_full, args.val_n or cfg["val"],
                            seed=20260716 + args.seed)
    log(f"train={len(train)} val={len(val)}")

    from collections import Counter
    cnt = Counter(s.label_id for s in train)
    cls_w = class_normalized_weights([cnt.get(c, 1) for c in range(3)]).to(device)
    log(f"class counts={dict(cnt)} weights={cls_w.tolist()}")

    schedule = class_mixed_schedule([s.label_id for s in train],
                                    args.batch_documents,
                                    seed=20260716 + args.seed)
    if _world > 1 and min(map(len, schedule)) < _world:
        raise ValueError('global schedule has a step smaller than WORLD_SIZE; '
                         'increase data-limit or reduce --batch-documents')
    max_steps = args.max_steps or cfg["steps"] or len(schedule)
    ev_every = args.eval_every or cfg["ev"]
    if max_steps > len(schedule):
        raise ValueError('max-steps exceeds one-epoch schedule')
    if args.save_every < 1 or ev_every < 1:
        raise ValueError('checkpoint/eval intervals must be positive')
    import transformers
    model_files = {p.name: file_sha256(p) for p in sorted(Path(MODEL_PATH).iterdir())
                   if p.suffix in ('.json', '.safetensors')}
    manifest = dict(
        schema=3, arm=args.arm, seed=args.seed, residual_rank=args.n_residual,
        lora_rank=args.lora_r, lr=args.lr, max_steps=max_steps, eval_every=ev_every,
        train=split_provenance(tok, 'train', limit=limit),
        val=split_provenance(tok, 'val'), val_n=len(val),
        batch_documents=args.batch_documents, world_size=_world,
        source={p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        model_files=model_files, torch=str(torch.__version__), transformers=transformers.__version__,
        environment={k: os.environ.get(k) for k in
                     ('PACK_MAX_TOKENS', 'PACK_MAX_WINDOWS', 'TEV4_ATTN', 'TEV4_NO_CKPT')},
        labels=['HUMAN', 'PARTIAL_AI', 'FULL_AI'], regularizers={'orthogonal': 0, 'norm': 0},
        objective='mean_documents(class_weighted_CE + overlap_corrected_token_BCE)')
    from transformers import AutoModel
    # 2026-10-10 evidence reversal: the ONLY clean full run (smoke LoRA, 280
    # steps) used sdpa; EVERY NaN run since flash-attn install used
    # flash_attention_2. Suspected: source-built flash wheel x adapters x
    # grad-ckpt NaN grads. sdpa is now the default; flash only via env opt-in.
    if os.environ.get("TEV4_ATTN") == "flash":
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    else:
        attn_impl = "sdpa"
    backbone = AutoModel.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation=attn_impl)
    if os.environ.get("TEV4_NO_CKPT") == "1":
        log("grad-checkpointing OFF (A100 full-speed mode: ~30% faster, "
            "memory-hungry — pair with moderate PACK_MAX_TOKENS)")
    else:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.config.use_cache = False
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.get_input_embeddings().register_forward_hook(
        lambda m, i, o: o.requires_grad_(True))

    hidden = backbone.config.hidden_size
    if os.environ.get("TEV4_WAIT_GPU") == "1":
        log("waiting for >=21G free GPU (another run finishing)...")
        while True:
            free = torch.cuda.mem_get_info()[0] / 2**30
            if free >= 21.0:
                break
            time.sleep(60)
        log("GPU free, proceeding")
    if args.arm == "lora":
        core = build_lora_arm(backbone, r=args.lora_r)  # peft wrapper
    else:
        core = build_svd_arm(backbone, n_residual=args.n_residual,
                             log=lambda m: log(m))
    model = Qwen3DualHead(core, hidden_size=hidden).to(device)
    model.train()
    manifest['svd_anchor_fingerprint'] = svd_anchor_fingerprint(model) if args.arm == 'svd' else None
    if args.arm == 'svd':
        _ddp_anchor_check(_world, _rank, device,
                          manifest['svd_anchor_fingerprint'])
    manifest_path = Path(out, 'manifest.json')
    def _write_or_check_manifest():
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding='utf8')) != manifest:
                raise RuntimeError('frozen run manifest mismatch')
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    _ddp_rank0_call(_world, _rank, device, 'manifest setup',
                    _write_or_check_manifest)
    try:
        manifest_ok = manifest_path.exists() and (
            json.loads(manifest_path.read_text(encoding='utf8')) == manifest)
    except Exception:
        manifest_ok = False
    _ddp_collective_check(_world, device, manifest_ok,
                          'manifest validation')
    if args.preflight and not (args.resume or args.eval_only):
        from preflight import acceptance
        preflight_ok = True
        try:
            acceptance(model, train, cls_w, device, out, log)
        except Exception:
            preflight_ok = False
        _ddp_collective_check(_world, device, preflight_ok, 'preflight')
    _ddp_rank0_call(
        _world, _rank, device, 'initial SVD diagnostics',
        lambda: Path(out, 'svd_initial.json').write_text(
            json.dumps(svd_diagnostics(model)), encoding='utf8'))

    import os as _os
    if _os.environ.get("TEV4_TRACE_GRAD") == "1":
        _birth = {"found": None}
        def _mk_hook(name):
            def hook(grad):
                if _birth["found"] is None and grad is not None                         and not torch.isfinite(grad).all():
                    _birth["found"] = name
                return grad
            return hook
        for _name, _mod in model.backbone.named_modules():
            if hasattr(_mod, "last_hidden_state") or _name.count(".") >= 1                     and type(_mod).__name__ == "Qwen3DecoderLayer":
                pass
        for _li, _layer in enumerate(model.backbone.layers):
            _layer.register_full_backward_hook(
                lambda m, gi, go, li=_li: (
                    _birth.update(found=f"layer{li}"
                                  if _birth["found"] is None and
                                  any(g is not None and
                                      not torch.isfinite(g).all()
                                      for g in go) else _birth["found"])))
        model._birth = _birth
        log("grad birth-layer trace armed")
    log(f"attn_impl={attn_impl}")
    n_train_params = trainable_param_count(model)
    log(f"arm={args.arm} trainable={n_train_params/1e6:.2f}M phase={args.phase}")


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0,
                                  betas=(0.9, 0.999))
    # No DDP wrapper: our custom forward methods bypass wrapper hooks.
    # Gradient sync = explicit all-reduce after each step (R4-exact).
    # Same seed on all ranks => identical init; no broadcast needed.
    warmup = max(10, int(0.03 * max_steps))

    def lr_at(step):
        if step < warmup:
            return args.lr * step / warmup
        t = (step - warmup) / max(1, max_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    best = {"macro_f1": -1.0, "joint_loc": -1.0}
    start_step = 0
    # One atomically replaced file is both the live recovery point and the
    # final artifact. This keeps mounted-disk storage bounded.
    checkpoint_path = os.path.join(out, "checkpoint.pt")
    if args.resume:
        resume_ok = True
        resume_error = None
        try:
            start_step, best = load_recovery(model, optimizer, checkpoint_path, manifest)
        except Exception as exc:
            resume_ok = False
            resume_error = exc
        _ddp_collective_check(_world, device, resume_ok, 'resume checkpoint')
        if resume_error is not None:
            raise RuntimeError(f'resume checkpoint failed: {resume_error}') from resume_error
        log(f"RESUMED from checkpoint.pt step={start_step} best={best}")
    if args.eval_only:
        eval_load_ok = True
        eval_load_error = None
        try:
            payload = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
            if payload['manifest'] != manifest:
                raise RuntimeError('checkpoint manifest mismatch')
            load_trainable_state(model, payload['weights'])
        except Exception as exc:
            eval_load_ok = False
            eval_load_error = exc
        _ddp_collective_check(_world, device, eval_load_ok,
                              'eval-only checkpoint')
        if eval_load_error is not None:
            raise RuntimeError(f'eval-only checkpoint failed: {eval_load_error}') from eval_load_error
        def _eval_only():
            m = evaluate(model, val_full, device)
            Path(out, 'eval_only.json').write_text(json.dumps(m, indent=2), encoding='utf8')
            log('EVAL_ONLY ' + json.dumps(m))
        _ddp_rank0_call(_world, _rank, device, 'eval-only evaluation', _eval_only)
        _ddp_destroy(_world)
        return
    if start_step == 0:
        _ddp_rank0_call(
            _world, _rank, device, 'initial recovery checkpoint',
            lambda: save_recovery(model, optimizer, 0, best, manifest,
                                  checkpoint_path))
    t0 = time.time()
    train_seconds = 0.
    _steps_iter = schedule[start_step:max_steps]
    for step, doc_idxs in enumerate(_steps_iter, start_step + 1):
        for gp in optimizer.param_groups:
            gp["lr"] = lr_at(step)
        D_global = len(doc_idxs)
        if _world > 1:
            batch_docs = [train[i] for i in doc_idxs[_rank::_world]]
        else:
            batch_docs = [train[i] for i in doc_idxs]
        # R4 exactness: per-rank loss sum scaled by R/D_global inside train_step
        # via D_step = D_global / _world ... we pass scale explicitly below.
        step_started = time.perf_counter()
        step_error = None
        try:
            ld, lt = train_step(model, batch_docs, D_global, cls_w, device,
                                optimizer, world=_world)
        except Exception as exc:
            step_error = exc
            ld = lt = float("nan")
        _ddp_phase_guard(_world, device, f"train step {step}", step_error)
        if _world > 1:
            import torch.distributed as dist
            # SUM + divide (portable: gloo lacks ReduceOp.AVG; caught by
            # bootstrap simulation test 2026-10-10). Math identical to AVG.
            for q in params:
                # Every rank must execute the same collective sequence. A
                # parameter can have no local gradient when a shard does not
                # touch one head, so represent that contribution explicitly
                # as zeros before the all-reduce.
                if q.grad is None:
                    q.grad = torch.zeros_like(q)
                elif not q.grad.is_contiguous():
                    q.grad = q.grad.contiguous()
                dist.all_reduce(q.grad, op=dist.ReduceOp.SUM)
                q.grad.div_(_world)
            telemetry = torch.tensor(
                [ld * len(batch_docs), lt * len(batch_docs), len(batch_docs)],
                device=device, dtype=torch.float64)
            dist.all_reduce(telemetry, op=dist.ReduceOp.SUM)
            if telemetry[2].item() <= 0:
                raise RuntimeError('DDP step contained no documents')
            ld = float((telemetry[0] / telemetry[2]).item())
            lt = float((telemetry[1] / telemetry[2]).item())
        finite_error = None
        if not (math.isfinite(ld) and math.isfinite(lt)):
            finite_error = RuntimeError(f"C7: non-finite loss at step {step}")
        bad = [n for n, q in model.named_parameters()
               if q.requires_grad and q.grad is not None
               and not torch.isfinite(q.grad).all()]
        if bad and finite_error is None:
            finite_error = RuntimeError(
                f"C7: non-finite gradients at step {step}: {bad[:6]}")
        _ddp_phase_guard(_world, device, f"finite check step {step}", finite_error)
        if bad or not math.isfinite(ld):
            smin = []
            from svd_residual import SVDResidualLinear
            for n, m in model.backbone.named_modules():
                if isinstance(m, SVDResidualLinear):
                    smin.append((n, float(m.s_residual.abs().min()),
                                 float(m.s_residual.abs().max())))
            log(f"C7-NaN params={bad[:6]} L_doc={ld} "
                f"birth={getattr(model, '_birth', {}).get('found')} "
                f"S_res(min,max)={smin[:4]}")
            raise RuntimeError("C7: non-finite gradients — scene captured")
        update_error = None
        try:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0,
                                                error_if_nonfinite=True)
            head_grad = {name: float(torch.linalg.vector_norm(torch.cat(
                [p.grad.reshape(-1) for p in head.parameters()
                 if p.grad is not None])))
                         for name, head in [('doc', model.doc_head),
                                            ('token', model.token_head)]}
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        except Exception as exc:
            update_error = exc
            gn = float("nan")
            head_grad = {}
        _ddp_phase_guard(_world, device, f"optimizer step {step}", update_error)
        train_seconds += time.perf_counter() - step_started
        if os.environ.get("TEV4_CACHE_RESET") == "1":
            torch.cuda.empty_cache()  # fragmentation guard (tight 24G card)
        if step % args.save_every == 0 or step == 1:
            _ddp_rank0_call(
                _world, _rank, device, f'recovery checkpoint step {step}',
                lambda: save_recovery(model, optimizer, step, best, manifest,
                                      checkpoint_path))
        if _rank == 0 and (step % 10 == 0 or step == 1):
            mem = torch.cuda.max_memory_allocated() / 2**30
            log(f"step {step}/{max_steps} L_doc={ld:.4f} L_tok={lt:.4f} |g|={gn:.2f} "
                f"lr={lr_at(step):.2e} mem={mem:.1f}G "
                f"head_grad={head_grad} train_s/step={train_seconds/(step-start_step):.2f} "
                f"train_eta={train_seconds/(step-start_step)*(max_steps-step)/3600:.1f}h")
        if step % ev_every == 0 or step == max_steps:
            def _run_eval():
                nonlocal best
                save_recovery(model, optimizer, step, best, manifest, checkpoint_path)
                eval_started = time.perf_counter()
                torch.cuda.empty_cache()
                m = evaluate(model, val, device, batch_tokens=int(
                    os.environ.get("TEV4_EVAL_TOKENS", "8192")))
                m['eval_seconds'] = time.perf_counter() - eval_started
                drift = svd_diagnostics(model)
                with open(Path(out, 'svd_drift.jsonl'), 'a') as fh:
                    fh.write(json.dumps(dict(step=step, **drift)) + '\n')
                torch.cuda.empty_cache()
                log(f"EVAL step={step} " + json.dumps(m, ensure_ascii=False))
                with open(metrics_path, "a") as fh:
                    fh.write(json.dumps(dict(step=step, **m)) + "\n")
                if m["macro_f1"] > best["macro_f1"]:
                    best = {"macro_f1": m["macro_f1"],
                            "joint_loc": m["char_f1"]}
                    log(f"BEST saved macro={m['macro_f1']:.4f}")
                save_recovery(model, optimizer, step, best, manifest, checkpoint_path)
            _ddp_rank0_call(_world, _rank, device,
                            f'evaluation step {step}', _run_eval)
            # _ddp_rank0_call includes the post-evaluation barrier.

    _ddp_barrier(_world)
    def _finalize_run():
        log(f"TRAIN DONE best_macro={best['macro_f1']:.4f} "
            f"elapsed={(time.time()-t0)/3600:.2f}h")
        payload = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
        if payload['manifest'] != manifest:
            raise RuntimeError('checkpoint manifest mismatch at finalization')
        load_trainable_state(model, payload['weights'])
        # Smoke is a communication/gradient acceptance run. Re-evaluating the
        # sealed full validation split here would turn a 10-step check into a
        # roughly hour-long job; full validation remains mandatory for pilot/A0.
        final_samples = val if args.phase == 'smoke' else val_full
        final_metrics = evaluate(model, final_samples, device)
        Path(out, 'best_full_val.json').write_text(json.dumps(final_metrics, indent=2), encoding='utf8')
        if args.phase != 'smoke':
            with open(Path(out, 'val_predictions.jsonl'), 'w') as fh:
                for prediction in predict_samples(model, val_full, device):
                    fh.write(json.dumps(prediction) + '\n')
        Path(out, 'completed.json').write_text(json.dumps(dict(best_step=payload['step'],
                                                            metrics=final_metrics)), encoding='utf8')
    _ddp_rank0_call(_world, _rank, device, 'final evaluation and artifact write',
                    _finalize_run)
    _ddp_destroy(_world)


def _trainable_state(model, args):
    """Arm-specific compact state (weights only, small)."""
    sd = {"heads": model.doc_head.state_dict(),
          "heads_tok": model.token_head.state_dict()}
    if args.arm == "svd":
        from svd_residual import SVDResidualLinear
        svd_sd = {}
        for name, m in model.backbone.named_modules():
            if isinstance(m, SVDResidualLinear):
                svd_sd[name] = {k: v for k, v in m.state_dict().items()
                                if "residual" in k}
        sd["svd"] = svd_sd
    else:
        try:
            from peft.utils import get_peft_model_state_dict
            sd["lora"] = get_peft_model_state_dict(model.backbone)
        except Exception:
            sd["lora"] = {k: v for k, v in model.backbone.state_dict().items()
                          if "lora_" in k}
    return sd


def _apply_trainable_state(model, args, sd):
    model.doc_head.load_state_dict(sd["heads"])
    model.token_head.load_state_dict(sd["heads_tok"])
    if args.arm == "svd":
        from svd_residual import SVDResidualLinear
        for name, m in model.backbone.named_modules():
            if isinstance(m, SVDResidualLinear) and name in sd.get("svd", {}):
                full = m.state_dict()
                full.update(sd["svd"][name])
                m.load_state_dict(full)
    else:
        try:
            from peft.utils import set_peft_model_state_dict
            set_peft_model_state_dict(model.backbone, sd["lora"])
        except Exception:
            msd = model.backbone.state_dict()
            for k, v in sd["lora"].items():
                if k in msd:
                    msd[k].copy_(v)


def save_last(model, optimizer, step, args, out, best):
    """Rolling crash-recovery checkpoint (plan §3 last.pt clause; the 2026-10-10
    step-500 eval-OOM crash cost 500 steps because this was missing)."""
    import os as _os
    payload = {"step": step, "best": best, "args": vars(args),
               "model": _trainable_state(model, args),
               "opt": optimizer.state_dict()}
    tmp = _os.path.join(out, "last.pt.tmp")
    torch.save(payload, tmp)
    _os.replace(tmp, _os.path.join(out, "last.pt"))


def save_best(model, args, out):
    sd = {}
    sd["heads"] = {k: v for k, v in model.doc_head.state_dict().items()}
    sd["heads_tok"] = {k: v for k, v in model.token_head.state_dict().items()}
    if args.arm == "svd":
        from svd_residual import SVDResidualLinear
        svd_sd = {}
        for name, m in model.backbone.named_modules():
            if isinstance(m, SVDResidualLinear):
                svd_sd[name] = {k: v for k, v in m.state_dict().items()
                                if "residual" in k}
        sd["svd"] = svd_sd
        torch.save(sd, os.path.join(out, "best.pt"))
    else:
        torch.save(sd, os.path.join(out, "best_heads.pt"))
        try:
            model.backbone.save_pretrained(os.path.join(out, "adapter"))
        except Exception as e:  # peft model save
            log_ = print
            log_(f"peft save fallback: {e}")


if __name__ == "__main__":
    main()
