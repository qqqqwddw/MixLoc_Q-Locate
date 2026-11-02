"""Independent CPU recomputation of Q-Locate eval exports.

Reads only the exported .f32 arrays, rank JSONL, and the Store text/GT.
Does NOT import or call the training evaluator's Metrics.update or decode.
Implements its own two-state DP (beta=.25, float64, machine-predecessor and
machine-final tie rules), raw mask, doc-gated mask, greedy span matching
(IoU>=.5; on equal IoU the later unmatched GT wins), and MAE = endpoint
error sum / (2 * matched spans). Integer counts must match exactly; floats
derived from identical integers must agree within 1e-8.
"""
import json
import math
from pathlib import Path
import numpy as np


def prf(tp, fp, fn):
    tp, fp, fn = int(tp), int(fp), int(fn)
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / max(1, tp + fp), 'recall': tp / max(1, tp + fn),
            'f1': 2 * tp / max(1, 2 * tp + fp + fn)}


def spans_of(mask):
    m = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False, m, False].astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def independent_decode(alpha, a, beta=.25):
    """Two-state max recursion in float64; ties select the machine path."""
    alpha = np.asarray(alpha, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    n = len(alpha)
    back = np.zeros((n, 2), dtype=np.int8)
    h, m = 0.0, -math.inf
    for i in range(n):
        from_m_h = (m + beta * a[i, 1]) >= h
        from_m_m = m >= (h + beta * a[i, 0])
        nh = -np.logaddexp(0.0, alpha[i]) + (m + beta * a[i, 1] if from_m_h else h)
        nm = -np.logaddexp(0.0, -alpha[i]) + (m if from_m_m else h + beta * a[i, 0])
        back[i] = (int(from_m_h), int(from_m_m))
        h, m = nh, nm
    state = int((m + beta * a[n, 1]) >= h)
    mask = np.zeros(n, dtype=bool)
    for i in range(n - 1, -1, -1):
        mask[i] = bool(state)
        state = int(back[i, state])
    return mask


def greedy_match(pred_spans, true_spans):
    """Greedy one-to-one IoU>=.5 in prediction order; equal IoU keeps the later GT."""
    used = set()
    matched = exact = err_sum = 0
    if len(set(pred_spans)) != len(pred_spans):
        raise ValueError('repeated prediction interval')
    for ps, pe in pred_spans:
        best = .5
        pick = None
        for j, (gs, ge) in enumerate(true_spans):
            if j in used:
                continue
            inter = max(0, min(pe, ge) - max(ps, gs))
            union = pe - ps + ge - gs - inter
            iou = inter / union if union else 0.
            if iou >= best:
                best, pick = iou, j
        if pick is not None:
            used.add(pick)
            matched += 1
            gs, ge = true_spans[pick]
            err_sum += abs(ps - gs) + abs(pe - ge)
        if (ps, pe) in set(map(tuple, true_spans)):
            exact += 1
    return {'predicted': len(pred_spans), 'gold': len(true_spans), 'exact': exact,
            'matched': matched, 'endpoint_error_sum': err_sum}


class IndependentMetrics:
    def __init__(self):
        self.matrix = np.zeros((3, 3), dtype=np.int64)
        self.char = {k: np.zeros(3, dtype=np.int64) for k in ('raw', 'decoded', 'gated')}
        self.e2e = np.zeros(3, dtype=np.int64)
        self.doc_mean = {k: [] for k in ('raw', 'decoded', 'gated')}
        self.span = {k: np.zeros(5, dtype=np.int64) for k in ('raw', 'decoded')}  # pred,gold,exact,matched,err
        self.span_length = {k: np.zeros(3, dtype=np.int64) for k in
                            ('1-32', '33-128', '129-512', '513+')}

    def update(self, doc, logits, alpha, b, a):
        pred = int(np.argmax(logits))
        label = doc.label
        n = len(doc.text)
        self.matrix[label, pred] += 1
        raw = np.asarray(alpha) >= 0
        decoded = independent_decode(alpha, a)
        gated = np.zeros(n, dtype=bool) if pred == 0 else np.ones(n, dtype=bool) if pred == 2 else decoded
        gold = np.zeros(n, dtype=bool)
        for s, e in doc.spans:
            gold[s:e] = True
        for key, mask in (('raw', raw), ('decoded', decoded), ('gated', gated)):
            counts = np.array([(gold & mask).sum(), (~gold & mask).sum(), (gold & ~mask).sum()], dtype=np.int64)
            if label == 1:
                self.char[key] += counts
                self.doc_mean[key].append(prf(*counts)['f1'])
            if key == 'gated':
                self.e2e += counts
        if label == 1:
            true = [tuple(map(int, sp)) for sp in doc.spans]
            for key, mask in (('raw', raw), ('decoded', decoded)):
                m = greedy_match(spans_of(mask), true)
                self.span[key] += np.array([m['predicted'], m['gold'], m['exact'],
                                            m['matched'], m['endpoint_error_sum']], dtype=np.int64)
            used = set()
            for ps, pe in spans_of(raw):
                best = .5
                pick = None
                for j, (gs, ge) in enumerate(true):
                    if j in used:
                        continue
                    inter = max(0, min(pe, ge) - max(ps, gs))
                    union = pe - ps + ge - gs - inter
                    iou = inter / union if union else 0.
                    if iou >= best:
                        best, pick = iou, j
                if pick is not None:
                    used.add(pick)
            for j, (gs, ge) in enumerate(true):
                size = ge - gs
                bucket = '1-32' if size <= 32 else '33-128' if size <= 128 else '129-512' if size <= 512 else '513+'
                self.span_length[bucket][0] += 1
                self.span_length[bucket][1] += int(j in used)

    def result(self):
        per = [prf(self.matrix[c, c], self.matrix[:, c].sum() - self.matrix[c, c],
                   self.matrix[c, :].sum() - self.matrix[c, c]) for c in range(3)]
        spans = {}
        for key, (pc, gc, exact, matched, err) in self.span.items():
            spans[key] = {'predicted': int(pc), 'gold': int(gc),
                          'exact': prf(exact, pc - exact, gc - exact),
                          'iou50': prf(matched, pc - matched, gc - matched),
                          'boundary_mae': err / (2 * matched) if matched else None,
                          'matched_spans': int(matched)}
        return {'n_documents': int(self.matrix.sum()),
                'accuracy': float(np.trace(self.matrix) / max(1, self.matrix.sum())),
                'macro_f1': sum(x['f1'] for x in per) / 3,
                'balanced_accuracy': sum(x['recall'] for x in per) / 3,
                'per_class': per, 'confusion_matrix': self.matrix.tolist(),
                'character': {k: prf(*v) for k, v in self.char.items()},
                'character_e2e': prf(*self.e2e),
                'spans': spans,
                'partial_doc_mean_f1': {k: float(np.mean(v)) if v else None for k, v in self.doc_mean.items()},
                'span_length_recall_counts': {k: v.tolist() for k, v in self.span_length.items()}}

    def compare(self, exported):
        """Integer counts exact; derived floats <=1e-8. Returns diff report."""
        issues = []

        def cmp_int(path, a, b):
            if int(a) != int(b):
                issues.append(f'{path}: {a} != {b}')

        def cmp_float(path, a, b):
            if a is None or b is None:
                if a is not b:
                    issues.append(f'{path}: {a} vs {b}')
                return
            if abs(float(a) - float(b)) > 1e-8:
                issues.append(f'{path}: |{a}-{b}|>1e-8')

        mine = self.result()
        cmp_int('n_documents', mine['n_documents'], exported['n_documents'])
        if mine['confusion_matrix'] != exported['confusion_matrix']:
            issues.append('confusion_matrix differs')
        for i, (x, y) in enumerate(zip(mine['per_class'], exported['per_class'])):
            for k in ('tp', 'fp', 'fn'):
                cmp_int(f'per_class{i}.{k}', x[k], y[k])
            for k in ('precision', 'recall', 'f1'):
                cmp_float(f'per_class{i}.{k}', x[k], y[k])
        cmp_float('accuracy', mine['accuracy'], exported['accuracy'])
        cmp_float('macro_f1', mine['macro_f1'], exported['macro_f1'])
        cmp_float('balanced_accuracy', mine['balanced_accuracy'], exported['balanced_accuracy'])
        for key in ('raw', 'decoded', 'gated'):
            for k in ('tp', 'fp', 'fn'):
                cmp_int(f'character.{key}.{k}', mine['character'][key][k], exported['character'][key][k])
            for k in ('precision', 'recall', 'f1'):
                cmp_float(f'character.{key}.{k}', mine['character'][key][k], exported['character'][key][k])
        for k in ('tp', 'fp', 'fn'):
            cmp_int(f'character_e2e.{k}', mine['character_e2e'][k], exported['character']['e2e'][k])
        for k in ('precision', 'recall', 'f1'):
            cmp_float(f'character_e2e.{k}', mine['character_e2e'][k], exported['character']['e2e'][k])
        for key in ('raw', 'decoded'):
            ms, es = mine['spans'][key], exported['spans'][key]
            for k in ('predicted', 'gold', 'matched_spans'):
                cmp_int(f'spans.{key}.{k}', ms[k], es[k])
            cmp_int(f'spans.{key}.exact.tp', ms['exact']['tp'], es['exact']['tp'])
            for k in ('precision', 'recall', 'f1'):
                cmp_float(f'spans.{key}.exact.{k}', ms['exact'][k], es['exact'][k])
                cmp_float(f'spans.{key}.iou50.{k}', ms['iou50'][k], es['iou50'][k])
            cmp_float(f'spans.{key}.boundary_mae', ms['boundary_mae'], es['boundary_mae'])
        for key in ('raw', 'decoded', 'gated'):
            cmp_float(f'partial_doc_mean_f1.{key}', mine['partial_doc_mean_f1'][key],
                      exported['partial_doc_mean_f1'][key])
        for bucket, v in mine['span_length_recall_counts'].items():
            if v != exported['span_length_recall_counts'].get(bucket):
                issues.append(f'span_length.{bucket}: {v} != {exported["span_length_recall_counts"].get(bucket)}')
        return {'status': 'PASS' if not issues else 'FAIL', 'issues': issues,
                'integers_exact': not any('!=' in s for s in issues),
                'float_tolerance': '1e-8'}


def recompute_directory(target, store, ids, world):
    """Full-file validation + recomputation over one eval_step directory."""
    target = Path(target)
    seen = set()
    metrics = IndependentMetrics()
    layout = []
    for rank in range(world):
        data = np.memmap(target / f'rank{rank}.f32', dtype='<f4', mode='r')
        cursor = 0
        with (target / f'rank{rank}.jsonl').open(encoding='utf-8') as file:
            for line in file:
                record = json.loads(line)
                doc = store.get(record['doc_id'])
                if record['doc_id'] in seen:
                    raise ValueError('duplicate id')
                seen.add(record['doc_id'])
                if record['label'] != doc.label or record['n_chars'] != len(doc.text):
                    raise ValueError('jsonl label/n_chars mismatch')
                values = {}
                expected = {'doc_logits': (3,), 'alpha': (record['n_chars'],),
                            'b': (record['n_chars'] + 1, 2), 'a': (record['n_chars'] + 1, 2)}
                for key, shape in expected.items():
                    sl = record['arrays'][key]
                    size = int(np.prod(shape))
                    if sl['shape'] != list(shape):
                        raise ValueError(f'{key} shape {sl["shape"]} != {shape}')
                    if sl['offset'] != cursor:
                        raise ValueError(f'{key} offset gap/overlap at {cursor}')
                    arr = np.asarray(data[sl['offset']:sl['offset'] + size], dtype=np.float32).reshape(shape)
                    if not np.isfinite(arr).all():
                        raise ValueError(f'{key} nonfinite')
                    values[key] = arr
                    cursor += size
                if cursor > data.size:
                    raise ValueError('array file overrun')
                layout.append({'rank': rank, 'doc_id': record['doc_id'], 'elements': cursor})
                metrics.update(doc, values['doc_logits'], values['alpha'], values['b'], values['a'])
                # exported span lists must equal independent decode of exported arrays
                if spans_of(values['alpha'] >= 0) != [tuple(x) for x in record['raw_spans']]:
                    raise ValueError('exported raw_spans mismatch')
                if spans_of(independent_decode(values['alpha'], values['a'])) != [tuple(x) for x in record['decoded_spans']]:
                    raise ValueError('exported decoded_spans mismatch')
                if int(np.argmax(values['doc_logits'])) != record['prediction']:
                    raise ValueError('exported prediction mismatch')
        if cursor != data.size:
            raise ValueError(f'rank{rank} tail residue: {data.size - cursor} elements')
        del data
    if seen != set(ids):
        raise ValueError('id coverage differs from evaluation set')
    return metrics, {'docs': len(seen), 'rank_layout_elements': layout}
