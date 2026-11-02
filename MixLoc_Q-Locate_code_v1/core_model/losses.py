"""One document objective; caller applies exactly one global-document scale."""
import torch
from torch.nn import functional as F


def class_weights(device='cpu', dtype=torch.float32):
    inverse = 1 / torch.tensor([192001, 79900, 94717], dtype=torch.float64)
    return (inverse / inverse.mean()).to(device=device, dtype=dtype)


def document_loss(out, mapping, weights):
    alpha, p, b, a = (out[k] for k in ('alpha', 'p', 'b', 'a'))
    label, spans, n = mapping['label'], mapping['spans'], mapping['n']
    y = mapping['target']
    ce = -weights[label] * F.log_softmax(out['doc_logits'], -1)[label]
    char_bce = F.binary_cross_entropy_with_logits(alpha, y, reduction='none')
    if label == 1:
        positive = torch.stack([char_bce[start:end].mean() for start, end in spans]).mean()
        negative = char_bce[y == 0].mean()
        loc = .5 * (positive + negative)
        dice = 1 - (2 * (p*y).sum()+1) / (p.sum()+y.sum()+1)
    else:
        loc = char_bce.mean(); dice = alpha.sum()*0
    bce_channels, rank_channels = [], []
    for channel in (0, 1):
        positives = [span[channel] for span in spans]
        target = torch.zeros_like(b[:, channel])
        if positives:
            target[positives] = 1
        terms = F.binary_cross_entropy_with_logits(b[:, channel], target, reduction='none')
        bc = .5*(terms[target == 1].mean()+terms[target == 0].mean()) if positives else terms.mean()
        bce_channels.append(bc)
        if positives:
            ps = set(positives); ranks = []
            start, end = (0, n-1) if channel == 0 else (1, n)
            for true in positives:
                candidates = [h for h in range(max(start, true-8), min(end, true+8)+1)
                              if h == true or h not in ps]
                ranks.append(torch.logsumexp(a[candidates, channel], 0)-a[true, channel])
            rank_channels.append(torch.stack(ranks).mean())
        else:
            rank_channels.append(a[:, channel].sum()*0)
    boundary_bce = torch.stack(bce_channels).mean()
    boundary_rank = torch.stack(rank_channels).mean()
    total = ce + loc + .2*dice + .1*(boundary_bce + .25*boundary_rank)
    return total, {'doc': ce, 'loc': loc, 'dice': dice,
                   'boundary_bce': boundary_bce, 'boundary_rank': boundary_rank,
                   'total': total}
