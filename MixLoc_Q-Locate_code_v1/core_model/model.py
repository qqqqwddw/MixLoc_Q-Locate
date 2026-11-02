"""Q-Locate v1.1 fixed Top-4 architecture, including the compact character graph."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def top4_mean(values):
    """Score descending; equal scores select the earlier character start.

    Selection is discrete and detached; selected original scores receive 1/K.
    No epsilon perturbation and no gradients to unselected scores.
    """
    k = min(4, values.numel())
    if k == values.numel():
        return values.mean()
    with torch.no_grad():
        cut = torch.topk(values, k, sorted=True).values[-1]
        greater = values > cut
        tied = values == cut
        chosen = greater | (tied & (tied.long().cumsum(0) <= k - greater.sum()))
    return values[chosen].mean()


def relative_boundaries(p, b):
    delta = F.pad(p, (0, 1)) - F.pad(p, (1, 0))
    t = b + torch.stack((2 * delta, -2 * delta), dim=-1)
    n = p.numel()
    a_s = t[:-1, 0] - (torch.logsumexp(t[:-1, 0], 0) - p.new_tensor(n).log())
    a_e = t[1:, 1] - (torch.logsumexp(t[1:, 1], 0) - p.new_tensor(n).log())
    return torch.stack((F.pad(a_s, (0, 1)), F.pad(a_e, (1, 0))), dim=-1)


def local_window_means(p, width, chunk=8192):
    """Reset prefix sums each block; right halo preserves all window starts."""
    n = min(width, p.numel())
    count = p.numel() - n + 1
    outputs = []
    for lo in range(0, count, chunk):
        hi = min(lo + chunk, count)
        prefix = F.pad(p[lo:hi + n - 1].cumsum(0), (1, 0))
        outputs.append((prefix[n:] - prefix[:-n]) / n)
    return torch.cat(outputs)


def scan_evidence(p, a, chunk=8192):
    cs, ss = [], []
    for scale in (8, 32, 128):
        n = min(scale, p.numel())
        c = local_window_means(p, n, chunk).square()
        support = (.5 * (F.logsigmoid(a[:len(c), 0])
                          + F.logsigmoid(a[n:n + len(c), 1]))).exp()
        cs.append(top4_mean(c)); ss.append(top4_mean(c * support))
    return torch.stack(cs + ss)


class QLocate(nn.Module):
    def __init__(self, backbone, hidden=2560, char_chunk=8192):
        super().__init__()
        self.backbone = backbone
        # Preserve legacy initialization order.
        self.doc_head = nn.Linear(hidden, 3)
        self.token_head = nn.Linear(hidden, 1)
        self.projection = nn.Linear(hidden, 128)
        self.norm = nn.LayerNorm(128, eps=1e-5)
        self.byte_embedding = nn.Embedding(257, 16, padding_idx=256)
        self.source = nn.Sequential(nn.Linear(708, 128), nn.GELU(), nn.Linear(128, 1))
        self.boundary = nn.Sequential(nn.Linear(522, 128), nn.GELU(), nn.Linear(128, 2))
        self.document = nn.Sequential(nn.Linear(392, 256), nn.GELU(), nn.Linear(256, 3))
        for head in (self.source, self.boundary, self.document):
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
        self.char_chunk = char_chunk
        self.checkpoint_heads = True

    def forward_windows(self, ids, mask):
        pos = (mask.long().cumsum(-1) - 1).clamp_min(0)
        h = self.backbone(input_ids=ids, attention_mask=mask,
                          position_ids=pos, use_cache=False).last_hidden_state
        context = torch.autocast('cuda', enabled=False) if h.is_cuda else nullcontext()
        with context:
            # Oracle models can use double. Production heads use float32.
            h = h.to(self.doc_head.weight.dtype)
            p = self.token_head(h).squeeze(-1).sigmoid()
            v = self.projection(h)
            lengths = mask.long().sum(-1) - 1
            last = h[torch.arange(len(ids), device=h.device), lengths]
        return p, v, last

    @staticmethod
    def _token_context(u, first, last):
        # Bounded local prefixes avoid cancellation on extreme canonical axes.
        pieces = []
        t = len(u)
        for lo in range(first, last, 8192):
            hi = min(lo + 8192, last); begin = max(0, lo - 32); end = min(t, hi + 32)
            prefix = F.pad(u[begin:end].cumsum(0), (0, 0, 1, 0))
            j = torch.arange(lo, hi, device=u.device)
            channels = [u[lo:hi]]
            for r in (8, 32):
                left = (j - r).clamp_min(0); right = (j + r + 1).clamp_max(t)
                channels += [(prefix[j-begin] - prefix[left-begin]) / (j-left).clamp_min(1)[:, None],
                             (prefix[right-begin] - prefix[j+1-begin]) / (right-j-1).clamp_min(1)[:, None]]
            pieces.append(torch.cat(channels, -1))
        return torch.cat(pieces)

    @staticmethod
    def _gather(values, mapping, lo, hi, token_offset=0):
        start, end = int(mapping['ptr_cpu'][lo]), int(mapping['ptr_cpu'][hi])
        selected = values[mapping['tokens'][start:end] - token_offset]
        result = values.new_zeros((hi - lo,) + values.shape[1:])
        result = result.index_add(0, mapping['rows'][start:end] - lo, selected)
        divisor = mapping['widths'][lo:hi]
        return result / divisor.reshape((-1,) + (1,) * (values.ndim - 1))

    def _maybe_checkpoint(self, function, *args):
        if self.training and torch.is_grad_enabled() and self.checkpoint_heads:
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def from_compact(self, p_token, v_token, last, mapping):
        u = self.norm(v_token)
        # q is T x 640, never N x 708; for long axes recompute neighborhoods
        # inside character checkpoint closures instead of persisting q.
        n = mapping['n']
        outputs = []
        for lo in range(0, n, self.char_chunk):
            hi = min(lo + self.char_chunk, n)
            def source_chunk(p_token, u, lo=lo, hi=hi):
                selected = mapping['tokens_cpu'][mapping['ptr_cpu'][lo]:mapping['ptr_cpu'][hi]]
                first, last = int(selected.min()), int(selected.max()) + 1
                q = self._token_context(u, first, last)
                q_char = self._gather(q, mapping, lo, hi, token_offset=first)
                u_char = q_char[:, :128]
                p0 = self._gather(p_token, mapping, lo, hi)
                emb = self.byte_embedding(mapping['bytes'][lo:hi]).flatten(1)
                x = torch.cat((q_char, emb, mapping['xi'][lo:hi],
                               mapping['uncovered'][lo:hi, None], p0[:, None]), -1)
                alpha = torch.logit(p0.clamp(1e-6, 1-1e-6)) + self.source(x).squeeze(-1)
                p = alpha.sigmoid()
                sums = torch.stack((u_char.sum(0), (p[:, None]*u_char).sum(0),
                                    ((1-p[:, None])*u_char).sum(0)))
                return alpha, sums
            outputs.append(self._maybe_checkpoint(source_chunk, p_token, u))
        alpha = torch.cat([x[0] for x in outputs]); p = alpha.sigmoid()
        sums = torch.stack([x[1] for x in outputs]).sum(0)
        b_parts = []
        for lo in range(0, n + 1, self.char_chunk):
            hi = min(lo + self.char_chunk, n + 1)
            def boundary_chunk(u, p, lo=lo, hi=hi):
                c_lo, c_hi = max(0, lo-1), min(n, hi)
                uc = self._gather(u, mapping, c_lo, c_hi)
                g = torch.arange(lo, hi, device=p.device)
                l = (g-1).clamp(0, n-1); r = g.clamp(0, n-1)
                has_l, has_r = (g > 0), (g < n)
                ul = uc[l-c_lo]*has_l[:, None]; ur = uc[r-c_lo]*has_r[:, None]
                el = self.byte_embedding(mapping['bytes'][l]).flatten(1)*has_l[:, None]
                er = self.byte_embedding(mapping['bytes'][r]).flatten(1)*has_r[:, None]
                xi_l = mapping['xi'][l]*has_l[:, None]; xi_r = mapping['xi'][r]*has_r[:, None]
                x = torch.cat((ul, ur, ur-ul, el, er,
                               (p[l]*has_l)[:, None], (p[r]*has_r)[:, None],
                               (~has_l)[:, None], (~has_r)[:, None], xi_l, xi_r,
                               (mapping['uncovered'][l]*has_l)[:, None],
                               (mapping['uncovered'][r]*has_r)[:, None]), -1)
                return self.boundary(x)
            b_parts.append(self._maybe_checkpoint(boundary_chunk, u, p))
        b = torch.cat(b_parts); a = relative_boundaries(p, b)
        pools = torch.cat((sums[0]/n, sums[1]/(p.sum()+1e-6),
                           sums[2]/((1-p).sum()+1e-6)))
        stats = torch.stack((p.mean(), p.var(unbiased=False)))
        evidence = torch.cat((pools, stats, scan_evidence(p, a, self.char_chunk)))
        logits = self.doc_head(last) + self.document(evidence)
        return {'alpha': alpha, 'p': p, 'b': b, 'a': a, 'doc_logits': logits,
                'evidence': evidence, 'old_doc_logits': self.doc_head(last)}
