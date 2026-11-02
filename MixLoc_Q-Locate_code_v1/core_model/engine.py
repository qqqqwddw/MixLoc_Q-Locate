"""Ordinary graph and exact compact replay, independent of world size."""
import random
import numpy as np
import torch
import torch.distributed as dist
from losses import document_loss


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, device):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def window_forward(model, doc, a, b, device):
    ids = torch.tensor(doc.ids[a:b].astype(np.int64), dtype=torch.long, device=device)[None]
    mask = torch.ones_like(ids)
    p, v, last = model.forward_windows(ids, mask)
    return p[0], v[0], last[0]


def compact_forward(model, doc, device, capture=False, compare_outputs=False):
    dtype = model.doc_head.weight.dtype
    p = torch.zeros(len(doc.ids), dtype=dtype, device=device)
    v = torch.zeros((len(doc.ids), 128), dtype=dtype, device=device)
    last = torch.zeros(model.doc_head.in_features, dtype=dtype, device=device)
    records = []
    for start, end in doc.windows:
        state = rng_state(device) if capture else None
        po, vo, ho = window_forward(model, doc, start, end, device)
        p[start:end] += po
        v[start:end] += vo
        last += ho / len(doc.windows)
        if capture:
            records.append((state, (po.detach().cpu(), vo.detach().cpu(), ho.detach().cpu())
                            if compare_outputs else None))
    mult = torch.tensor(doc.multiplicity.astype(np.float64), device=device, dtype=dtype)
    return (p / mult, v / mult[:, None], last), records


def full_graph(model, doc, mapping, device):
    compact, _ = compact_forward(model, doc, device)
    return model.from_compact(*compact, mapping)


def backward_document(model, doc, weights, scale=1., force_replay=False,
                      check_replay=False):
    device = model.doc_head.weight.device
    mapping = doc.tensors(device, model.doc_head.weight.dtype)
    use_replay = force_replay or len(doc.ids) > 4096
    if not use_replay:
        output = full_graph(model, doc, mapping, device)
        loss, parts = document_loss(output, mapping, weights)
        (loss*scale).backward()
        return {k: float(v.detach()) for k,v in parts.items()}, {'replay':False, 'windows':len(doc.windows)}
    with torch.no_grad():
        compact, records = compact_forward(model, doc, device, capture=True, compare_outputs=check_replay)
    leaves = [x.detach().requires_grad_(True) for x in compact]
    output = model.from_compact(*leaves, mapping)
    loss, parts = document_loss(output, mapping, weights)
    dtypes={k:str(output[k].dtype) for k in ('alpha','p','b','a','doc_logits')}
    dtypes['loss']=str(loss.dtype)
    (loss*scale).backward()
    values = {k:float(v.detach()) for k,v in parts.items()}
    if any(x.grad is None for x in leaves):
        raise RuntimeError('missing compact gradient')
    gradients = [x.grad.detach() for x in leaves]
    del output, loss, leaves
    max_errors = [0.,0.,0.]
    max_relative = [0.,0.,0.]
    post_rng = rng_state(device)
    for wi,(start,end) in enumerate(doc.windows):
        restore_rng(records[wi][0], device)
        po,vo,ho = window_forward(model,doc,start,end,device)
        if check_replay:
            for j,(actual,reference) in enumerate(zip((po,vo,ho),records[wi][1])):
                ref=reference.to(actual)
                error=(actual.detach()-ref).abs()
                max_errors[j]=max(max_errors[j],float(error.max()))
                max_relative[j]=max(max_relative[j],float(error.norm()/ref.norm().clamp_min(1e-30)))
                if not bool((error <= 1e-6+1e-5*ref.abs()).all()):
                    raise RuntimeError(f'cache/replay forward mismatch window={wi} channel={j} max={max_errors[j]}')
        mult=mapping['mult'][start:end]
        carrier=(gradients[0][start:end]*po/mult).sum()
        carrier=carrier+(gradients[1][start:end]*vo/mult[:,None]).sum()
        carrier=carrier+(gradients[2]*ho).sum()/len(doc.windows)
        carrier.backward()
    restore_rng(post_rng,device)
    return values,{'replay':True,'windows':len(doc.windows),'first_pass_windows':len(doc.windows),
                   'replay_windows':len(doc.windows),'forward_max_abs':max_errors,
                   'forward_max_relative_l2':max_relative,'forward_compared':check_replay,
                   'input_state':{'windows':doc.windows,'position_ids':'local range(0,window_length)',
                                  'attention_mask':'all ones; no padding; window microbatch 1',
                                  'occurrences':sum(b-a for a,b in doc.windows),
                                  'canonical_tokens':len(doc.ids),'rng_per_window':True,
                                  'checkpoint':'all backbone layers, nonreentrant; chunked heads'},
                   'head_dtypes':dtypes}


def sync_gradients(model):
    world=dist.get_world_size() if dist.is_initialized() else 1
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad=torch.zeros_like(parameter)
        if not parameter.grad.is_contiguous():
            parameter.grad=parameter.grad.contiguous()
        if world>1:
            dist.all_reduce(parameter.grad,op=dist.ReduceOp.SUM)
            parameter.grad.div_(world)


def train_step(model, documents, weights, global_count, check_replay=False):
    world=dist.get_world_size() if dist.is_initialized() else 1
    totals = {k:0. for k in ('doc','loc','dice','boundary_bce','boundary_rank','total')}
    details=[]
    groups=[];current=[];maxlen=0
    for doc in documents:
        length=len(doc.ids)
        if length>4096:
            if current:groups.append(current);current=[];maxlen=0
            groups.append([doc]);continue
        if current and (len(current)>=8 or (len(current)+1)*max(maxlen,length)>15360):
            groups.append(current);current=[];maxlen=0
        current.append(doc);maxlen=max(maxlen,length)
    if current:groups.append(current)
    for group in groups:
        if weights.is_cuda:
            torch.cuda.reset_peak_memory_stats(weights.device)
        if len(group)==1:
            doc=group[0]
            parts,detail=backward_document(model,doc,weights,world/global_count,check_replay=check_replay)
            for k,v in parts.items():totals[k]+=v
            detail.update(id=doc.doc_id,chars=len(doc.text),tokens=len(doc.ids))
        else:
            device=weights.device
            ids=torch.nn.utils.rnn.pad_sequence([torch.tensor(doc.ids.astype(np.int64),device=device) for doc in group],batch_first=True)
            lengths=torch.tensor([len(doc.ids) for doc in group],device=device)
            mask=(torch.arange(ids.shape[1],device=device)[None]<lengths[:,None]).long()
            p,v,last=model.forward_windows(ids,mask)
            losses=[]
            for i,doc in enumerate(group):
                mp=doc.tensors(device,model.doc_head.weight.dtype)
                out=model.from_compact(p[i,:len(doc.ids)],v[i,:len(doc.ids)],last[i],mp)
                loss,parts=document_loss(out,mp,weights);losses.append(loss)
                for k,value in parts.items():totals[k]+=float(value.detach())
            (sum(losses)*(world/global_count)).backward()
            detail={'replay':False,'pack_documents':len(group),'padded_tokens':int(ids.numel()),
                    'ids':[d.doc_id for d in group],'windows':len(group)}
        if weights.is_cuda:
            detail.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(weights.device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(weights.device))
            from pathlib import Path
            cg=Path('/sys/fs/cgroup')
            detail['cgroup_current_bytes']=int((cg/'memory.current').read_text()) if (cg/'memory.current').exists() else None
            detail['cgroup_peak_bytes']=int((cg/'memory.peak').read_text()) if (cg/'memory.peak').exists() else None
        details.append(detail)
    sync_gradients(model)
    values=torch.tensor(list(totals.values()),device=weights.device,dtype=torch.float64)
    if world>1:dist.all_reduce(values)
    return dict(zip(totals,(values/global_count).tolist())),details
