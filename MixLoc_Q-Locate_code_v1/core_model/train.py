"""Shared single/four GPU functional and full training entry; no implicit launch."""
import argparse
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from data import Store,Document
from engine import train_step,backward_document,full_graph,rng_state,restore_rng
from losses import class_weights
from runtime import load_model,save_checkpoint,load_checkpoint,lr_for,atomic_json,FP32TailLinear
from evaluate import evaluate


def equal_nested(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_nested(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_nested(x,y) for x,y in zip(a,b))
    return a==b


def gradient_report(model):
    groups={}
    finite=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if p.grad is None:raise RuntimeError('missing synchronized gradient '+name)
        finite.append(torch.isfinite(p.grad).all())
        group='svd' if name.startswith('backbone.') else name.split('.')[0]
        groups.setdefault(group,[]).append(p.grad.float().square().sum())
    if not torch.stack(finite).all():raise FloatingPointError('nonfinite gradient')
    keys=list(groups)
    values=torch.stack([torch.stack(groups[k]).sum().sqrt() for k in keys]).detach().cpu().tolist()
    return dict(zip(keys,values))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True);ap.add_argument('--config',required=True)
    ap.add_argument('--generated',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--mode',choices=('functional','full'),required=True)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    root=Path(args.root);generated=Path(args.generated);out=Path(args.output)
    cfg=json.loads(Path(args.config).read_text())
    if cfg.get('implementation_revision')!='v1.1-top4':raise ValueError('current Top4 contract required')
    if cfg!=json.loads((Path(__file__).parent/'config.json').read_text()):
        raise ValueError('configuration differs from this fixed implementation contract')
    if json.loads((generated/'input_contract.json').read_text())['status']!='PASS':raise ValueError('input not prepared')
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    if args.mode=='full' and world!=4:raise ValueError('formal contract requires four ranks')
    if not torch.cuda.is_available():raise RuntimeError('real Qwen requires CUDA')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(1)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(seconds=7200))
    if out.exists() and (out/'checkpoint.pt').exists() and not args.resume:raise ValueError('existing checkpoint needs explicit resume')
    out.mkdir(parents=True,exist_ok=True)
    store=Store(generated/'documents.sqlite')
    functional=json.loads((generated/'functional_ids.json').read_text())
    if args.mode=='functional':
        order=functional['train'];schedule=[order[i:i+4] for i in range(0,len(order),4)];val=functional['val']
    else:
        schedule=json.loads((generated/'schedule.json').read_text());val=[row[0] for row in store.metadata('val')]
    horizon=len(schedule)
    if args.mode=='functional' and (horizon!=3 or [len(x) for x in schedule]!=[4,4,3]):raise ValueError('functional schedule')
    if args.mode=='full' and (horizon!=1433 or len(schedule[-1])!=26):raise ValueError('full schedule')
    model,base_identity=load_model(root/'tev4_storage/models/Qwen3-4B',generated,device)
    weights=class_weights(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,
                                betas=(.9,.999),eps=1e-8,weight_decay=0.)
    manifest={'experiment':'Q-Locate-v1.1-top4','mode':args.mode,'world_size':world,
              'horizon':horizon,'seed':0,'base':base_identity,'config':cfg,
              'store':str((generated/'documents.sqlite').resolve()),
              'store_bytes':(generated/'documents.sqlite').stat().st_size,
              'input_contract':json.loads((generated/'input_contract.json').read_text()),
              'data_order':str((generated/('functional_ids.json' if args.mode=='functional' else 'schedule.json')).resolve())}
    start=0;cursor=0
    if args.resume:
        restored=load_checkpoint(model,optimizer,out/'checkpoint.pt',manifest,horizon)
        start=restored['step'];cursor=restored['cursor'];del restored
    if rank==0:atomic_json(manifest,out/'manifest.json')
    if args.resume and (start in (500,1000) or start==horizon) and not (out/f'eval_step{start}/result.json').exists():
        evaluate(model,store,val,out,start)
    initial={};restore_check=None;gradient_history=[]
    if args.mode=='functional' and start==0:
        saved_rng=rng_state(device)
        doc=store.get(functional['train'][0]);mp=doc.tensors(device)
        with torch.no_grad():
            from engine import compact_forward
            compact,_=compact_forward(model,doc,device)
            result=model.from_compact(*compact,mp)
            p0=model._gather(compact[0],mp,0,len(doc.text))
            initial={'doc_max_abs':float((result['doc_logits']-result['old_doc_logits']).abs().max()),
                     'source_max_abs':float((result['p']-p0.clamp(1e-6,1-1e-6)).abs().max()),
                     'clamped_characters':int(((p0<1e-6)|(p0>1-1e-6)).sum())}
            assert initial['doc_max_abs']==0 and initial['source_max_abs']<1e-6
        del result,compact,mp
        restore_rng(saved_rng,device)
    for step in range(start+1,horizon+1):
        begin=time.time();model.train();optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=lr_for(step,horizon)
        batch=schedule[step-1];documents=[store.get(id_) for id_ in batch[rank::world]]
        metrics,details=train_step(model,documents,weights,len(batch),check_replay=args.mode=='functional')
        gradients=gradient_report(model)
        ranks_equal=None
        if args.mode=='functional' and world>1:
            flattened=torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])
            reference=flattened.clone();dist.broadcast(reference,src=0)
            ranks_equal=torch.equal(flattened,reference)
            if not ranks_equal:raise RuntimeError('synchronized gradients differ across ranks')
            del flattened,reference
        gradient_history.append(gradients)
        norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
        optimizer.step();cursor+=len(batch)
        if not torch.equal(model.byte_embedding.weight[256],torch.zeros_like(model.byte_embedding.weight[256])):
            raise RuntimeError('PAD row changed')
        rank_record={'step':step,'rank':rank,'local_documents':len(documents),'global_documents':len(batch),
                     'lr':optimizer.param_groups[0]['lr'],'loss':metrics,'gradient_norm_before_clip':norm,
                     'gradient_groups':gradients,'document_work':details,'elapsed_seconds':time.time()-begin,
                     'synchronized_gradients_equal_rank0':ranks_equal,
                     'observed_at':datetime.now(timezone.utc).isoformat()}
        with (out/f'train_rank{rank}.jsonl').open('a') as file:file.write(json.dumps(rank_record)+'\n')
        if rank==0:print('STEP',step,'/',horizon,json.dumps({k:v for k,v in rank_record.items() if k not in ('document_work','gradient_groups')}),flush=True)
        if step%50==0 or step==horizon or (args.mode=='functional' and step==2):
            save_checkpoint(model,optimizer,step,cursor,horizon,manifest,out/'checkpoint.pt')
        if args.mode=='functional' and step==2:
            # Capture reference before controlled perturbation; all states restored.
            expected_rng=rng_state(device)
            with torch.no_grad():model.source[-1].bias.add_(1.)
            for state in optimizer.state.values():
                if 'exp_avg' in state:state['exp_avg'].add_(.1);break
            random.random();np.random.random();torch.rand(1);torch.rand(1,device=device)
            saved=load_checkpoint(model,optimizer,out/'checkpoint.pt',manifest,horizon)
            model_ok=all(torch.equal(p.detach().cpu(),saved['weights'][n]) for n,p in model.named_parameters() if p.requires_grad)
            opt_ok=equal_nested(optimizer.state_dict(),saved['optimizer'])
            rng_ok=equal_nested(rng_state(device),expected_rng)
            scheduler_ok=(saved['rank_states'][rank]['scheduler']=={'horizon':horizon,'last_step':step,'last_lr':lr_for(step,horizon)})
            assert model_ok and opt_ok and rng_ok and scheduler_ok and saved['cursor']==cursor and saved['step']==2
            restore_check={'mode':'same_process_save_perturb_load_continue','weights':model_ok,'optimizer':opt_ok,
                            'rank_rng':rng_ok,'scheduler':scheduler_ok,'cursor':cursor,'step':2,'world_size':world}
            atomic_json(restore_check,out/f'restore_rank{rank}.json');del saved
        if (args.mode=='full' and step in (500,1000)) or step==horizon:
            evaluate(model,store,val,out,step)
    profile=None
    if args.mode=='functional':
        # Real document window, separate no-update resource measurement.
        doc=store.get(functional['long_id']);window=doc.windows[0]
        saved_rng=rng_state(device);optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        from engine import window_forward
        t0=time.time()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            p,v,last=window_forward(model,doc,*window,device)
            # For full head input, use the real canonical document compact graph;
            # resource report labels backbone window + its actual covered chars.
            left=min(int(x[0]) for x in doc.offsets[:4096] if x[1]>x[0]);right=max(int(x[1]) for x in doc.offsets[:4096])
            subtext=doc.text[left:right]
            spans=[[max(a,left)-left,min(b,right)-left] for a,b in doc.spans if max(a,left)<min(b,right)]
            from data import byte_ids
            sub=Document(doc.doc_id+'-profile',doc.label,subtext,spans,doc.ids[:4096],
                         np.clip(doc.offsets[:4096]-left,0,len(subtext)),[[0,4096]],np.ones(4096,dtype=np.int16),byte_ids(subtext))
            mp=sub.tensors(device)
            result=model.from_compact(p,v,last,mp)
            # All head paths differentiated; resource only, not synthetic training labels.
            scalar=result['alpha'].square().mean()+result['b'].square().mean()+result['doc_logits'].square().mean()
            scalar.backward()
        if not torch.isfinite(scalar):raise FloatingPointError('profile loss')
        profile_gradients=gradient_report(model)
        torch.cuda.synchronize(device)
        operators=sorted({e.key for e in prof.key_averages() if any(k in e.key.lower() for k in ('attention','flash','efficient'))})
        profile={'tokens':4096,'chars':len(subtext),'world_size':world,'seconds':time.time()-t0,
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
                 'attention_operators':operators,'scope':'one real 4096-token Qwen+SVD+all-heads forward/backward; no optimizer step',
                 'gradient_groups':profile_gradients,'loss_finite':True,
                 'head_dtypes':{k:str(result[k].dtype) for k in ('alpha','p','b','a','doc_logits')},
                 'loss_dtype':str(scalar.dtype),
                 'dtype':next(m.last_dtypes for m in model.modules() if isinstance(m,FP32TailLinear))}
        if not operators:raise RuntimeError('attention backend not observed')
        del result,scalar,p,v,last,mp
        optimizer.zero_grad(set_to_none=True);restore_rng(saved_rng,device)
        atomic_json(profile,out/f'profile_rank{rank}.json')
        if start==0:
            for key in gradient_history[0]:
                if not any(row[key]>0 for row in gradient_history):raise RuntimeError('untrained parameter group '+key)
    if world>1:dist.barrier()
    if rank==0:
        final=json.loads((out/f'eval_step{horizon}/result.json').read_text())
        atomic_json({'status':'FUNCTIONAL_SINGLE_GPU_PASS' if args.mode=='functional' and world==1 else 'FUNCTIONAL_4GPU_PASS' if args.mode=='functional' else 'COMPLETE',
                     'endpoint_step':horizon,'cursor':cursor,'world_size':world,'initial_identity':initial,
                     'metrics':final['metrics'],'restore':restore_check,'full_training_started':args.mode=='full',
                     'four_rank_verified':world==4,'profile':profile},out/'completed.json')
    if world>1:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
