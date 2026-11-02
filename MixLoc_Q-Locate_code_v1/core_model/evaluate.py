"""Single inference pass: distributed metrics and full float32 array exports."""
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist
from engine import compact_forward
from decode import decode,mask_spans
from runtime import atomic_json


def prf(tp,fp,fn):
    return {'tp':int(tp),'fp':int(fp),'fn':int(fn),'precision':tp/max(1,tp+fp),
            'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn)}


class Metrics:
    def __init__(self):
        self.matrix=np.zeros((3,3),dtype=np.int64)
        self.counts={key:np.zeros(3,dtype=np.int64) for key in ('raw','decoded','gated','e2e')}
        self.span={key:np.zeros(6,dtype=np.float64) for key in ('raw','decoded')}
        self.partial_means={key:[] for key in ('raw','decoded','gated')}
        self.anchor={key:np.zeros(4,dtype=np.int64) for key in ('raw','decoded')}
        self.groups={}
        self.boundary={key:[] for key in ('start_b_mean_H','end_b_mean_H','start_b_mean_P','end_b_mean_P')}
        self.gt_boundary=np.zeros(4,dtype=np.int64)
        self.span_length={key:np.zeros(3,dtype=np.int64) for key in ('1-32','33-128','129-512','513+')}

    def update(self,doc,logits,alpha,b,a):
        pred=int(np.argmax(logits));n=len(doc.text);label=doc.label
        self.matrix[label,pred]+=1
        raw=np.asarray(alpha)>=0
        decoded,_=decode(alpha,a)
        gated=np.zeros(n,dtype=bool) if pred==0 else np.ones(n,dtype=bool) if pred==2 else decoded
        gold=np.zeros(n,dtype=bool)
        for s,e in doc.spans:gold[s:e]=True
        length='1-512' if n<=512 else '513-2048' if n<=2048 else '2049-8192' if n<=8192 else '8193+'
        group_keys=['length:'+length,'language:'+doc.language]
        for key in group_keys:
            if key not in self.groups:self.groups[key]={'matrix':np.zeros((3,3),dtype=np.int64),'partial_raw':np.zeros(3,dtype=np.int64),'human_raw':np.zeros(2,dtype=np.int64)}
            self.groups[key]['matrix'][label,pred]+=1
        masks={'raw':raw,'decoded':decoded,'gated':gated}
        for key,mask in masks.items():
            counts=np.array([(gold&mask).sum(),(~gold&mask).sum(),(gold&~mask).sum()],dtype=np.int64)
            if label==1:
                self.counts[key]+=counts;self.partial_means[key].append(prf(*counts)['f1'])
                if key=='raw':
                    for g in group_keys:self.groups[g]['partial_raw']+=counts
            if key=='gated':self.counts['e2e']+=counts
            if key in ('raw','decoded'):
                if label==0:self.anchor[key][:2]+=np.array([mask.sum(),n])
                if label==2:self.anchor[key][2:]+=np.array([(~mask).sum(),n])
            if key=='raw' and label==0:
                for g in group_keys:self.groups[g]['human_raw']+=np.array([mask.sum(),n])
        for key in ('raw','decoded'):
            if label!=1:continue
            pred_spans=mask_spans(masks[key]);true_spans=doc.spans
            values=self.span[key]
            values[0]+=len(pred_spans);values[1]+=len(true_spans)
            values[2]+=len(set(pred_spans)&set(map(tuple,true_spans)))
            used=set()
            for ps,pe in pred_spans:
                best=.5;match=None
                for j,(gs,ge) in enumerate(true_spans):
                    if j in used:continue
                    inter=max(0,min(pe,ge)-max(ps,gs));union=pe-ps+ge-gs-inter
                    iou=inter/union if union else 0.
                    if iou>=best:best,match=iou,j
                if match is not None:
                    used.add(match);values[3]+=1
                    gs,ge=true_spans[match];values[4]+=abs(ps-gs)+abs(pe-ge);values[5]+=2
            for j,(gs,ge) in enumerate(true_spans):
                size=ge-gs;bucket='1-32' if size<=32 else '33-128' if size<=128 else '129-512' if size<=512 else '513+'
                if key=='raw':self.span_length[bucket][0]+=1;self.span_length[bucket][1]+=int(j in used)
                else:self.span_length[bucket][2]+=int(j in used)
        if label in (0,1):
            suffix='H' if label==0 else 'P'
            self.boundary['start_b_mean_'+suffix].append(float(b[:-1,0].mean()))
            self.boundary['end_b_mean_'+suffix].append(float(b[1:,1].mean()))
        if label==1:
            probability=1/(1+np.exp(-np.clip(alpha,-80,80)))
            delta=np.r_[probability,0]-np.r_[0,probability]
            for s,e in doc.spans:
                self.gt_boundary+=np.array([int(delta[s]>0),int(delta[e]<0),int(a[s,0]>0)+int(a[e,1]>0),1])
        return {'doc_id':doc.doc_id,'label':label,'prediction':pred,'n_chars':n,
                'raw_spans':mask_spans(raw),'decoded_spans':mask_spans(decoded)}

    def result(self):
        matrix=self.matrix;per=[]
        for c in range(3):per.append(prf(matrix[c,c],matrix[:,c].sum()-matrix[c,c],matrix[c,:].sum()-matrix[c,c]))
        spans={}
        for key,(pc,gc,exact,matched,error,ends) in self.span.items():
            spans[key]={'predicted':int(pc),'gold':int(gc),'exact':prf(exact,pc-exact,gc-exact),
                        'iou50':prf(matched,pc-matched,gc-matched),
                        'boundary_mae':error/ends if ends else None,'matched_spans':int(matched)}
        return {'n_documents':int(matrix.sum()),'accuracy':float(np.trace(matrix)/max(1,matrix.sum())),
                'macro_f1':sum(x['f1'] for x in per)/3,'balanced_accuracy':sum(x['recall'] for x in per)/3,
                'per_class':per,'confusion_matrix':matrix.tolist(),'H_to_P':int(matrix[0,1]),'P_to_H':int(matrix[1,0]),
                'character':{k:prf(*v) for k,v in self.counts.items()},'spans':spans,
                'partial_doc_mean_f1':{k:float(np.mean(v)) if v else None for k,v in self.partial_means.items()},
                'anchor_characters':{k:{'human_fp':int(v[0]),'human_total':int(v[1]),'human_fp_rate':v[0]/max(1,v[1]),
                                         'full_fn':int(v[2]),'full_total':int(v[3]),'full_fn_rate':v[2]/max(1,v[3])} for k,v in self.anchor.items()},
                'groups':{k:{'matrix':v['matrix'].tolist(),'partial_raw':prf(*v['partial_raw']),
                             'human_raw_fp':v['human_raw'].tolist()} for k,v in self.groups.items()},
                'raw_boundary_document_means':{k:float(np.mean(v)) if v else None for k,v in self.boundary.items()},
                'true_partial_boundary_diagnostics':{'source_start_positive':int(self.gt_boundary[0]),'source_end_negative':int(self.gt_boundary[1]),
                      'relative_boundary_positive':int(self.gt_boundary[2]),'gold_spans':int(self.gt_boundary[3])},
                'span_length_recall_counts':{k:v.tolist() for k,v in self.span_length.items()}}


@torch.no_grad()
def evaluate(model,store,ids,out,step):
    world=dist.get_world_size() if dist.is_initialized() else 1
    rank=dist.get_rank() if dist.is_initialized() else 0
    target=Path(out)/f'eval_step{step}';target.mkdir(parents=True,exist_ok=True)
    device=model.doc_head.weight.device;was_training=model.training;model.eval()
    index=target/f'rank{rank}.jsonl';binary=target/f'rank{rank}.f32'
    local=Metrics();offset=0
    try:
        with index.open('w',encoding='utf-8') as jf,binary.open('wb') as bf:
            for index_in_split,doc_id in enumerate(ids[rank::world]):
                doc=store.get(doc_id);mapping=doc.tensors(device)
                compact,_=compact_forward(model,doc,device)
                output=model.from_compact(*compact,mapping)
                tensors={k:output[k].float().cpu().numpy() for k in ('doc_logits','alpha','b','a')}
                if not all(np.isfinite(x).all() for x in tensors.values()):raise FloatingPointError('evaluation outputs')
                record=local.update(doc,tensors['doc_logits'],tensors['alpha'],tensors['b'],tensors['a'])
                slices={}
                for key,value in tensors.items():
                    data=np.asarray(value,dtype='<f4');slices[key]={'offset':offset,'shape':list(data.shape)}
                    bf.write(data.tobytes());offset+=data.size
                record['arrays']=slices;record['array_file']=binary.name
                jf.write(json.dumps(record,ensure_ascii=False)+'\n')
                if (index_in_split+1)%500==0:print('EVAL',step,'rank',rank,'docs',index_in_split+1,flush=True)
                del mapping,compact,output,tensors
        atomic_json(local.result(),target/f'rank{rank}_metrics.json')
    finally:model.train(was_training)
    if world>1:dist.barrier()
    final=None
    if rank==0:
        # Re-read exported arrays: catches incomplete writes and verifies all IDs.
        merged=Metrics();seen=set();independent=np.zeros((3,3),dtype=np.int64);char=np.zeros(3,dtype=np.int64)
        for other in range(world):
            data=np.memmap(target/f'rank{other}.f32',dtype='<f4',mode='r')
            with (target/f'rank{other}.jsonl').open(encoding='utf-8') as file:
                for line in file:
                    record=json.loads(line);doc_id=record['doc_id']
                    if doc_id in seen:raise ValueError('duplicate val ID')
                    seen.add(doc_id);doc=store.get(doc_id);values={}
                    for key,sl in record['arrays'].items():
                        size=int(np.prod(sl['shape']));values[key]=np.asarray(data[sl['offset']:sl['offset']+size]).reshape(sl['shape'])
                    calculated=merged.update(doc,values['doc_logits'],values['alpha'],values['b'],values['a'])
                    if json.loads(json.dumps(calculated))!={k:record[k] for k in calculated}:raise ValueError('exported predictions differ')
                    pred=int(values['doc_logits'].argmax());independent[doc.label,pred]+=1
                    if doc.label==1:
                        # Independent interval overlap counts, no exported TP/FP/FN reuse.
                        mask=values['alpha']>=0;positive=int(mask.sum())
                        tp=sum(int(mask[s:e].sum()) for s,e in doc.spans)
                        gold=sum(e-s for s,e in doc.spans)
                        char+=np.array([tp,positive-tp,gold-tp])
            del data
        if seen!=set(ids) or len(seen)!=len(ids):raise ValueError('evaluation ID coverage')
        final=merged.result()
        if final['confusion_matrix']!=independent.tolist():raise ValueError('independent matrix')
        if [final['character']['raw'][k] for k in ('tp','fp','fn')]!=char.tolist():raise ValueError('independent raw character counts')
        if world==1 and final!=local.result():raise ValueError('inference vs export metric mismatch')
        atomic_json({'step':step,'metrics':final,'id_coverage':'PASS','independent_matrix_and_raw_character':'PASS',
                     'array_dtype':'float32','invalid_relative_gap':'zero_placeholder'},target/'result.json')
    if world>1:dist.barrier()
    return final
