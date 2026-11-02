"""Detach-safe launch with one process lock and a persistent actual return code."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from runtime import atomic_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--mode',choices=('functional','full'),required=True)
    ap.add_argument('--world',type=int,choices=(1,4),required=True)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    code=Path(__file__).resolve().parent;out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'supervisor.json').exists() and not args.resume:
            raise RuntimeError('run already attempted; inspect it, choose explicit resume or a new run directory')
        if args.mode=='full':
            readiness=json.loads((code/'audit/training_ready_report.json').read_text())
            if readiness['status']!='TRAINING_READY':raise RuntimeError('four-GPU functional acceptance pending')
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if gpu:raise RuntimeError('GPU compute processes already present: '+gpu)
        argv=[sys.executable]
        if args.world>1:argv+=['-m','torch.distributed.run','--standalone','--nproc_per_node',str(args.world)]
        argv+=['-u',str(code/'train.py')] if args.world==1 else [str(code/'train.py')]
        argv+=['--root',args.root,'--config',str(code/'config.json'),'--generated',str(code/'generated'),
               '--output',str(out),'--mode',args.mode]
        if args.resume:argv+=['--resume']
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        record={'supervisor_pid':os.getpid(),'argv':argv,'started_at':time.time(),'returncode':None,
                'mode':args.mode,'world_size':args.world,'resume':args.resume,
                'gpu_inventory':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader'],text=True)}
        log_path=out/('resume_'+str(int(record['started_at']))+'.log' if args.resume else 'train.log')
        record['log_path']=str(log_path)
        atomic_json(record,out/'supervisor.json')
        with log_path.open('w') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,cwd=code,env=env)
            record['child_pid']=child.pid;atomic_json(record,out/'supervisor.json')
            result=child.wait()
        record.update(returncode=result,finished_at=time.time());atomic_json(record,out/'supervisor.json')
        raise SystemExit(result)


if __name__=='__main__':main()
