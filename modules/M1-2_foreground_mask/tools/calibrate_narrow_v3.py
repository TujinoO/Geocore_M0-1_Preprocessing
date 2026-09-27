from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import torch
from train_retraining_v3 import evaluate,load_model,dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    run=Path(args.run);package=run/'candidate';torch.set_num_threads(4)
    if (package/'test_result.json').exists():raise RuntimeError('Final test already opened')
    rows=json.loads((run/'dataset/samples.json').read_text(encoding='utf-8'))
    rows=[r for r in rows if r['split']=='train' and r['native_size']==2048]
    model,mf=load_model(package,torch.device('cuda'));trials=[]
    for size in (512,1024,1536,2048):
        config=copy.deepcopy(mf);config['inference']['source_tile_size']=size
        result=evaluate(model,rows,config,torch.device('cuda'),[.3,.4,.5,.6,.7])
        t,m=max(result.items(),key=lambda x:x[1]['dice'])
        record={'size':size,'threshold':float(t),'metrics':m};trials.append(record)
        print(json.dumps(record),flush=True)
    dump(package/'narrow_domain_scale_diagnostic.json',{'scope':'training-only diagnostic; no independent low-resolution labels', 'trials':trials})


if __name__=='__main__':main()
