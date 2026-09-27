from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from train_retraining_v3 import load_model,evaluate,dump
from calibrate_retraining_v3 import calibrate_bn


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--base-name',default='candidate');args=p.parse_args()
    run=Path(args.run);package=run/args.base_name;out=run/(args.base_name+'_narrow_bn_probe')
    if out.exists():raise FileExistsError(out)
    out.mkdir();torch.set_num_threads(4)
    rows=json.loads((run/'dataset/samples.json').read_text(encoding='utf-8'))
    low=[r for r in rows if r['split']=='train' and r['native_size']==2048]
    model,mf=load_model(package,torch.device('cuda'))
    calibrate_bn(model,low,mf,torch.device('cuda'))
    values=evaluate(model,low,mf,torch.device('cuda'),[.2,.3,.4,.5,.6,.7,.8])
    t,m=max(values.items(),key=lambda pair:pair[1]['dice'])
    dump(out/'diagnostic.json',{'scope':'training narrow-camera BN calibration and resubstitution only', 'threshold':float(t),'metrics':m,'all_thresholds':values})
    state={k:v.detach().cpu() for k,v in model.state_dict().items() if k.endswith(('running_mean','running_var','num_batches_tracked'))}
    torch.save(state,out/'bn_stats.pth')
    torch.save(model.state_dict(),out/'weights.pth')
    mf['inference']['threshold']=float(t)
    dump(out/'model_manifest.json',mf)
    evaluate(model,low,mf,torch.device('cuda'),[float(t)],out/'previews')
    print(json.dumps({'threshold':float(t),'metrics':m}),flush=True)


if __name__=='__main__':main()
