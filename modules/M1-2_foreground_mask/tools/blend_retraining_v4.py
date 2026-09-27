"""Exploratory weight interpolation, selected only on declared development data."""
import argparse,copy,json
from pathlib import Path
import torch
from train_retraining_v3 import load_model,evaluate,dump
from train_retraining_v4 import choose

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--base',required=True);a=p.parse_args();run=Path(a.run)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;device=torch.device('cuda')
    rows=json.loads((run/'dataset/samples.json').read_text(encoding='utf-8'))
    model,mf=load_model(run/'frozen_bn',device);old,_=load_model(a.base,'cpu')
    newer={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};original=old.state_dict();del old
    for alpha in (.5,.75,.9):
        out=run/f'blend_{int(alpha*100)}';out.mkdir(exist_ok=False)
        state={k:(original[k].lerp(v,alpha) if v.is_floating_point() else v) for k,v in newer.items()}
        model.load_state_dict(state);cfg=copy.deepcopy(mf);thresholds=[.35,.45,.55,.65]
        val=evaluate(model,[r for r in rows if r['split']=='val'],cfg,device,thresholds)
        dev=evaluate(model,[r for r in rows if r['split']=='adaptation_dev'],cfg,device,thresholds)
        ok,score,t,v,d=choose(val,dev)
        cfg['inference']['threshold']=float(t)
        cfg['training']={'weight_interpolation':{'V3_fraction':1-alpha,'frozen_bn_V4_fraction':alpha},
             'selection_score':score,'validation_dice':v['dice'],'adaptation_development_dice':d['dice'],'independent_test_available':False}
        torch.save(state,out/'weights.pth');dump(out/'model_manifest.json',cfg)
        dump(out/'status.json',{'status':'training_complete','last_epoch':0,'best_rank':[int(ok),score],
             'scope':'Weight interpolation of trained checkpoints; no extra gradient epochs, development-selected'})
        dump(out/'evaluation.json',{'val':val,'adaptation_dev':dev})
        print(json.dumps({'alpha':alpha,'score':score,'guard':ok,'threshold':t,'val':v,'adaptation_dev':d}),flush=True)

if __name__=='__main__':main()
