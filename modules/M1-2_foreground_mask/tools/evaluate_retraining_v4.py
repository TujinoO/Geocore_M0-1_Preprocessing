"""Calibrate on development sets only, then report labeled evidence by scope."""
import argparse,copy,json
from pathlib import Path
import torch
from train_retraining_v3 import evaluate,load_model,dump
from train_retraining_v4 import choose
from prepare_retraining_v3 import digest

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['calibrate','evaluate'])
    p.add_argument('--dataset',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((Path(a.dataset)/'samples.json').read_text(encoding='utf-8'))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;device=torch.device('cuda')
    model,mf=load_model(a.model,device);model.eval()
    if a.mode=='calibrate':
        trials=[]
        for size in (1536,2048,2560):
            cfg=copy.deepcopy(mf);cfg['inference']['source_tile_size']=size
            thresholds=[round(.25+i*.05,2) for i in range(11)]
            v=evaluate(model,[r for r in rows if r['split']=='val'],cfg,device,thresholds)
            d=evaluate(model,[r for r in rows if r['split']=='adaptation_dev'],cfg,device,thresholds)
            ok,score,t,vm,dm=choose(v,d)
            trial={'source_tile_size':size,'threshold':float(t),'guard_passed':ok,'score':score,'val':vm,'adaptation_dev':dm}
            trials.append(trial);dump(out/f'scale_{size}.json',{'val':v,'adaptation_dev':d});print(json.dumps(trial),flush=True)
        best=max(trials,key=lambda r:(r['guard_passed'],r['score']))
        dump(out/'selection.json',{'selected':best,'trials':trials,'independent_test_available':False,
            'scope':'ZK6711 held-out-from-gradient validation and known-hole ZKZ4-5 adaptation development; both used for model selection.'})
        import shutil
        shutil.copy2(Path(a.model)/'weights.pth',out/'weights.pth')
        mf['inference'].update(source_tile_size=best['source_tile_size'],threshold=best['threshold'])
        mf['calibration']={'selection_file':'selection.json','independent_test_available':False}
        dump(out/'model_manifest.json',mf)
    else:
        subsets={'validation_ZK6711':[r for r in rows if r['split']=='val'],
                 'adaptation_dev_ZKZ4-5':[r for r in rows if r['split']=='adaptation_dev'],
                 'new_hard_training':[r for r in rows if r['is_new']],
                 'old_low_training':[r for r in rows if not r['is_new'] and r['split']=='train' and r['native_size']==2048],
                 'old_wide_training':[r for r in rows if not r['is_new'] and r['split']=='train' and r['native_size']!=2048]}
        reports={}
        for name,subset in subsets.items():
            result=evaluate(model,subset,mf,device,[mf['inference']['threshold']],out/name)
            reports[name]={'count':len(subset),'metrics':next(iter(result.values()))}
            print(json.dumps({name:reports[name]}),flush=True)
        dump(out/'summary.json',{'model':str(a.model),'model_weights_sha256':digest(Path(a.model)/'weights.pth'),
            'threshold':mf['inference']['threshold'],'source_tile_size':mf['inference']['source_tile_size'],
            'independent_test_available':False,'subsets':reports})

if __name__=='__main__':main()
