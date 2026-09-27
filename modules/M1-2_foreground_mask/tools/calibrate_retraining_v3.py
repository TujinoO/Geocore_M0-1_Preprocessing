"""Select deployment scale/threshold on validation only; optional train-only BN statistics."""
from __future__ import annotations
import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_retraining_v3 import dump, evaluate, load_model
from geocore_mask.inference.scaled import positions


@torch.no_grad()
def calibrate_bn(model, rows, manifest, device):
    """Recompute BN buffers from training images at inference scale, no gradients."""
    norms=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    momentum=[m.momentum for m in norms]
    for m in norms:m.reset_running_stats();m.momentum=None
    model.train();batch=[]
    mean=np.asarray(manifest['input']['mean'],np.float32);std=np.asarray(manifest['input']['std'],np.float32)
    # Keep deterministic randomized source ordering; no validation/test pixels.
    order=np.random.default_rng(260926).permutation(len(rows))
    for index in order:
        row=rows[int(index)]
        with Image.open(row['image']) as im:arr=np.asarray(im.convert('RGB'))
        h,w=arr.shape[:2];s=min(h,w,2048)
        for y in positions(h,s,int(s*.75)):
            for x in positions(w,s,int(s*.75)):
                patch=np.asarray(Image.fromarray(arr[y:y+s,x:x+s]).resize((512,512),Image.Resampling.BILINEAR),np.float32)
                batch.append(torch.from_numpy(((patch-mean)/std).transpose(2,0,1).copy()))
                if len(batch)==2:
                    with torch.autocast('cuda',dtype=torch.float16):model(torch.stack(batch).to(device))
                    batch=[]
    if batch:
        with torch.autocast('cuda',dtype=torch.float16):model(torch.stack(batch).to(device))
    for m,value in zip(norms,momentum):m.momentum=value
    model.eval()


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--package',required=True)
    args=p.parse_args();dataset=Path(args.dataset);package=Path(args.package)
    torch.set_num_threads(4)
    if (package/'test_result.json').exists():raise RuntimeError('Test already opened: cannot tune further')
    if (package/'validation_calibration.json').exists():raise FileExistsError('Calibration already recorded')
    rows=json.loads((dataset/'samples.json').read_text(encoding='utf-8'))
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    device=torch.device('cuda');model,mf=load_model(package,device)
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    thresholds=[round(v,2) for v in np.arange(.2,.801,.05)]
    trials=[];best=None;best_state=None;best_manifest=None
    # Predeclared small grid, never changed in response to final test results.
    for kind in ('trained_bn','train_recalibrated_bn'):
        model.load_state_dict(original)
        if kind=='train_recalibrated_bn':calibrate_bn(model,train,mf,device)
        for size in (1536,2048,2560):
            current=copy.deepcopy(mf);current['inference']['source_tile_size']=size
            started=time.time();scores=evaluate(model,val,current,device,thresholds)
            threshold,score=max(scores.items(),key=lambda x:x[1]['dice'])
            record={'bn':kind,'source_tile_size':size,'threshold':float(threshold),'metrics':score,'seconds':time.time()-started}
            trials.append(record);print(json.dumps(record),flush=True)
            if best is None or score['dice']>best['metrics']['dice']+1e-4:
                best=record;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                current['inference']['threshold']=float(threshold);best_manifest=current
    shutil.copy2(package/'weights.pth',package/'weights_before_calibration.pth')
    shutil.copy2(package/'model_manifest.json',package/'manifest_before_calibration.json')
    torch.save(best_state,package/'weights.pth')
    best_manifest['validation_calibration']={'selected':best,'used_test':False,'bn_recalibration_sources':'train_only',
        'selection_rule':'maximum validation pixel Dice; tie tolerance 0.0001'}
    dump(package/'model_manifest.json',best_manifest)
    dump(package/'validation_calibration.json',{'trials':trials,'selected':best,'test_used':False})
    print('[calibration complete; independent test remains locked]',flush=True)


if __name__=='__main__':main()
