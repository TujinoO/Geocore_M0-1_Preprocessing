"""Reproducible grouped U-Net retraining with shared deployment inference."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from scipy import ndimage
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geocore_mask.models.registry import build_model
from geocore_mask.inference.scaled import predict_array


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2),encoding='utf-8')
    os.replace(tmp,path)


def counts(pred, target):
    return [int((pred & target).sum()),int((pred & ~target).sum()),int((~pred & target).sum()),int((~pred & ~target).sum())]


def metrics(values):
    tp,fp,fn,tn=map(int,values)
    return {'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,
            'precision':tp/(tp+fp) if tp+fp else (1. if tp+fn==0 else 0.),
            'recall':tp/(tp+fn) if tp+fn else 1.,
            'false_positive_rate':fp/(fp+tn) if fp+tn else 0.,
            'tp':tp,'fp':fp,'fn':fn,'tn':tn}


class CoreDataset(Dataset):
    def __init__(self, rows, dataset, mean, std, steps, batch, lowres_probability=0.):
        self.rows=rows; self.dataset=Path(dataset); self.length=steps*batch
        self.mean=np.asarray(mean,np.float32); self.std=np.asarray(std,np.float32)
        self.lowres_probability=lowres_probability
        self.lowres_rows=[r for r in rows if r['native_size']==2048]
        self.groups=defaultdict(list)
        self.images={}; self.masks={}; self.points={}
        for r in rows:
            self.groups[r['split_group']].append(r)
            im=np.load(self.dataset/'cache'/f"{r['id']}_rgb.npy")
            mask=np.load(self.dataset/'cache'/f"{r['id']}_mask.npy").astype(bool)
            self.images[r['id']]=im; self.masks[r['id']]=mask
            low=mask[::8,::8]
            boundary=ndimage.binary_dilation(low)^ndimage.binary_erosion(low)
            self.points[r['id']]=[np.argwhere(low),np.argwhere(boundary),np.argwhere(~low)]
        self.group_names=list(self.groups)
        self.group_weights=[math.sqrt(len(self.groups[g])) for g in self.group_names]

    def __len__(self): return self.length

    def __getitem__(self,index):
        group=random.choices(self.group_names,weights=self.group_weights)[0]
        row=random.choice(self.lowres_rows) if self.lowres_rows and self.lowres_probability>0 and random.random()<self.lowres_probability else random.choice(self.groups[group])
        key=row['id']
        image,mask=self.images[key],self.masks[key]
        native=random.choice([s for s in (1024,1536,2048,2560,3072) if s<=row['native_size']])
        size=max(128,round(native/row['native_size']*1024))
        mode=random.choices(range(4),[.35,.30,.20,.15])[0]
        pts=self.points[key][mode] if mode<3 else []
        if len(pts):
            cy,cx=pts[random.randrange(len(pts))]*8
        else:
            cy,cx=random.randrange(1024),random.randrange(1024)
        y=int(np.clip(cy-size//2+random.randint(-size//4,size//4),0,1024-size))
        x=int(np.clip(cx-size//2+random.randint(-size//4,size//4),0,1024-size))
        im=Image.fromarray(image[y:y+size,x:x+size]).resize((512,512),Image.Resampling.BILINEAR)
        ma=Image.fromarray(mask[y:y+size,x:x+size]).resize((512,512),Image.Resampling.NEAREST)
        for transform in (Image.Transpose.FLIP_LEFT_RIGHT,Image.Transpose.FLIP_TOP_BOTTOM):
            if random.random()<.5: im=im.transpose(transform);ma=ma.transpose(transform)
        angle=random.randrange(4)*90
        if angle: im=im.rotate(angle);ma=ma.rotate(angle)
        im=ImageEnhance.Brightness(im).enhance(random.uniform(.65,1.45))
        im=ImageEnhance.Contrast(im).enhance(random.uniform(.75,1.3))
        im=ImageEnhance.Color(im).enhance(random.uniform(.7,1.3))
        if random.random()<.12: im=im.filter(ImageFilter.GaussianBlur(random.uniform(.2,.9)))
        arr=np.asarray(im,np.float32)
        if random.random()<.35:
            gamma=random.uniform(.75,1.3)
            arr=255*np.power(arr/255,gamma)
        if random.random()<.3:
            arr*=np.random.uniform(.9,1.1,(1,1,3)).astype(np.float32)
        if random.random()<.15:
            arr+=np.random.normal(0,random.uniform(.2,2),arr.shape).astype(np.float32)
        arr=np.clip(arr,0,255)
        target=np.asarray(ma,np.float32)
        return torch.from_numpy(((arr-self.mean)/self.std).transpose(2,0,1).copy()),torch.from_numpy(target.copy())


def loss_fn(logits, target):
    logits=logits.float()
    both=torch.stack((1-target,target),dim=1)
    bce=F.binary_cross_entropy_with_logits(logits,both)
    p=logits[:,1].sigmoid()
    dice=(2*(p*target).sum((1,2))+1)/(p.sum((1,2))+target.sum((1,2))+1)
    return bce+(1-dice).mean()


def make_manifest(mean,std):
    return {'model_name':'CoreMaskUNet','model_version':'v3.0.0-manual-20260926','framework':'pytorch',
            'task':'M1-2 foreground mask','num_classes':2,'weights':'weights.pth','output_activation':'sigmoid',
            'input':{'channels':3,'image_size':512,'mean':mean,'std':std,'color_order':'RGB','dtype':'uint8'},
            'inference':{'source_tile_size':2048,'model_input_size':512,'overlap':.25,'threshold':.5,'amp':True,'device':'auto'},
            'postprocess':{'enable':False},'review_required':True}


def load_model(package,device):
    mf=json.loads((Path(package)/'model_manifest.json').read_text(encoding='utf-8'))
    model=build_model('CoreMaskUNet',3,2)
    state=torch.load(Path(package)/mf['weights'],map_location='cpu',weights_only=False)
    state=state.get('state_dict',state)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    return model.to(device),mf


def evaluate(model, rows, manifest, device, thresholds, output=None):
    totals={t:np.zeros(4,np.int64) for t in thresholds}; records=[]
    old_head=model.outc.conv[1]; model.outc.conv[1]=nn.Sigmoid(); model.eval()
    if output: Path(output).mkdir(parents=True,exist_ok=True)
    try:
        for row in rows:
            with Image.open(row['image']) as im: image=np.asarray(im.convert('RGB'))
            with Image.open(row['mask']) as im: target=np.asarray(im)>127
            prob=predict_array(model,image,manifest,device)
            per={}
            for t in thresholds:
                c=counts(prob>=t,target);totals[t]+=c;per[str(t)]=metrics(c)
            records.append({'id':row['id'],'group':row['group'],'metrics':per})
            if output:
                pred=prob>=thresholds[0]
                thumb=Image.fromarray(image).resize((512,512),Image.Resampling.BILINEAR)
                p=np.asarray(Image.fromarray(pred).resize((512,512),Image.Resampling.NEAREST))
                t=np.asarray(Image.fromarray(target).resize((512,512),Image.Resampling.NEAREST))
                rgb=np.asarray(thumb).copy(); rgb[p]=(rgb[p]*.55+np.array([0,210,255])*.45).astype(np.uint8)
                error=np.asarray(thumb).copy();error[p & ~t]=[255,50,30];error[t & ~p]=[250,220,0]
                panel=Image.new('RGB',(1536,545),'white')
                panel.paste(thumb,(0,0));panel.paste(Image.fromarray(rgb),(512,0));panel.paste(Image.fromarray(error),(1024,0))
                ImageDraw.Draw(panel).text((5,517),f"{row['id']} dice={per[str(thresholds[0])]['dice']:.4f}  red=false positive; yellow=missed core",fill='black')
                panel.save(Path(output)/f"{row['id']}.jpg",quality=90)
                Image.fromarray(pred.astype(np.uint8)*255).save(Path(output)/f"{row['id']}_mask.png")
    finally:
        model.outc.conv[1]=old_head
    result={str(t):metrics(c) for t,c in totals.items()}
    if output:
        groups={}
        for group in sorted({r['group'] for r in records}):
            entries=[r['metrics'][str(thresholds[0])] for r in records if r['group']==group]
            groups[group]=metrics([sum(r[k] for r in entries) for k in ('tp','fp','fn','tn')])
        dump(Path(output)/'metrics.json',{'aggregate':result,'groups':groups,'images':records})
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--base',default=str(ROOT/'models/core_mask_unet_v1'))
    p.add_argument('--epochs',type=int,default=50);p.add_argument('--steps',type=int,default=48)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--patience',type=int,default=12)
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--evaluate-only',action='store_true');p.add_argument('--baseline',action='store_true')
    p.add_argument('--sanity-only',action='store_true',help='Training low-resolution domain check; not independent evaluation')
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--ema',action='store_true',help='Exponential moving average weights and BN buffers, decay 0.99')
    p.add_argument('--lowres-probability',type=float,default=0.,help='Additional narrow-camera training sampling probability')
    args=p.parse_args(); torch.set_num_threads(4)
    random.seed(260926);np.random.seed(260926);torch.manual_seed(260926)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for this training run')
    device=torch.device('cuda');torch.backends.cudnn.benchmark=True
    dataset=Path(args.dataset);out=Path(args.output)
    rows=json.loads((dataset/'samples.json').read_text(encoding='utf-8'))
    audit=json.loads((dataset/'audit.json').read_text(encoding='utf-8'))
    if audit['errors']: raise ValueError('Dataset audit has unresolved errors')
    splits={s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
    groups={s:{r['split_group'] for r in v} for s,v in splits.items()}
    if any(groups[a]&groups[b] for a,b in (('train','val'),('train','test'),('val','test'))): raise ValueError('Group leakage')
    model,base=load_model(args.base,device)
    if args.sanity_only:
        model,mf=load_model(out,device)
        subset=[r for r in splits['train'] if r['native_size']==2048]
        report=evaluate(model,subset,mf,device,[mf['inference']['threshold']],out/'training_domain_sanity')
        dump(out/'training_domain_sanity/scope.json',{'scope':'training-set diagnostic, not independent generalization','count':len(subset)})
        print(json.dumps(report),flush=True)
        return
    if args.evaluate_only:
        model,mf=load_model(out,device)
        test_dir=out/'test_evaluation'
        if test_dir.exists(): raise FileExistsError('Final test already evaluated; preserve evidence')
        result=evaluate(model,splits['test'],mf,device,[mf['inference']['threshold']],test_dir)
        dump(out/'test_result.json',result);print(json.dumps(result),flush=True)
        if args.baseline:
            old,old_mf=load_model(ROOT/'models/core_mask_unet_v2',device)
            old_mf['output_activation']='legacy_softmax'
            old_mf['inference'].update(source_tile_size=512,amp=False)
            b=evaluate(old,splits['test'],old_mf,device,[.5],out/'legacy_v2_test')
            dump(out/'comparison.json',{'new_model':result,'legacy_v2':b,'baseline_note':'legacy deployed sigmoid followed by softmax; native 512 source window, shared safe stripe merger; no postprocess for either model'})
            print(json.dumps({'baseline_test':b}),flush=True)
        final_status=json.loads((out/'status.json').read_text(encoding='utf-8'))
        final_status.update(status='training_and_test_complete',test_evaluated=True,baseline_evaluated=args.baseline)
        dump(out/'status.json',final_status)
        return
    if out.exists() and any(out.iterdir()) and not args.resume: raise FileExistsError('Use a fresh output or --resume')
    out.mkdir(parents=True,exist_ok=True)
    pixels=np.concatenate([np.load(dataset/'cache'/f"{r['id']}_rgb.npy")[::8,::8].reshape(-1,3) for r in splits['train']])
    mean=pixels.mean(0).tolist();std=np.maximum(pixels.std(0),5).tolist()
    mf=make_manifest(mean,std)
    config=vars(args).copy();config.update(seed=260926,mean=mean,std=std,source_crop_sizes=[1024,1536,2048,2560,3072],
        train_count=len(splits['train']),val_count=len(splits['val']),test_count=len(splits['test']),
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),
        dataset_manifest_sha256=hashlib.sha256((dataset/'samples.json').read_bytes()).hexdigest())
    if not args.resume: dump(out/'training_config.json',config)
    ds=CoreDataset(splits['train'],dataset,mean,std,args.steps,args.batch_size,args.lowres_probability)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=True)
    model.outc.conv[1]=nn.Identity()  # same weights; stable fused BCE-with-logits in AMP
    ema_model=copy.deepcopy(model).eval() if args.ema else None
    if ema_model is not None:
        for parameter in ema_model.parameters():parameter.requires_grad_(False)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='max',factor=.5,patience=4,min_lr=1e-6)
    scaler=torch.amp.GradScaler('cuda')
    history=[];best=-1.;stale=0;start=1;best_epoch=0
    if args.resume:
        ck=torch.load(out/'resume.pth',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']);optim.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
        if ema_model is not None:ema_model.load_state_dict(ck['ema_model'])
        history=ck['history'];best=ck['best'];stale=ck['stale'];best_epoch=ck['best_epoch'];start=ck['epoch']+1
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    thresholds=[round(t,2) for t in np.arange(.25,.751,.05)]
    for epoch in range(start,args.epochs+1):
        started=time.time();model.train();losses=[];optim.zero_grad(set_to_none=True)
        for step,(images,targets) in enumerate(loader):
            images=images.to(device,non_blocking=True);targets=targets.to(device,non_blocking=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16): logits=model(images)
            loss=loss_fn(logits,targets)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss/2).backward()
            if (step+1)%2==0 or step+1==len(loader):
                scaler.unscale_(optim);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optim);scaler.update();optim.zero_grad(set_to_none=True)
                if ema_model is not None:
                    with torch.no_grad():
                        for k,value in ema_model.state_dict().items():
                            source=model.state_dict()[k]
                            if value.is_floating_point():value.lerp_(source,.01)
                            else:value.copy_(source)
            losses.append(float(loss.detach()))
            if step%12==0: print(json.dumps({'epoch':epoch,'step':step,'loss':losses[-1]}),flush=True)
            if args.smoke and step>=1: break
        evaluation_model=ema_model if ema_model is not None else model
        val=evaluate(evaluation_model,splits['val'][:2] if args.smoke else splits['val'],mf,device,thresholds)
        threshold,m=max(val.items(),key=lambda kv:kv[1]['dice']);score=m['dice'];sched.step(score)
        row={'epoch':epoch,'loss':float(np.mean(losses)),'val':m,'threshold':float(threshold),
             'lr':optim.param_groups[0]['lr'],'seconds':time.time()-started,
             'peak_cuda_GiB':torch.cuda.max_memory_allocated()/1024**3}
        history.append(row)
        if score>best+1e-4:
            best=score;best_epoch=epoch;stale=0
            torch.save(evaluation_model.state_dict(),out/'weights.pth')
            mf['inference']['threshold']=float(threshold)
            mf['training']={'config':'training_config.json','best_epoch':epoch,'best_val_dice':score,
                'dataset_sha256':config['dataset_manifest_sha256'],'test_used_for_selection':False,'ema':args.ema}
            dump(out/'model_manifest.json',mf)
        else: stale+=1
        dump(out/'history.json',history)
        checkpoint={'model':model.state_dict(),'optimizer':optim.state_dict(),'scheduler':sched.state_dict(),
            'scaler':scaler.state_dict(),'epoch':epoch,'history':history,'best':best,'best_epoch':best_epoch,'stale':stale,
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        if ema_model is not None:checkpoint['ema_model']=ema_model.state_dict()
        torch.save(checkpoint,out/'resume.tmp.pth');os.replace(out/'resume.tmp.pth',out/'resume.pth')
        dump(out/'status.json',{'status':'training','last_epoch':epoch,'best_epoch':best_epoch,'best_val_dice':best,'last':row})
        print(json.dumps(row),flush=True)
        if args.smoke or stale>=args.patience: break
    dump(out/'status.json',{'status':'smoke_complete' if args.smoke else 'training_complete_test_pending',
        'last_epoch':history[-1]['epoch'],'best_epoch':best_epoch,'best_val_dice':best})
    print('[training complete; independent test remains locked]',flush=True)


if __name__=='__main__': main()
