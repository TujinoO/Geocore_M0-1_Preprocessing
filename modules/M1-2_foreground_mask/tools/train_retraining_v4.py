"""Hard-example adaptation with explicit same-hole development and retention guard."""
import argparse,copy,json,random,time,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from train_retraining_v3 import CoreDataset,load_model,evaluate,dump
from prepare_retraining_v3 import digest

class ReplayDataset(CoreDataset):
    def __getitem__(self,index):
        names,weights=self.group_names,self.group_weights
        group=random.choices(names,weights=weights)[0]
        self.group_names,self.group_weights=[group],[1.]
        try:x,y=super().__getitem__(index)
        finally:self.group_names,self.group_weights=names,weights
        return x,y,group=='old_other'

def choose(val,adapt):
    eligible=[]
    for t,v in val.items():
        d=adapt[t];score=(v['dice']+d['dice'])/2
        eligible.append((v['dice']>=.895 and d['recall']>=.8,score,t,v,d))
    return max(eligible,key=lambda x:(x[0],x[1]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=28);p.add_argument('--steps',type=int,default=48)
    p.add_argument('--lr',type=float,default=8e-5);p.add_argument('--seed',type=int,default=270927)
    p.add_argument('--background-weight',type=float,default=1.4);p.add_argument('--update-bn',action='store_true')
    p.add_argument('--ema',action='store_true');p.add_argument('--patience',type=int,default=4)
    p.add_argument('--hard-fraction',type=float,default=.5)
    p.add_argument('--replay-distillation',type=float,default=0.)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);dataset=Path(a.dataset)
    import shutil
    shutil.copy2(__file__,out/'executed_training_source.py')
    if json.loads((dataset/'audit.json').read_text(encoding='utf-8'))['errors']:raise ValueError('Audit errors')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True;device=torch.device('cuda')
    rows=json.loads((dataset/'samples.json').read_text(encoding='utf-8'))
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val'];adapt=[r for r in rows if r['split']=='adaptation_dev']
    assert not {r['split_group'] for r in train}&{r['split_group'] for r in val}
    model,mf=load_model(a.base,device);mf=copy.deepcopy(mf)
    mf['model_version']='v4.0.0-hard-example-20260927'
    for key in ('training','validation_calibration','deployment'):mf.pop(key,None)
    mf['deployment_warning']='Adapted candidate: development evidence only; ZKZ4-5 is no longer independent. Review outputs on new scenes.'
    cfg=vars(a).copy();cfg.update(train_count=len(train),val_count=len(val),same_hole_dev_count=len(adapt),
        independent_test_count=0,normalization='Preserve V3 train-only normalization',
        selection='Maximize equal-weight mean Dice of ZK6711 validation and known ZKZ4-5 adaptation development; prefer val Dice>=0.895 and adaptation recall>=0.8. NOT independent acceptance.',
        base_weights_sha256=digest(Path(a.base)/'weights.pth'),dataset_sha256=digest(dataset/'samples.json'),torch_version=torch.__version__,gpu=torch.cuda.get_device_name())
    dump(out/'training_config.json',cfg)
    ds=ReplayDataset(train,dataset,mf['input']['mean'],mf['input']['std'],a.steps,2)
    ds.groups={'new_negative':[r for r in train if r['is_new'] and r['polygons']==0],
               'new_mixed':[r for r in train if r['is_new'] and r['polygons']>0],
               'old_low':[r for r in train if not r['is_new'] and r['native_size']==2048],
               'old_other':[r for r in train if not r['is_new'] and r['native_size']!=2048]}
    ds.group_names=list(ds.groups);ds.group_weights=[a.hard_fraction*.24,a.hard_fraction*.76,.20,.80-a.hard_fraction]
    assert all(ds.groups.values())
    loader=DataLoader(ds,batch_size=2,num_workers=0,pin_memory=True)
    baseline={'val':evaluate(model,val,mf,device,[.4]),'adaptation_dev':evaluate(model,adapt,mf,device,[.4])}
    dump(out/'starting_baseline.json',baseline);print('[baseline] '+json.dumps(baseline),flush=True)
    teacher=copy.deepcopy(model).eval() if a.replay_distillation>0 else None
    if teacher is not None:
        for param in teacher.parameters():param.requires_grad_(False)
    model.outc.conv[1]=nn.Identity()
    ema=copy.deepcopy(model).eval() if a.ema else None
    if ema is not None:
        for param in ema.parameters():param.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=2,min_lr=1e-6)
    scaler=torch.amp.GradScaler('cuda');best=(-1,-1.);stale=0;history=[]
    for epoch in range(1,a.epochs+1):
        start=time.time();model.train()
        if not a.update_bn:
            for layer in model.modules():
                if isinstance(layer,nn.modules.batchnorm._BatchNorm):layer.eval()
        opt.zero_grad(set_to_none=True);losses=[]
        for step,(x,target,replay) in enumerate(loader):
            x=x.to(device);target=target.to(device)
            with torch.autocast(device_type='cuda',dtype=torch.float16):logits=model(x)
            logits=logits.float();both=torch.stack((1-target,target),dim=1)
            weights=1+(a.background_weight-1)*(1-target)
            bce=(F.binary_cross_entropy_with_logits(logits,both,reduction='none')*weights[:,None]).mean()/weights.mean()
            prob=logits[:,1].sigmoid();dice=(2*(prob*target).sum((1,2))+1)/(prob.sum((1,2))+target.sum((1,2))+1)
            loss=bce+(1-dice).mean()
            if teacher is not None and replay.any():
                replay=replay.to(device)
                with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.float16):soft=teacher(x[replay])
                loss=loss+a.replay_distillation*F.binary_cross_entropy_with_logits(logits[replay],soft.float())
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss/2).backward()
            if (step+1)%2==0:
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
                if ema is not None:
                    with torch.no_grad():
                        for k,v in ema.state_dict().items():
                            s=model.state_dict()[k]
                            if v.is_floating_point():v.lerp_(s,.02)
                            else:v.copy_(s)
            losses.append(float(loss.detach()))
        record={'epoch':epoch,'loss':float(np.mean(losses)),'lr':opt.param_groups[0]['lr']}
        if epoch%2==0 or epoch==a.epochs:
            chosen=ema if ema is not None else model
            thresholds=[.35,.45,.55,.65]
            v=evaluate(chosen,val,mf,device,thresholds);d=evaluate(chosen,adapt,mf,device,thresholds)
            ok,score,t,vm,dm=choose(v,d);sched.step(score)
            record.update(guard_passed=ok,selection_score=score,threshold=float(t),val=vm,adaptation_dev=dm)
            rank=(int(ok),score)
            if rank>(best[0],best[1]+1e-4):
                best=rank;stale=0;torch.save(chosen.state_dict(),out/'weights.pth')
                mf['inference']['threshold']=float(t)
                mf['training']={'best_epoch':epoch,'selection_score':score,'validation_dice':vm['dice'],
                    'adaptation_development_dice':dm['dice'],'independent_test_available':False,
                    'normalization_inherited_from':'V3 training only','config':'training_config.json','ema':a.ema}
                dump(out/'model_manifest.json',mf)
            else:stale+=1
            torch.save({'epoch':epoch,'model':model.state_dict(),'ema_model':ema.state_dict() if ema else None,
                'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'scheduler':sched.state_dict(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all()},out/'last_checkpoint.tmp.pth')
            os.replace(out/'last_checkpoint.tmp.pth',out/'last_checkpoint.pth')
        record.update(seconds=time.time()-start,peak_cuda_GiB=torch.cuda.max_memory_allocated()/1024**3)
        history.append(record);dump(out/'history.json',history)
        dump(out/'status.json',{'status':'training','last':record,'best_rank':best,'stale_evaluations':stale})
        print(json.dumps(record),flush=True)
        if stale>=a.patience:break
    dump(out/'status.json',{'status':'training_complete','last_epoch':epoch,'best_rank':best,'independent_test_available':False})

if __name__=='__main__':main()
