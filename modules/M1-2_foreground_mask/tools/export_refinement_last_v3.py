import argparse
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    run=Path(args.run);out=run/'refinement_last'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    state=torch.load(run/'refinement/resume.pth',map_location='cpu',weights_only=False)
    torch.save(state['ema_model'],out/'weights.pth')
    mf=json.loads((run/'refinement/model_manifest.json').read_text(encoding='utf-8'))
    mf['model_version']='v3.0.0-narrow-camera-candidate'
    mf['training']['best_epoch']=state['epoch']
    mf['training']['selection']='last EMA state for training-only narrow-camera diagnostic, not independent acceptance'
    (out/'model_manifest.json').write_text(json.dumps(mf,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
