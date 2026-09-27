"""Choose the completed candidate on documented development guards, no test claim."""
import argparse,json
from pathlib import Path
from train_retraining_v3 import dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    trials=[]
    for name in ('frozen_bn','replay_ema','blend_50','blend_75','blend_90'):
        if not (run/name).exists():continue
        status=json.loads((run/name/'status.json').read_text(encoding='utf-8'))
        if status['status']!='training_complete':raise RuntimeError('Candidate unfinished: '+name)
        mf=json.loads((run/name/'model_manifest.json').read_text(encoding='utf-8'))
        trials.append({'name':name,'rank':status['best_rank'],'training':mf['training']})
    best=max(trials,key=lambda x:tuple(x['rank']))
    dump(run/'trial_selection.json',{'selected':best['name'],'trials':trials,'independent_test_available':False,
        'criteria':'Prefer validation Dice>=0.895 and same-hole development recall>=0.8, then equal-weight mean Dice.'})
    print(best['name'])

if __name__=='__main__':main()
