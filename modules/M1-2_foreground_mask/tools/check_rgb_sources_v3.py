from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from osgeo import gdal

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from geocore_mask.inference.streaming import rgb_bands


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--raw',required=True)
    args=p.parse_args();dataset=Path(args.dataset);raw=Path(args.raw)
    rows=json.loads((dataset/'samples.json').read_text(encoding='utf-8'))
    selected={}
    for row in rows:selected.setdefault(row['group'],row)
    files=list(raw.glob('*.dat'));report=[]
    for group,row in selected.items():
        matches=[f for f in files if f.stem.replace('(','_').replace(')','_')==group]
        if len(matches)!=1:raise ValueError('Ambiguous raw source: '+group)
        path=matches[0];ds=gdal.Open(str(path))
        try:
            order=rgb_bands(ds,path);explicit_required=False
        except ValueError as exc:
            if 'band order missing' not in str(exc):raise
            order=rgb_bands(ds,path,[1,2,3]);explicit_required=True
        s=row['native_size'];x=row['source_x'];y=row['source_y']
        arr=np.stack([ds.GetRasterBand(i).ReadAsArray(x,y,s,s) for i in order],axis=2)
        with Image.open(row['image']) as im:target=np.asarray(im.convert('RGB'))
        same=bool(np.array_equal(arr,target))
        report.append({'id':row['id'],'group':group,'bands':order,'explicit_bands_required':explicit_required,'exact_rgb_match':same})
        if not same:raise ValueError('Raw inference RGB differs from annotation PNG: '+group)
        ds=None
    (dataset.parent/'raw_rgb_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
