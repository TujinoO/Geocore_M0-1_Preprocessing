from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from geocore_mask.inference.predictor import CoreMaskPredictor
from geocore_mask.inference.streaming import predict_raster
from geocore_mask.utils.model_package import load_model_package


def main():
    p=argparse.ArgumentParser(description='Bounded-memory V4 U-Net RGB raster inference')
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model-package',default=str(ROOT/'models/core_mask_unet_v4'));p.add_argument('--device',default='auto')
    p.add_argument('--bands',help='Explicit one-based RGB bands, e.g. 3,2,1 for BGR ENVI')
    p.add_argument('--max-rows',type=int,help='Optional prefix for engineering smoke test; omitted processes full raster')
    args=p.parse_args()
    predictor=CoreMaskPredictor(load_model_package(model_package=args.model_package),device=args.device)
    order=[int(v) for v in args.bands.split(',')] if args.bands else None
    result=predict_raster(predictor.model,predictor.manifest,predictor.device,args.input,args.output,bands=order,max_rows=args.max_rows)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
