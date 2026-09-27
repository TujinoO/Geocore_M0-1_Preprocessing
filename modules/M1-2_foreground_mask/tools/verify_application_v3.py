"""Verify final candidate through application service and actual ENVI outputs."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from osgeo import gdal

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prepare_retraining_v3 import digest
from train_retraining_v3 import dump, load_model
from geocore_mask.inference.scaled import predict_array
from geocore_mask.inference.streaming import rgb_bands


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--raw',required=True)
    args=p.parse_args();run=Path(args.run);package=run/'candidate';raw=Path(args.raw)
    sys.path.insert(0,str(run/'api_test_deps'))
    from api import service
    from api.schemas import ForegroundMaskRequest
    rows=json.loads((run/'dataset/samples.json').read_text(encoding='utf-8'))
    unchanged=[]
    for row in rows:
        image=Path(row['image']);shp=image.parent.parent/'shp'/f"{row['id']}.shp"
        paths={'png':image,'pgw':image.with_suffix('.pgw'),**{ext:shp.with_suffix('.'+ext) for ext in ('shp','shx','dbf')}}
        if any(digest(path)!=row['hashes'][key] for key,path in paths.items()):
            raise ValueError('Source changed during training: '+row['id'])
        unchanged.append(row['id'])
    # Actual application service dispatch, not just loading the neural network.
    # Float16 AMP/cuDNN is not guaranteed bit-identical across processes.
    # Check a direct probability reference, plus tightly bounded threshold flips.
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    service.RUNTIME_DIR=run/'api_runtime';service.UPLOAD_DIR=service.RUNTIME_DIR/'uploads';service.OUTPUT_DIR=service.RUNTIME_DIR/'outputs'
    row=next(r for r in rows if r['id']=='145')
    task=service.create_task()
    request=ForegroundMaskRequest(input_path=row['image'],model_package=str(package),output_dir=str(run/('api_smoke_'+task.task_id)),enable_postprocess=False)
    service.run_foreground_mask(task.task_id,request)
    result=service.TASKS[task.task_id]
    if result.error:raise RuntimeError(result.error)
    with Image.open(result.output_files['mask_png']) as im:api_mask=np.asarray(im)
    with Image.open(package/'test_evaluation/145_mask.png') as im:test_mask=np.asarray(im)
    changed=api_mask!=test_mask
    ds=gdal.Open(result.output_files['probability_tif'])
    if ds.GetRasterBand(1).DataType!=gdal.GDT_Float32:raise AssertionError('Probability export loses precision')
    api_prob=ds.ReadAsArray();ds=None
    model,mf=load_model(package,torch.device('cuda'));model.eval()
    with Image.open(row['image']) as im:api_reference_image=np.asarray(im.convert('RGB'))
    reference=predict_array(model,api_reference_image,mf,torch.device('cuda'))
    api_diff=float(np.max(np.abs(reference-api_prob)))
    flip_fraction=float(changed.mean())
    flip_margin=float(np.max(np.abs(api_prob[changed]-mf['inference']['threshold']))) if changed.any() else 0.
    api_comparison={'exact_frozen_mask_match':not bool(changed.any()),
        'different_pixels':int(changed.sum()),'total_pixels':int(changed.size),
        'different_pixel_fraction':flip_fraction,'max_flip_threshold_margin':flip_margin,
        'direct_array_probability_max_difference':api_diff,
        'limits':{'probability_max_difference':1e-3,'frozen_mask_flip_fraction':1e-4,'flip_threshold_margin':1e-3}}
    dump(run/'api_numeric_comparison.json',api_comparison)
    if api_diff>1e-3 or flip_fraction>1e-4 or flip_margin>1e-3:
        raise AssertionError('API differs beyond float16 numerical tolerance: '+str(api_comparison))
    if not np.array_equal(api_mask,(api_prob>=mf['inference']['threshold']).astype(np.uint8)*255):
        raise AssertionError('API mask is inconsistent with its exported probabilities')
    del api_mask,test_mask,changed,api_prob,reference,api_reference_image
    # Compare the full short raw raster against identical in-memory inference.
    torch.backends.cudnn.benchmark=False  # default used by the standalone CLI
    short=raw/'RGB-20230910_090147-00000.dat';ds=gdal.Open(str(short))
    arr=np.stack([ds.GetRasterBand(i).ReadAsArray() for i in (1,2,3)],axis=2)
    reference=predict_array(model,arr,mf,torch.device('cuda'))
    ds=None;del arr
    actual=gdal.Open(str(run/'raw_short_full/probability.tif'))
    diff=float(np.max(np.abs(reference-actual.ReadAsArray())));actual=None;del reference
    if diff>1e-6:raise AssertionError('Streaming differs from full-array prediction: '+str(diff))
    # Verify optional profile really loads and produces finite native-size output.
    narrow_model,narrow_mf=load_model(run/'refinement_last_narrow_bn_probe',torch.device('cuda'))
    narrow_row=next(r for r in rows if r['native_size']==2048)
    with Image.open(narrow_row['image']) as im:narrow_image=np.asarray(im.convert('RGB'))
    narrow_prob=predict_array(narrow_model,narrow_image,narrow_mf,torch.device('cuda'))
    if narrow_prob.shape!=narrow_image.shape[:2] or not np.isfinite(narrow_prob).all():
        raise AssertionError('Invalid narrow-profile output')
    if narrow_prob.min()<0 or narrow_prob.max()>1:raise AssertionError('Narrow probabilities out of range')
    del narrow_model,narrow_image,narrow_prob
    raw_reports=[]
    for folder in ('raw_short_full','raw_highres_prefix'):
        meta=json.loads((run/folder/'metadata.json').read_text(encoding='utf-8'))
        if meta['status']!='complete':raise AssertionError('Incomplete raster output')
        mask=gdal.Open(str(run/folder/'mask.tif'));prob=gdal.Open(str(run/folder/'probability.tif'))
        if (mask.RasterXSize,mask.RasterYSize)!=(meta['image_width'],meta['image_height']):raise AssertionError('Output extent mismatch')
        for y in (0,mask.RasterYSize//2,max(0,mask.RasterYSize-64)):
            h=min(64,mask.RasterYSize-y);pa=prob.ReadAsArray(0,y,mask.RasterXSize,h);ma=mask.ReadAsArray(0,y,mask.RasterXSize,h)
            if not np.isfinite(pa).all() or pa.min()<0 or pa.max()>1:raise AssertionError('Invalid probabilities')
            if not np.array_equal(ma,(pa>=mf['inference']['threshold']).astype(np.uint8)*255):raise AssertionError('Mask threshold mismatch')
        raw_reports.append(meta);mask=prob=None
    dump(run/'engineering_checks.json',{'passed':True,'checks':{
        'synthetic_streaming_and_probability_tests':4,'source_images_and_labels_unchanged':len(unchanged),
        'raw_rgb_exact_match_groups':9,'application_service_numeric_comparison':api_comparison,
        'application_service_output':result.output_files,
        'application_backend':'cudnn.benchmark=False; AMP enabled; numerical tolerance rather than bitwise guarantee',
        'optional_narrow_profile_load_and_inference':True,
        'probability_export_float32':True,'short_full_raster_stream_vs_array_max_difference':diff},
        'actual_raster_runs':raw_reports,'limits':'Long high-resolution ENVI was checked on a prefix; this is not a full-borehole accuracy or throughput certification.'})
    print('[engineering checks passed]',flush=True)


if __name__=='__main__':main()
