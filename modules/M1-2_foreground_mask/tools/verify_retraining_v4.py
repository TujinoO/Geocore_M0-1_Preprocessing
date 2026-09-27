"""Check immutable labels, application service, and full raw inference outputs."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from osgeo import gdal
import torch
from prepare_retraining_v3 import digest
from train_retraining_v3 import dump,load_model
from geocore_mask.inference.scaled import predict_array

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--model',required=True)
    p.add_argument('--api-deps',required=True);p.add_argument('--raw',required=True);a=p.parse_args();run=Path(a.run)
    sys.path.insert(0,a.api_deps)
    from api import service
    from api.schemas import ForegroundMaskRequest
    rows=json.loads((run/'dataset/samples.json').read_text(encoding='utf-8'))
    for r in rows:
        image=Path(r['image']);shp=image.parent.parent/'shp'/f"{r['id']}.shp"
        paths={'png':image,'pgw':image.with_suffix('.pgw'),**{k:shp.with_suffix('.'+k) for k in ('shp','shx','dbf')}}
        for k,path in paths.items():
            if digest(path)!=r['hashes'][k]:raise ValueError('Source changed: '+str(path))
    output=run/'engineering';output.mkdir(exist_ok=False)
    service.RUNTIME_DIR=output/'api_runtime';service.UPLOAD_DIR=service.RUNTIME_DIR/'uploads';service.OUTPUT_DIR=service.RUNTIME_DIR/'outputs'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    model,mf=load_model(a.model,torch.device('cuda'));model.eval();checks=[]
    for key in ('145','195','206'):
        row=next(r for r in rows if r['id']==key);task=service.create_task()
        request=ForegroundMaskRequest(input_path=row['image'],model_package=a.model,output_dir=str(output/f'api_{key}'),enable_postprocess=False)
        service.run_foreground_mask(task.task_id,request);result=service.TASKS[task.task_id]
        if result.error:raise RuntimeError(result.error)
        with Image.open(row['image']) as im:im=np.asarray(im.convert('RGB'))
        reference=predict_array(model,im,mf,torch.device('cuda'))
        ds=gdal.Open(result.output_files['probability_tif']);assert ds.GetRasterBand(1).DataType==gdal.GDT_Float32
        prob=ds.ReadAsArray();ds=None;diff=float(np.abs(prob-reference).max())
        assert diff<=.001
        with Image.open(result.output_files['mask_png']) as im:mask=np.asarray(im)
        assert np.array_equal(mask,(prob>=mf['inference']['threshold']).astype(np.uint8)*255)
        checks.append({'id':key,'max_probability_difference':diff,'threshold_mask_consistent':True,'warnings':result.warnings})
    reports=[]
    for name in ('raw_short_full','raw_ZKZ4_full'):
        folder=run/name;meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        assert meta['status']=='complete' and meta['output_scope']=='full_raster'
        mask=gdal.Open(str(folder/'mask.tif'));prob=gdal.Open(str(folder/'probability.tif'));src=gdal.Open(meta['input_path'])
        assert (mask.RasterXSize,mask.RasterYSize)==(src.RasterXSize,src.RasterYSize)
        assert mask.GetGeoTransform()==src.GetGeoTransform() and mask.GetProjection()==src.GetProjection()
        checked=0
        for y in np.linspace(0,mask.RasterYSize-64,32).astype(int):
            pa=prob.ReadAsArray(0,int(y),mask.RasterXSize,64);ma=mask.ReadAsArray(0,int(y),mask.RasterXSize,64)
            assert np.isfinite(pa).all() and pa.min()>=0 and pa.max()<=1
            assert np.array_equal(ma,(pa>=mf['inference']['threshold']).astype(np.uint8)*255)
            checked+=pa.size
        reports.append({'metadata':meta,'sampled_checked_pixels':checked});mask=prob=src=None
    dump(run/'engineering_checks.json',{'passed':True,'source_pairs_unchanged':len(rows),
        'application_checks':checks,'full_raster_runs':reports,'scope':'Engineering verification only, not independent segmentation accuracy; HTTP server/frontend not exercised.'})
    print('[V4 engineering checks passed]',flush=True)

if __name__=='__main__':main()
