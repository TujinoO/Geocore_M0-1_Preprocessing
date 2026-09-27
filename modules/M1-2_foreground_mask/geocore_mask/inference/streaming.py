"""Windowed GDAL input and incremental GeoTIFF output for large RGB scans."""
from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from geocore_mask.inference.scaled import iter_probability_stripes


def peak_process_memory_bytes():
    import sys
    if sys.platform != 'win32':
        return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
            ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    fn=ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if fn(wintypes.HANDLE(-1),ctypes.byref(counters),counters.cb):
        return int(counters.peak_ws)
    return None


def rgb_bands(ds, path, explicit=None):
    """Return GDAL one-based RGB band indices; handle device's zero-based HDR."""
    if explicit:
        indices=list(explicit)
    elif Path(path).suffix.lower()=='.dat':
        header=Path(path).with_suffix('.hdr')
        if not header.is_file(): header=Path(str(path)+'.hdr')
        match=re.search(r'default\s+bands\s*=\s*\{([^}]+)\}',header.read_text(encoding='utf-8',errors='replace'),re.I)
        if not match: raise ValueError('ENVI RGB band order missing; supply --bands explicitly')
        indices=[int(v.strip()) for v in match.group(1).split(',')]
        if 0 in indices: indices=[v+1 for v in indices]
    else:
        indices=[1,2,3]
    if len(indices)!=3 or len(set(indices))!=3 or min(indices)<1 or max(indices)>ds.RasterCount:
        raise ValueError('Invalid RGB band order: '+str(indices))
    return indices


def predict_raster(model, manifest, device, input_path, output_dir, *, bands=None, max_rows=None):
    from osgeo import gdal
    gdal.UseExceptions();gdal.SetCacheMax(128*1024*1024)
    started=time.time();path=Path(input_path);out=Path(output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError('Choose an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    ds=gdal.Open(str(path),gdal.GA_ReadOnly)
    order=rgb_bands(ds,path,bands)
    if any(ds.GetRasterBand(i).DataType!=gdal.GDT_Byte for i in order):
        raise ValueError('This RGB model requires uint8. Provide a calibrated uint8 conversion for other data types.')
    width=ds.RasterXSize;height=min(ds.RasterYSize,max_rows) if max_rows else ds.RasterYSize
    if height<1: raise ValueError('No rows selected')
    options=['TILED=YES','COMPRESS=DEFLATE','BIGTIFF=IF_SAFER','NUM_THREADS=2']
    mask_path=out/'mask.tif';prob_path=out/'probability.tif'
    mask_ds=gdal.GetDriverByName('GTiff').Create(str(mask_path),width,height,1,gdal.GDT_Byte,options=options)
    prob_ds=gdal.GetDriverByName('GTiff').Create(str(prob_path),width,height,1,gdal.GDT_Float32,options=options)
    gt=ds.GetGeoTransform(can_return_null=True);projection=ds.GetProjection()
    for dest in (mask_ds,prob_ds):
        if gt: dest.SetGeoTransform(gt)
        if projection: dest.SetProjection(projection)
    read=lambda x,y,w,h: np.stack([ds.GetRasterBand(i).ReadAsArray(x,y,w,h) for i in order],axis=2)
    threshold=float(manifest['inference']['threshold']);fg=0;written=0
    warnings=['Streaming output: preview is downsampled; full-resolution outputs are GeoTIFF.']
    if manifest.get('deployment_warning'):warnings.append(manifest['deployment_warning'])
    status_path=out/'metadata.json'
    metadata={'status':'running','input_path':str(path),'model_version':manifest.get('model_version'),
              'warnings':warnings,
              'image_width':width,'image_height':height,'source_height':ds.RasterYSize,'rgb_bands_one_based':order,
              'threshold':threshold,'probability_dtype':'float32','mask_values':[0,255],
              'source_tile_size':manifest['inference']['source_tile_size'],'postprocess':False,
              'preview_only_downsampled':True,'output_scope':'prefix_rows' if height!=ds.RasterYSize else 'full_raster'}
    try:
        for y,prob in iter_probability_stripes(model,read,width,height,manifest,device):
            mask=(prob>=threshold).astype(np.uint8);fg+=int(mask.sum())
            mask_ds.GetRasterBand(1).WriteArray(mask*255,0,y)
            prob_ds.GetRasterBand(1).WriteArray(prob,0,y)
            written=y+len(prob)
            metadata.update(rows_written=written,elapsed_seconds=round(time.time()-started,3))
            status_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
            print(f'[inference] {written}/{height} rows',flush=True)
        mask_ds.FlushCache();prob_ds.FlushCache()
        # Overview is explicitly a preview, never a training or geospatial raster.
        scale=min(1.,1600/max(width,height));pw=max(1,round(width*scale));ph=max(1,round(height*scale))
        rgb=np.stack([ds.GetRasterBand(i).ReadAsArray(0,0,width,height,buf_xsize=pw,buf_ysize=ph) for i in order],axis=2)
        preview_mask=mask_ds.GetRasterBand(1).ReadAsArray(0,0,width,height,buf_xsize=pw,buf_ysize=ph)>0
        overlay=rgb.copy();overlay[preview_mask]=(overlay[preview_mask]*.55+np.array([0,210,255])*.45).astype(np.uint8)
        Image.fromarray(overlay).save(out/'overlay_preview.png')
        metadata.update(status='complete',foreground_area_ratio=fg/(width*height),elapsed_seconds=round(time.time()-started,3),
                        peak_process_working_set_bytes=peak_process_memory_bytes())
    except Exception as exc:
        metadata.update(status='incomplete',rows_written=written,error=str(exc))
        raise
    finally:
        status_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        mask_ds=prob_ds=ds=None
    return {'output_files':{'mask_tif':str(mask_path),'probability_tif':str(prob_path),
              'overlay_png':str(out/'overlay_preview.png'),'metadata_json':str(status_path)},
            'metrics':{'foreground_area_ratio':metadata['foreground_area_ratio'],'elapsed_seconds':metadata['elapsed_seconds']},
            'warnings':warnings}
