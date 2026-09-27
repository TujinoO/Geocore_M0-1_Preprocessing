"""Freeze completed hard examples and preserve V3 evidence without source edits."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from osgeo import gdal,ogr
from prepare_retraining_v3 import digest,save_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--previous',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();source=Path(a.source);old=Path(a.previous);out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    for d in ('masks','cache','qa'):(out/d).mkdir()
    gdal.UseExceptions();ogr.UseExceptions()
    rows=json.loads((old/'samples.json').read_text(encoding='utf-8'))
    additions=json.loads((source/'manifests/hard_examples_20260926.json').read_text(encoding='utf-8'))
    expected={r['id'] for r in rows}|{r['tile_id'] for r in additions}
    actual={x.stem for x in (source/'images').glob('*.png')}
    if actual!=expected:raise ValueError('Unexpected image inventory: '+str(actual^expected))
    for r in rows:
        im=Path(r['image']);shp=source/'shp'/f"{r['id']}.shp"
        paths={'png':im,'pgw':im.with_suffix('.pgw'),**{e:shp.with_suffix('.'+e) for e in ('shp','shx','dbf')}}
        for k,path in paths.items():
            if digest(path)!=r['hashes'][k]:raise ValueError('Previous input changed: '+str(path))
        shutil.copy2(r['mask'],out/'masks'/f"{r['id']}.png")
        r['mask']=str(out/'masks'/f"{r['id']}.png")
        for suffix in ('rgb','mask'):shutil.copy2(old/'cache'/f"{r['id']}_{suffix}.npy",out/'cache'/f"{r['id']}_{suffix}.npy")
        r['previous_split']=r['split']
        if r['split']=='test':r['split']='adaptation_dev'
        r['is_new']=False
    errors=[];empty=[]
    for r in additions:
        key=r['tile_id'];image=source/'images'/f'{key}.png';shp=source/'shp'/f'{key}.shp'
        if digest(image)!=r['hashes']['.png']:raise ValueError('New image pixels changed: '+key)
        ds=gdal.Open(str(image));gt=ds.GetGeoTransform();assert gt==(0.,1.,0.,0.,0.,-1.)
        vector=ogr.Open(str(shp));layer=vector.GetLayer(0);n=layer.GetFeatureCount()
        for feature in layer:
            geom=feature.GetGeometryRef()
            if geom is None or geom.IsEmpty() or not geom.IsValid():errors.append({'id':key,'fid':feature.GetFID(),'error':'invalid or empty geometry'})
        layer.ResetReading()
        mem=gdal.GetDriverByName('MEM').Create('',ds.RasterXSize,ds.RasterYSize,1,gdal.GDT_Byte);mem.SetGeoTransform(gt);mem.GetRasterBand(1).Fill(0)
        if n:gdal.RasterizeLayer(mem,[1],layer,burn_values=[1])
        mask=mem.ReadAsArray();im=Image.open(image).convert('RGB')
        if n and not mask.any():errors.append({'id':key,'error':'no rasterized foreground'})
        if not n:empty.append(key)
        Image.fromarray(mask*255).save(out/'masks'/f'{key}.png')
        np.save(out/'cache'/f'{key}_rgb.npy',np.asarray(im.resize((1024,1024),Image.Resampling.BILINEAR)))
        np.save(out/'cache'/f'{key}_mask.npy',np.asarray(Image.fromarray(mask).resize((1024,1024),Image.Resampling.NEAREST)))
        thumb=np.asarray(im.resize((384,384)));m=np.asarray(Image.fromarray(mask).resize((384,384),Image.Resampling.NEAREST)).astype(bool)
        overlay=thumb.copy();overlay[m]=(overlay[m]*.55+np.array([0,210,255])*.45).astype(np.uint8)
        panel=Image.new('RGB',(768,410),'white');panel.paste(Image.fromarray(thumb),(0,0));panel.paste(Image.fromarray(overlay),(384,0))
        ImageDraw.Draw(panel).text((5,388),f'{key} polygons={n} foreground={mask.mean():.4f}',fill='black');panel.save(out/'qa'/f'{key}.jpg',quality=92)
        paths={'png':image,'pgw':image.with_suffix('.pgw'),**{e:shp.with_suffix('.'+e) for e in ('shp','shx','dbf')}}
        rows.append({'id':key,'image':str(image),'mask':str(out/'masks'/f'{key}.png'),'group':'ZKZ4-5_hard_examples',
            'split_group':next(x['split_group'] for x in rows if x['split']=='adaptation_dev'),'split':'train',
            'previous_split':'pending','native_size':im.width,'source_x':r['x'],'source_y':r['y'],'polygons':n,
            'foreground_ratio':float(mask.mean()),'hashes':{k:digest(v) for k,v in paths.items()},'is_new':True,
            'annotation_authority':'User declared all 36 additions complete on 2026-09-27; old progress CSV remains pending and was not overwritten.'})
        print(key,n,float(mask.mean()),flush=True);ds=mem=vector=None
    train_groups={r['split_group'] for r in rows if r['split']=='train'};val_groups={r['split_group'] for r in rows if r['split']=='val'}
    assert not train_groups&val_groups
    for page in range(6):
        sheet=Image.new('RGB',(1536,1230),'white')
        for j,r in enumerate(additions[page*6:(page+1)*6]):
            with Image.open(out/'qa'/f"{r['tile_id']}.jpg") as im:sheet.paste(im,((j%2)*768,(j//2)*410))
        sheet.save(out/'qa'/f'contact_{page+1:02}.jpg',quality=92)
    save_json(out/'samples.json',rows)
    save_json(out/'audit.json',{'errors':errors,'count':len(rows),'new_count':36,'empty_new_ids':empty,
        'splits':{s:sum(r['split']==s for r in rows) for s in ('train','val','adaptation_dev')},
        'independent_test_available':False,'split_note':'ZK6711 remains out of gradient training. Prior ZKZ4-5 test is now same-hole adaptation development, used for selection, NOT independent.',
        'source_authority':'Current user declaration; visual QA required before training.'})
    if errors:raise ValueError(str(errors))

if __name__=='__main__':main()
