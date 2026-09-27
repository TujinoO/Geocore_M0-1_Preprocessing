"""Scout and export pending hard-example annotations without changing old labels.

Scores are model/color heuristics, NOT measured false-positive rates or labels.
All coordinates are native source pixels; annotation PGWs remain tile-local.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from osgeo import gdal, ogr
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train_retraining_v3 import load_model
from geocore_mask.inference.scaled import predict_window
from geocore_mask.inference.streaming import rgb_bands
from create_empty_arcgis_shapefiles import create_shapefile, validate_empty_polygon_shapefile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def intersects(a,b):
    return (max(a['x'],b['x'])<min(a['x']+a['size'],b['x']+b['size']) and
            max(a['y'],b['y'])<min(a['y']+a['size'],b['y']+b['size']))


def read_rgb(ds,order,row,preview=None):
    kw={} if preview is None else {'buf_xsize':preview,'buf_ysize':preview,'resample_alg':gdal.GRIORA_Average}
    return np.stack([ds.GetRasterBand(i).ReadAsArray(row['x'],row['y'],row['size'],row['size'],**kw) for i in order],axis=2)


def sheets(rows,out,folder):
    for page in range((len(rows)+11)//12):
        sheet=Image.new('RGB',(1536,1740),'white');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(rows[page*12:(page+1)*12]):
            x=(j%3)*512;y=(j//3)*435
            with Image.open(out/folder/f"{r['candidate_id']}.jpg") as im:
                sheet.paste(im.resize((400,400)),(x,y))
            draw.text((x+3,y+402),f"{r['candidate_id']} x={r['x']} y={r['y']}",fill='black')
            draw.text((x+3,y+418),f"orange={r['orange_fraction']:.2f} p-orange={r['orange_pred_fraction']:.2f} bright={r['brightness']:.0f}",fill='black')
        sheet.save(out/f'{folder}_sheet_{page+1:02}.jpg',quality=92)


def scout(args):
    out=Path(args.work);out.mkdir(parents=True,exist_ok=False)
    (out/'thumbs').mkdir();(out/'overlays').mkdir()
    source=Path(args.source);ds=gdal.Open(str(source));order=rgb_bands(ds,source)
    with (Path(args.annotation)/'manifests/annotation_tiles.csv').open(encoding='utf-8-sig') as f:
        historical=[{'id':r['tile_id'],'x':int(r['source_x']),'y':int(r['source_y']),'size':int(r['tile_size'])}
                    for r in csv.DictReader(f) if 'ZKZ4-5' in r['group_id']]
    torch.set_num_threads(4);model,mf=load_model(args.model,torch.device('cuda'));model.eval()
    # Fixed native grid, no padding; reject ANY intersection with prior crops.
    size=3072;rows=[];excluded=0
    for y in range(0,ds.RasterYSize-size+1,4096):
        for x in (0,(ds.RasterXSize-size)//2,ds.RasterXSize-size):
            r={'candidate_id':f'C{len(rows)+1:03}','x':x,'y':y,'size':size}
            if any(intersects(r,h) for h in historical):excluded+=1;continue
            rgb=read_rgb(ds,order,r,512)
            rgbf=rgb.astype(np.float32);red,green,blue=rgbf.transpose(2,0,1)
            gray=rgbf.mean(2)
            orange=(red>green*1.18)&(green>blue*1.15)&(red>35)&((red-blue)>22)
            prob=predict_window(model,rgb,mf,torch.device('cuda'))
            pred=prob>=mf['inference']['threshold']
            edge=(np.abs(np.diff(gray,axis=0)).mean()+np.abs(np.diff(gray,axis=1)).mean())/2
            r.update(orange_fraction=float(orange.mean()),orange_pred_fraction=float((orange&pred).mean()),
                     uncertainty=float(((prob>.2)&(prob<.8)).mean()),foreground_prediction=float(pred.mean()),
                     brightness=float(gray.mean()),texture=float(edge),depth_bin=min(11,int(y/ds.RasterYSize*12)))
            # Reject near-uniform scanner lead-in; retain textured empty trays.
            r['eligible']=r['brightness']>12 and r['texture']>1.3
            r['score']=3*r['orange_pred_fraction']+r['orange_fraction']+.5*r['uncertainty']+min(r['texture']/30,.25)
            rows.append(r)
            Image.fromarray(rgb).save(out/'thumbs'/f"{r['candidate_id']}.jpg",quality=92)
            overlay=rgb.copy();overlay[pred]=(overlay[pred]*.55+np.array([0,210,255])*.45).astype(np.uint8)
            Image.fromarray(overlay).save(out/'overlays'/f"{r['candidate_id']}.jpg",quality=90)
        if y%40960==0:print(f'[scout] row={y} candidates={len(rows)}',flush=True)
    # Four proposals per longitudinal zone, preventing single-box domination.
    proposed=[]
    for zone in range(12):
        local=sorted((r for r in rows if r['eligible'] and r['depth_bin']==zone),key=lambda r:r['score'],reverse=True)
        chosen=[]
        for r in local:
            if len(chosen)>=4:break
            if any(intersects(r,s) for s in proposed+chosen):continue
            chosen.append(r)
        proposed+=chosen
    dump(out/'candidates.json',rows);dump(out/'proposals.json',proposed)
    sheets(proposed,out,'thumbs');sheets(proposed,out,'overlays')
    dump(out/'source.json',{'path':str(source),'size_bytes':source.stat().st_size,
         'header_sha256':digest(source.with_suffix('.hdr')),'width':ds.RasterXSize,'height':ds.RasterYSize,
         'rgb_bands_one_based':order,'historical_rectangles_excluded':historical,'excluded_candidates':excluded,
         'ranking':'512px average preview; 3072 source context; model/color heuristic, not ground-truth error',
         'source_group':'ZKZ4-5','independent_test_eligible':False})
    print(f'[scout complete] {len(rows)} candidates, {len(proposed)} proposals; {excluded} overlapping candidates excluded',flush=True)


def details(args):
    out=Path(args.work)
    info=json.loads((out/'source.json').read_text(encoding='utf-8'))
    candidates=json.loads((out/'candidates.json').read_text(encoding='utf-8'))
    chosen=json.loads(Path(args.selection).read_text(encoding='utf-8'))
    ids={s['candidate_id'] for s in chosen}
    blocked=[r for r in candidates if r['candidate_id'] in ids]+info['historical_rectangles_excluded']
    proposed=[]
    for parent in candidates:
        if not parent['candidate_id'].startswith('C'):continue
        with Image.open(out/'thumbs'/f"{parent['candidate_id']}.jpg") as im:thumb=np.asarray(im).astype(np.float32)
        for dy in (0,1024,2048):
            for dx in (0,1024,2048):
                r={'x':parent['x']+dx,'y':parent['y']+dy,'size':1024,'depth_bin':parent['depth_bin'],'parent':parent['candidate_id']}
                if any(intersects(r,b) for b in blocked):continue
                chip=thumb[round(dy/6):round((dy+1024)/6),round(dx/6):round((dx+1024)/6)]
                red,green,blue=chip.transpose(2,0,1)
                orange=(red>green*1.3)&(green>blue*1.9)&(red>45)&((red-blue)>35)
                r['strict_orange_fraction']=float(orange.mean())
                if r['strict_orange_fraction']>.20:proposed.append(r)
    picks=[]
    for zone in range(12):
        local=sorted((r for r in proposed if r['depth_bin']==zone),key=lambda r:r['strict_orange_fraction'],reverse=True)
        zone_picks=[]
        for r in local:
            if len(zone_picks)>=2:break
            if any(intersects(r,b) or r['parent']==b.get('parent') for b in picks+zone_picks):continue
            zone_picks.append(r)
        picks+=zone_picks
    ds=gdal.Open(info['path']);torch.set_num_threads(4)
    model,mf=load_model(args.model,torch.device('cuda'));model.eval()
    for i,r in enumerate(picks):
        r['candidate_id']=f'D{i+1:03}'
        rgb=read_rgb(ds,info['rgb_bands_one_based'],r,512)
        prob=predict_window(model,rgb,mf,torch.device('cuda'));pred=prob>=mf['inference']['threshold']
        r.update(orange_fraction=r['strict_orange_fraction'],orange_pred_fraction=float(pred.mean()),
                 brightness=float(rgb.mean()),foreground_prediction=float(pred.mean()))
        Image.fromarray(rgb).save(out/'thumbs'/f"{r['candidate_id']}.jpg",quality=92)
        overlay=rgb.copy();overlay[pred]=(overlay[pred]*.55+np.array([0,210,255])*.45).astype(np.uint8)
        Image.fromarray(overlay).save(out/'overlays'/f"{r['candidate_id']}.jpg",quality=90)
    dump(out/'detail_proposals.json',picks)
    dump(out/'candidates_with_details.json',candidates+picks)
    # Preserve the main scout contact sheets.
    detail_out=out/'detail_review';detail_out.mkdir(exist_ok=False)
    import shutil
    for folder in ('thumbs','overlays'):
        (detail_out/folder).mkdir()
        for r in picks:shutil.copy2(out/folder/f"{r['candidate_id']}.jpg",detail_out/folder/f"{r['candidate_id']}.jpg")
        sheets(picks,detail_out,folder)
    print(f'[details] {len(picks)} proposals',flush=True)


def export(args):
    out=Path(args.work);annotation=Path(args.annotation)
    name='hard_examples_20260926'
    manifest=annotation/'manifests'/f'{name}.json'
    progress=annotation/'manifests'/f'{name}_标注进度.csv'
    gallery=annotation/'困难样本补充_20260926.html'
    preview_dir=annotation/'previews'/name
    for target in (manifest,progress,gallery,preview_dir,out/'exported_samples.json',out/'existing_files_before.json'):
        if target.exists():raise FileExistsError(str(target))
    info=json.loads((out/'source.json').read_text(encoding='utf-8'))
    candidate_file=out/'candidates_with_details.json'
    if not candidate_file.exists():candidate_file=out/'candidates.json'
    candidates={r['candidate_id']:r for r in json.loads(candidate_file.read_text(encoding='utf-8'))}
    selected=json.loads(Path(args.selection).read_text(encoding='utf-8'))
    if args.additional_selection:
        selected+=json.loads(Path(args.additional_selection).read_text(encoding='utf-8'))
    rows=[dict(candidates[s['candidate_id']],selection_reason=s['reason'],annotation_focus=s['focus']) for s in selected]
    if len({r['candidate_id'] for r in rows})!=len(rows):raise ValueError('Duplicate selection')
    for i,r in enumerate(rows):
        if any(intersects(r,s) for s in rows[:i]+info['historical_rectangles_excluded']):raise ValueError('Overlapping selected tile')
    old_files=[p for d in ('images','shp') for p in (annotation/d).iterdir() if p.is_file()]
    before={str(p):digest(p) for p in old_files};dump(out/'existing_files_before.json',before)
    source=Path(info['path']);ds=gdal.Open(str(source));order=info['rgb_bands_one_based']
    if (source.stat().st_size!=info['size_bytes'] or digest(source.with_suffix('.hdr'))!=info['header_sha256']
            or (ds.RasterXSize,ds.RasterYSize)!=(info['width'],info['height'])):
        raise ValueError('Source metadata changed after scouting')
    # Numbering follows the historical maximum, never fills deleted/missing IDs.
    with (annotation/'manifests/annotation_tiles.csv').open(encoding='utf-8-sig') as f:
        largest=max(int(r['tile_id']) for r in csv.DictReader(f) if r['tile_id'].isdigit())
    largest=max([largest]+[int(p.stem) for p in (annotation/'images').glob('*.png') if p.stem.isdigit()])
    for i,r in enumerate(rows):
        r['tile_id']=f'{largest+i+1:03}'
        paths=[annotation/'images'/f"{r['tile_id']}{ext}" for ext in ('.png','.pgw')]
        paths += [annotation/'shp'/f"{r['tile_id']}{ext}" for ext in ('.shp','.shx','.dbf','.cpg','.prj')]
        if any(p.exists() for p in paths):raise FileExistsError(str(paths))
    exported=[]
    preview_dir.mkdir()
    for r in rows:
        key=r['tile_id'];png=annotation/'images'/f'{key}.png';shp=annotation/'shp'/f'{key}.shp'
        rgb=read_rgb(ds,order,r)
        Image.fromarray(rgb).save(png)
        Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR).save(preview_dir/f'{key}.jpg',quality=92)
        png.with_suffix('.pgw').write_text('1\n0\n0\n-1\n0.5\n-0.5\n',encoding='ascii')
        create_shapefile(shp.with_suffix('.png'),ogr)
        with Image.open(png) as im:
            if not np.array_equal(np.asarray(im),rgb):raise AssertionError('PNG differs from native RGB')
        validate_empty_polygon_shapefile(shp,ogr)
        test=gdal.Open(str(png));assert test.GetGeoTransform()==(0.,1.,0.,0.,0.,-1.);test=None
        hashes={p.suffix:digest(p) for p in [png,png.with_suffix('.pgw')]+[shp.with_suffix(e) for e in ('.shp','.shx','.dbf','.cpg')]}
        r.update(image_path=str(png),shp_path=str(shp),source_path=str(source),source_group='ZKZ4-5',
                 review_status='pending',training_eligible=False,independent_test_eligible=False,hashes=hashes)
        exported.append(r);print(f'[export] {key} <- {r["candidate_id"]} ({r["x"]},{r["y"]})',flush=True)
    if any(digest(p)!=sha for p,sha in before.items()):raise AssertionError('Pre-existing file changed')
    dump(out/'exported_samples.json',exported)
    dump(manifest,exported)
    # Editable review status is deliberately separate from old frozen manifests.
    columns=['tile_id','candidate_id','review_status','reviewer','review_note','selection_reason','annotation_focus','source_group','x','y','size','image_path','shp_path']
    with progress.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        writer.writerows(exported)
    dump(out/'export_verification.json',{'count':len(exported),'native_png_matches':len(exported),
        'empty_polygon_shp_valid':len(exported),'pgw_matches':len(exported),'overlap_with_historical':0,
        'overlap_between_new':0,'unchanged_existing_files':len(before),'pending':len(exported),
        'training_started':False,'raw_source_modified':False})
    cards=[]
    for r in exported:
        cards.append(f'<article><h3>{r["tile_id"]} · {html.escape(r["selection_reason"])}</h3><a href="images/{r["tile_id"]}.png"><img loading="lazy" src="previews/{name}/{r["tile_id"]}.jpg"></a><p>{html.escape(r["annotation_focus"])}</p><p>原图尺寸 {r["size"]}×{r["size"]}；坐标 x={r["x"]}, y={r["y"]}；状态：待人工标注</p></article>')
    with gallery.open('x',encoding='utf-8') as f:
        f.write('<!doctype html><meta charset="utf-8"><title>困难样本补充</title><style>body{font:16px sans-serif;margin:24px;background:#eee}main{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}article{background:white;padding:12px}img{width:100%}</style><h1>困难样本补充：待标注，不可直接训练</h1><p>只画岩心，不画箱体。空SHP是待编辑模板，不等于已确认负样本。请在对应CSV中记录 approved 或 approved_negative。</p><main>'+''.join(cards)+'</main>')


def verify(args):
    import shutil
    out=Path(args.work);annotation=Path(args.annotation)
    rows=json.loads((out/'exported_samples.json').read_text(encoding='utf-8'))
    before=json.loads((out/'existing_files_before.json').read_text(encoding='utf-8'))
    assert all(digest(p)==h for p,h in before.items()),'Existing file modified'
    old_png_hashes={h for p,h in before.items() if Path(p).suffix.lower()=='.png'}
    new_hashes=set()
    for r in rows:
        png=Path(r['image_path']);shp=Path(r['shp_path'])
        validate_empty_polygon_shapefile(shp,ogr)
        with Image.open(png) as im:assert im.size==(r['size'],r['size']) and im.mode=='RGB'
        assert r['hashes']['.png'] not in old_png_hashes|new_hashes,'Duplicate PNG'
        new_hashes.add(r['hashes']['.png'])
        for ext,h in r['hashes'].items():
            path=png.with_suffix(ext) if ext in ('.png','.pgw') else shp.with_suffix(ext)
            assert digest(path)==h,'Export changed before delivery'
    target=annotation/'previews/hard_examples_20260926'
    for page in range((len(rows)+11)//12):
        path=target/f'补充样本总览_{page+1:02}.jpg'
        if path.exists():raise FileExistsError(str(path))
        sheet=Image.new('RGB',(1152,1656),'white');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(rows[page*12:(page+1)*12]):
            x=(j%3)*384;y=(j//3)*414
            with Image.open(target/f"{r['tile_id']}.jpg") as im:sheet.paste(im.resize((384,384)),(x,y))
            draw.text((x+5,y+389),f"{r['tile_id']}  {r['size']} x {r['size']}  PENDING",fill='black')
        sheet.save(path,quality=92)
    report=json.loads((out/'export_verification.json').read_text(encoding='utf-8'))
    report.update(final_file_hashes_verified=True,duplicate_new_png=0,
                  new_png_duplicates_old=0,final_existing_files_unchanged=len(before))
    dest=annotation/'manifests/hard_examples_20260926_检查结果.json'
    if dest.exists():raise FileExistsError(str(dest))
    dump(dest,report)
    shutil.copy2(out/'source_copy_check.json',annotation/'manifests/hard_examples_20260926_来源抽查.json')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['scout','details','export','verify'])
    p.add_argument('--source');p.add_argument('--annotation',required=True);p.add_argument('--work',required=True)
    p.add_argument('--model');p.add_argument('--selection');p.add_argument('--additional-selection');args=p.parse_args()
    gdal.UseExceptions();ogr.UseExceptions()
    {'scout':scout,'details':details,'export':export,'verify':verify}[args.mode](args)


if __name__=='__main__':main()
