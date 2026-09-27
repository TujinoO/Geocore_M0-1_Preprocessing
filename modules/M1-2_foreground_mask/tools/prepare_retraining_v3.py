"""Freeze reviewed, numbered ArcGIS annotations without altering source files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr
from PIL import Image, ImageDraw

gdal.UseExceptions()
ogr.UseExceptions()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, out = Path(args.source), Path(args.output)
    if out.exists():
        raise FileExistsError('Choose a fresh dataset output: ' + str(out))
    out.mkdir(parents=True)
    for folder in ('masks', 'qa', 'cache'):
        (out/folder).mkdir()
    with (source/'manifests/annotation_tiles.csv').open(encoding='utf-8-sig') as f:
        lookup = {r['tile_id']: r for r in csv.DictReader(f)}
    rows, repairs, errors = [], [], []
    for path in sorted((source/'images').glob('*.png')):
        key = path.stem
        row = lookup[key]
        sha = digest(path)
        if sha != row['sha256']:
            raise ValueError('Source PNG differs from numbered source manifest: ' + key)
        im = Image.open(path).convert('RGB')
        w, h = im.size
        hashes = {'png': sha}
        shp = source/'shp'/f'{key}.shp'
        for ext in ('.shp', '.shx', '.dbf'):
            hashes[ext[1:]] = digest(shp.with_suffix(ext))
        hashes['pgw'] = digest(path.with_suffix('.pgw'))
        raster = gdal.Open(str(path), gdal.GA_ReadOnly)
        gt = raster.GetGeoTransform(can_return_null=True)
        if gt is None or not np.allclose(gt, (0,1,0,0,0,-1)):
            raise ValueError('Unexpected pixel transform: ' + key + ' ' + str(gt))
        vector = ogr.Open(str(shp), 0)
        layer = vector.GetLayer(0)
        memory = ogr.GetDriverByName('Memory').CreateDataSource('')
        clean = memory.CreateLayer('core', geom_type=ogr.wkbMultiPolygon)
        polygons = 0
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None or geom.IsEmpty():
                continue
            geom = geom.Clone()
            if not geom.IsValid():
                old_area = geom.GetArea()
                geom = geom.MakeValid()
                repairs.append({'id':key, 'fid':feature.GetFID(), 'old_area':old_area, 'new_area':geom.GetArea()})
            if geom.IsEmpty() or not geom.IsValid():
                errors.append({'id':key,'error':'unrepairable geometry'})
                continue
            if ogr.GT_Flatten(geom.GetGeometryType()) not in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
                errors.append({'id':key,'error':'non-polygon geometry after repair'})
                continue
            f = ogr.Feature(clean.GetLayerDefn()); f.SetGeometry(geom); clean.CreateFeature(f)
            polygons += 1
        target = gdal.GetDriverByName('MEM').Create('',w,h,1,gdal.GDT_Byte)
        target.SetGeoTransform(gt)
        target.GetRasterBand(1).Fill(0)
        if polygons:
            gdal.RasterizeLayer(target,[1],clean,burn_values=[1])
        mask = target.ReadAsArray().astype(np.uint8)
        if polygons and not mask.any():
            errors.append({'id':key,'error':'all polygons outside image'})
        Image.fromarray(mask*255).save(out/'masks'/f'{key}.png')
        group = row['group_id']
        # Keep all potentially same-hole 20230910 acquisitions together.
        split_group = 'campaign_20230910' if group.startswith('RGB-20230910') else group
        split = 'val' if 'ZK6711-' in group else ('test' if 'ZKZ4-5_' in group else 'train')
        # A spatially downsampled uint8 cache preserves aspect ratio and reduces
        # repeated large-PNG decode overhead. Native masks remain available for QA.
        cache_size = 1024
        small = im.resize((cache_size,cache_size),Image.Resampling.BILINEAR)
        small_mask = Image.fromarray(mask).resize((cache_size,cache_size),Image.Resampling.NEAREST)
        np.save(out/'cache'/f'{key}_rgb.npy',np.asarray(small))
        np.save(out/'cache'/f'{key}_mask.npy',np.asarray(small_mask))
        thumb = im.resize((384,384),Image.Resampling.BILINEAR)
        m = np.asarray(Image.fromarray(mask).resize((384,384),Image.Resampling.NEAREST)).astype(bool)
        overlay = np.asarray(thumb).copy()
        overlay[m] = (overlay[m]*0.6 + np.array([0,220,255])*0.4).astype(np.uint8)
        panel = Image.new('RGB',(768,410),'white')
        panel.paste(thumb,(0,0)); panel.paste(Image.fromarray(overlay),(384,0))
        ImageDraw.Draw(panel).text((5,386),f'{key}  {split}  polygons={polygons}  foreground={mask.mean():.3f}',fill='black')
        panel.save(out/'qa'/f'{key}.jpg',quality=90)
        rows.append({'id':key, 'image':str(path), 'mask':str(out/'masks'/f'{key}.png'),
                     'group':group, 'split_group':split_group, 'split':split, 'native_size':w,
                     'source_x':int(row['source_x']), 'source_y':int(row['source_y']),
                     'polygons':polygons, 'foreground_ratio':float(mask.mean()), 'hashes':hashes,
                     'annotation_authority':'user declares current submitted set manually complete, 2026-09-26'})
        print(f'{key} {split} polygons={polygons} foreground={mask.mean():.3f}',flush=True)
        raster = vector = memory = target = None
    groups = {s:{r['split_group'] for r in rows if r['split']==s} for s in ('train','val','test')}
    assert not(groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
    save_json(out/'samples.json',rows)
    save_json(out/'audit.json', {'count':len(rows),'splits':{s:sum(r['split']==s for r in rows) for s in groups},
                               'groups':{s:sorted(g) for s,g in groups.items()},'repairs':repairs,'errors':errors,
                               'geometry_boundary_rule':'rasterize within unchanged original PNG extent; pixel centres',
                               'test_is_locked':True})
    for page in range((len(rows)+11)//12):
        sheet=Image.new('RGB',(1536,2460),'white')
        for i,r in enumerate(rows[page*12:(page+1)*12]):
            with Image.open(out/'qa'/f"{r['id']}.jpg") as p:
                sheet.paste(p,((i%2)*768,(i//2)*410))
        sheet.save(out/'qa'/f'contact_{page+1:02}.jpg',quality=90)
    if errors:
        raise RuntimeError('Dataset errors; do not train until resolved. See audit.json')
    print(json.dumps({'prepared':len(rows),'repairs':len(repairs),'errors':errors}),flush=True)


if __name__ == '__main__':
    main()
