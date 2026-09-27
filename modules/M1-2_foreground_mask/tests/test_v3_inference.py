import sys
import unittest
import tempfile
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geocore_mask.inference.scaled import foreground_probability, iter_probability_stripes, predict_array
from geocore_mask.inference.streaming import predict_raster, rgb_bands


class Constant(torch.nn.Module):
    def forward(self,x):
        return torch.ones((len(x),2,x.shape[2],x.shape[3]),device=x.device)*.8


class InferenceTests(unittest.TestCase):
    def test_gdal_stream_output_and_band_order(self):
        from osgeo import gdal
        # Retain generated fixtures; this workspace prohibits recursive cleanup.
        with nullcontext(tempfile.mkdtemp(prefix='core_mask_v3_test_')) as tmp:
            root=Path(tmp);path=root/'rgb.dat'
            ds=gdal.GetDriverByName('ENVI').Create(str(path),37,59,3,gdal.GDT_Byte)
            ds.SetGeoTransform((123,2,0,456,0,-2))
            for i in range(1,4): ds.GetRasterBand(i).Fill(i*10)
            ds=None
            with path.with_suffix('.hdr').open('a') as f:f.write('\ndefault bands = {2, 1, 0}\n')
            ds=gdal.Open(str(path));self.assertEqual(rgb_bands(ds,path),[3,2,1]);ds=None
            mf={'model_version':'test','input':{'image_size':16,'mean':[0,0,0],'std':[1,1,1]},
                'deployment_warning':'Candidate requires manual review.',
                'output_activation':'sigmoid','inference':{'source_tile_size':24,'overlap':.25,'threshold':.5}}
            result=predict_raster(Constant(),mf,'cpu',path,root/'out')
            ds=gdal.Open(result['output_files']['mask_tif'])
            self.assertEqual(ds.GetGeoTransform(),(123,2,0,456,0,-2))
            np.testing.assert_array_equal(ds.ReadAsArray(),np.full((59,37),255,np.uint8));ds=None
            ds=gdal.Open(result['output_files']['probability_tif'])
            np.testing.assert_allclose(ds.ReadAsArray(),.8,atol=1e-6);ds=None
            meta=json.loads((root/'out/metadata.json').read_text())
            self.assertEqual(meta['status'],'complete');self.assertEqual(meta['rows_written'],59)
            self.assertIn(mf['deployment_warning'],result['warnings'])
            self.assertIn(mf['deployment_warning'],meta['warnings'])

    def test_already_sigmoid_not_softmax(self):
        out=torch.tensor([[[[.1]],[[.9]]]])
        self.assertAlmostEqual(float(foreground_probability(out,'sigmoid')), .9, places=6)

    def test_streaming_covers_irregular_last_stride_and_edges(self):
        mf={'input':{'image_size':16,'mean':[0,0,0],'std':[1,1,1]},
            'output_activation':'sigmoid','inference':{'source_tile_size':24,'overlap':.25}}
        im=np.zeros((79,61,3),np.uint8)
        seen=np.zeros(im.shape[:2],int)
        for y,strip in iter_probability_stripes(Constant(),lambda x,y,w,h:im[y:y+h,x:x+w],61,79,mf,'cpu'):
            seen[y:y+len(strip)]+=1
            np.testing.assert_allclose(strip,.8,atol=1e-6)
        np.testing.assert_array_equal(seen,np.ones_like(seen))

    def test_small_image_not_dropped(self):
        mf={'input':{'image_size':16,'mean':[0,0,0],'std':[1,1,1]},
            'output_activation':'sigmoid','inference':{'source_tile_size':32,'overlap':.25}}
        out=predict_array(Constant(),np.zeros((7,11,3),np.uint8),mf,'cpu')
        self.assertEqual(out.shape,(7,11));np.testing.assert_allclose(out,.8,atol=1e-6)


if __name__=='__main__':unittest.main()
