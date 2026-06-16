# Geo-Core AI M0-1 Fusion Module

This repository contains a runnable Python implementation of the M0-1
hyperspectral-optical image fusion module.

Implemented modes:

- `upsample_only`: geometric upsampling baseline.
- `fast_preview`: RGB-guided high-frequency preview fusion.
- `classical`: low-rank spectral basis plus RGB edge-guided coefficient fusion.
- `deep_unsupervised`: NumPy-based unsupervised unrolled autoencoder-style fusion.

The module writes the output contract described in the design document:

- `manifest.json`
- `fused_cube.zarr` as an uncompressed Zarr v2-compatible chunk store
- `metadata/band_metadata.csv`
- `metadata/spectral_stitch_model.json`
- `metrics/quality_report.json`
- preview PNG files

## Quick Demo

```powershell
python -m geocore_m01_fusion.cli --demo --mode classical --output runs/demo_classical
```

Run all smoke tests:

```powershell
python tests/run_tests.py
```

## Full-Size ENVI Streaming

For full-size core-box data, use `--streaming`. This path memory-maps the
original ENVI files and writes `fused_cube.zarr` tile by tile, so it does not
materialize the full fused cube in RAM.

```powershell
python -m geocore_m01_fusion.cli `
  --streaming `
  --mode classical `
  --rgb-hdr "E:\...\RGB-20230909_141858-00000.hdr" `
  --rgb-dat "E:\...\RGB-20230909_141858-00000.dat" `
  --nir-hdr "E:\...\NIR-20230909_141902-00000.hdr" `
  --nir-dat "E:\...\NIR-20230909_141902-00000.dat" `
  --swir-hdr "E:\...\SWIR-20230909_141858-00000.hdr" `
  --swir-dat "E:\...\SWIR-20230909_141858-00000.dat" `
  --output "runs\ZKH3_132_140_classical_streaming" `
  --chunk-size 256 256 32
```

Recommended first full-size order:

1. Run `upsample_only` to validate geometry, metadata, and output volume.
2. Run `fast_preview` for quick RGB-guided texture checks.
3. Run `classical` as the default production fusion.
4. Run `deep_unsupervised` on selected boxes or ROIs first because it is slower.

## Prepare A Registered Test ROI

Use the ROI command to build an ENVI-style registered small sample before
testing fusion modes. The current default follows a tie-points workflow:

- coarse scale mapping,
- SWIR-to-RGB local warp,
- NIR-to-SWIR tie point matching,
- inverse-distance local warp.

```powershell
python -m geocore_m01_fusion.roi_cli `
  --root "E:\Experiment_data\2026 岩心高光谱数据\...\2023_09_09_14_18_58-ZKH3号-132-140-0.0_0.0-0.0_0.0" `
  --output "roi_outputs\ZKH3_132_140_roi_768x512_tiepoints_warp" `
  --crop-height 768 `
  --crop-width 512
```

The output `roi_manifest.json` records generated tie points, rejected tie
points, and local warp statistics.
