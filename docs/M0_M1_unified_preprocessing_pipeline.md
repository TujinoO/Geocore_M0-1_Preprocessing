# M0/M1 Unified Preprocessing Pipeline

## Goal

This workspace provides a single data-transfer pipeline for:

1. `M0-1` hyperspectral-optical image fusion.
2. `M1-1` core box rotation correction.
3. `M1-2` core foreground mask extraction.
4. `M1-3` core column separation and depth marking.

The pipeline is manifest-driven. M1 modules do not hard-code M0 output files;
they read `manifest.json` and write a unified `preprocess_context.json`.

## Data Flow

```text
fusion_result/manifest.json
  | resolves preview_rgb, fused_cube.zarr, band_metadata.csv, registration_model.json
  v
m1_1_rotation/
  | writes corrected_boxes, masks, metadata.json
  v
m1_2_foreground_mask/<box_id>/
  | writes mask.png, mask.tif, probability.tif, overlay.png, metadata.json
  v
m1_3_separator_mark/<core_box_id>/
  | writes reconstructed strip, depth mapping, segments.json
  v
segments_with_cube_refs.json
```

## Coordinate Contract

- M0 fusion grid: `rgb_reference`, defined by `manifest.grid`.
- M1 working grid: the image used by M1-1, normally `manifest.previews.preview_rgb`.
- M1-1 corrected box grid: one local grid per corrected core box.
- M1-3 reconstructed strip grid: the linearized depth strip used for segment cutting.

`m1_transform_stack.json` records how M1-1 crop/rotation outputs relate to the
M1 working grid and the M0 RGB reference grid. Segment records keep exact
`source_strip_bbox` coordinates and parent `bbox_rgb_reference_parent_box`
references for downstream Zarr reads.

## Output Contract

Each final segment in `segments_with_cube_refs.json` contains:

- `segment_id`, depth range, image and mask path from M1-3.
- `cube_ref.manifest_path`
- `cube_ref.cube_path`
- `cube_ref.band_metadata`
- `transform_stack_ref`
- `source_refs.m1_1_corrected_image`
- `source_refs.m1_2_mask_path`
- `bbox_rgb_reference_parent_box`

Downstream M2 modules should use this reference object instead of copying large
high-spectral cubes.

## M1-2 Engine Modes

- `model`: require a trained M1-2 model package and fail if unavailable.
- `auto`: try the trained model, then fall back to a classical mask.
- `classical`: always use the lightweight fallback for handoff testing.

The classical fallback exists only to keep the pipeline and context contract
testable when model weights are absent. Production processing should use
`--m1-2-engine model`.

## CLI Examples

Inspect an M0 result:

```powershell
python -m geocore_preprocessing.cli inspect-manifest `
  --manifest D:\path\to\fusion_result\manifest.json
```

Run from an existing M0 result:

```powershell
python -m geocore_preprocessing.cli run `
  --manifest D:\path\to\fusion_result\manifest.json `
  --output-dir D:\path\to\preprocessing_run `
  --hole-id ZKH3 `
  --core-box-prefix ZKH3_132_140 `
  --depth-start-m 132.0 `
  --depth-end-m 140.0 `
  --m1-2-engine model `
  --m1-2-model-package E:\Code\Geocore_M0&1_Preprocessing\modules\M1-2_foreground_mask\models\core_mask_unet_v1
```

Use full-resolution M1-1 input when the M0 preview is only a display pyramid:

```powershell
python -m geocore_preprocessing.cli run `
  --manifest D:\path\to\fusion_result\manifest.json `
  --m1-1-input-image D:\path\to\RGB-20230909_141858-00000.dat `
  --m1-1-hdr-path D:\path\to\RGB-20230909_141858-00000.hdr `
  --output-dir D:\path\to\preprocessing_run `
  --depth-start-m 132.0 `
  --depth-end-m 140.0 `
  --m1-2-engine model `
  --m1-2-model-package E:\Code\Geocore_M0&1_Preprocessing\modules\M1-2_foreground_mask\models\core_mask_unet_v1
```
