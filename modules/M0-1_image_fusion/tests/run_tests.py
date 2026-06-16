from __future__ import annotations

import json
from pathlib import Path

from geocore_m01_fusion.cli import make_demo_inputs
from geocore_m01_fusion.config import EnviInput, FusionConfig, FusionMode
from geocore_m01_fusion.envi import write_envi
from geocore_m01_fusion.output import read_zarr_v2_uncompressed
from geocore_m01_fusion.pipeline import fuse_arrays, fuse_envi_files
from geocore_m01_fusion.api_service import create_tie_point_session_from_arrays, update_tie_point_session
from geocore_m01_fusion.tiepoints import build_warp_model_from_session, read_session


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test_outputs"


def test_all_modes() -> None:
    rgb, nir, swir, nir_w, swir_w = make_demo_inputs(height=48, width=64, nir_shape=(12, 16), swir_shape=(13, 16))
    expected_bands = nir.shape[2] + swir.shape[2]
    for mode in FusionMode:
        out_dir = OUT / mode.value
        cfg = FusionConfig(
            mode=mode,
            chunk_size=(16, 16, 4),
            rank=5,
            max_basis_pixels=1000,
            deep_iterations=2,
            write_previews=True,
        )
        result = fuse_arrays(
            rgb,
            nir,
            swir,
            output_dir=out_dir,
            nir_wavelengths=nir_w,
            swir_wavelengths=swir_w,
            config=cfg,
            input_metadata={"source": "unit_test"},
        )
        assert result.product.cube.shape == (48, 64, expected_bands)
        assert result.manifest["algorithm_mode"] == mode.value
        assert (out_dir / "manifest.json").exists()
        assert (out_dir / "metadata" / "band_metadata.csv").exists()
        assert (out_dir / "metrics" / "quality_report.json").exists()
        assert (out_dir / "previews" / "preview_rgb.png").exists()
        stored = read_zarr_v2_uncompressed(out_dir / "fused_cube.zarr")
        assert stored.shape == result.product.cube.shape
        assert stored.dtype.name == "float32"
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["cube"]["shape"] == [48, 64, expected_bands]
        assert manifest["metadata"]["band_metadata"] == "metadata/band_metadata.csv"


def test_envi_entrypoint() -> None:
    rgb, nir, swir, nir_w, swir_w = make_demo_inputs(height=32, width=40, nir_shape=(8, 10), swir_shape=(9, 10))
    envi_dir = OUT / "envi_inputs"
    envi_dir.mkdir(parents=True, exist_ok=True)
    rgb_hdr, rgb_dat = write_envi(rgb, envi_dir / "rgb.hdr", envi_dir / "rgb.dat")
    nir_hdr, nir_dat = write_envi(nir, envi_dir / "nir.hdr", envi_dir / "nir.dat", wavelengths=nir_w)
    swir_hdr, swir_dat = write_envi(swir, envi_dir / "swir.hdr", envi_dir / "swir.dat", wavelengths=swir_w)
    cfg = FusionConfig(mode=FusionMode.UPSAMPLE_ONLY, chunk_size=(16, 16, 4), write_previews=False)
    result = fuse_envi_files(
        EnviInput(rgb_hdr, rgb_dat),
        EnviInput(nir_hdr, nir_dat),
        EnviInput(swir_hdr, swir_dat),
        output_dir=OUT / "envi_fusion",
        config=cfg,
        mmap=True,
    )
    assert result.product.cube.shape[:2] == rgb.shape[:2]
    assert result.product.cube.shape[2] == nir.shape[2] + swir.shape[2]
    assert (OUT / "envi_fusion" / "manifest.json").exists()


def test_streaming_entrypoint_all_modes() -> None:
    rgb, nir, swir, nir_w, swir_w = make_demo_inputs(height=40, width=48, nir_shape=(10, 12), swir_shape=(11, 12))
    envi_dir = OUT / "streaming_inputs"
    envi_dir.mkdir(parents=True, exist_ok=True)
    rgb_hdr, rgb_dat = write_envi(rgb, envi_dir / "rgb.hdr", envi_dir / "rgb.dat")
    nir_hdr, nir_dat = write_envi(nir, envi_dir / "nir.hdr", envi_dir / "nir.dat", wavelengths=nir_w)
    swir_hdr, swir_dat = write_envi(swir, envi_dir / "swir.hdr", envi_dir / "swir.dat", wavelengths=swir_w)
    expected_bands = nir.shape[2] + swir.shape[2]
    for mode in FusionMode:
        cfg = FusionConfig(
            mode=mode,
            streaming=True,
            chunk_size=(16, 16, 4),
            rank=5,
            max_basis_pixels=500,
            deep_iterations=2,
            write_previews=True,
        )
        out_dir = OUT / f"streaming_{mode.value}"
        result = fuse_envi_files(
            EnviInput(rgb_hdr, rgb_dat),
            EnviInput(nir_hdr, nir_dat),
            EnviInput(swir_hdr, swir_dat),
            output_dir=out_dir,
            config=cfg,
            mmap=True,
        )
        assert result.product is None
        assert result.manifest["backend"] == "streaming"
        assert result.manifest["algorithm_mode"] == mode.value
        assert result.manifest["cube"]["shape"] == [40, 48, expected_bands]
        assert (out_dir / "fused_cube.zarr" / ".zarray").exists()
        assert (out_dir / "metadata" / "band_metadata.csv").exists()
        assert (out_dir / "metrics" / "quality_report.json").exists()
        stored = read_zarr_v2_uncompressed(out_dir / "fused_cube.zarr")
        assert stored.shape == (40, 48, expected_bands)


def test_interactive_tie_point_session() -> None:
    rgb, nir, _, _, _ = make_demo_inputs(height=64, width=64, nir_shape=(32, 32), swir_shape=(32, 32))
    reference = rgb[:, :, 1].astype("float32") / 255.0
    moving = nir[:, :, 2]
    out_dir = OUT / "tiepoints_session"
    result = create_tie_point_session_from_arrays(
        reference,
        moving,
        stage="unit_test_stage",
        output_dir=out_dir,
        reference_name="demo_rgb_green",
        moving_name="demo_nir_band",
        registration_params={
            "grid_rows": 4,
            "grid_cols": 4,
            "template_radius": 4,
            "search_radius_y": 4,
            "search_radius_x": 4,
            "min_tie_points": 4,
        },
        render_preview=True,
    )
    session_path = Path(result["session_path"])
    assert session_path.exists()
    assert Path(result["preview_paths"]["side_by_side"]).exists()
    first_id = result["session"]["points"][0]["id"]
    edit_result = update_tie_point_session(
        session_path,
        [
            {"op": "reject", "id": first_id},
            {"op": "add", "ref_y": 20, "ref_x": 20, "moving_y": 20, "moving_x": 20, "score": 1.0},
        ],
        render_preview=False,
    )
    assert edit_result["quality"]["manual_count"] == 1
    session = read_session(session_path)
    model = build_warp_model_from_session(session, min_active_points=3)
    assert model.to_dict()["tie_point_count"] >= 3


def main() -> int:
    test_all_modes()
    test_envi_entrypoint()
    test_streaming_entrypoint_all_modes()
    test_interactive_tie_point_session()
    print("All M0-1 fusion tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
