from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
REQUIRED_SHP_SIDECARS = (".shp", ".shx", ".dbf")


@dataclass
class PreparedSample:
    image_path: Path
    mask_path: Path
    shp_path: Path
    overlay_path: Path | None
    foreground_ratio: float
    polygon_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build core foreground mask training samples from ArcGIS shapefiles.",
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root, e.g. datasets/core_mask_v2.")
    parser.add_argument("--image-dir", default="images", help="Image folder relative to dataset root.")
    parser.add_argument("--shp-dir", default="shp", help="Shapefile folder relative to dataset root.")
    parser.add_argument("--mask-dir", default="masks", help="Output mask folder relative to dataset root.")
    parser.add_argument("--overlay-dir", default="overlays", help="Output overlay folder relative to dataset root.")
    parser.add_argument("--splits-dir", default="splits", help="Output split folder relative to dataset root.")
    parser.add_argument(
        "--image-pattern",
        action="append",
        default=None,
        help="Image glob pattern. Can be repeated. Defaults to common image suffixes.",
    )
    parser.add_argument("--allow-missing-shp", action="store_true", help="Skip images that do not have a same-name .shp.")
    parser.add_argument("--skip-overlays", action="store_true", help="Do not write visual QA overlay PNGs.")
    parser.add_argument("--all-touched", action="store_true", help="Rasterize all pixels touched by polygon boundaries.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated masks/overlays/config files.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-config", default="configs/train_core_mask.example.json")
    parser.add_argument("--config-out", default=None, help="Optional training config JSON to write.")
    parser.add_argument("--experiment-name", default=None, help="Experiment name for generated config.")
    parser.add_argument("--image-size", type=int, default=None, help="Override image_size in generated config.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size in generated config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs in generated config.")
    return parser.parse_args()


def resolve_path(path_text: str | None, *, base: Path = ROOT) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def collect_images(image_dir: Path, patterns: list[str] | None) -> list[Path]:
    if patterns:
        images: list[Path] = []
        for pattern in patterns:
            images.extend(image_dir.glob(pattern))
    else:
        images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted({p.resolve() for p in images})


def require_gdal():
    try:
        from osgeo import gdal, ogr
    except Exception as exc:
        raise RuntimeError(
            "GDAL/OGR is required to rasterize ArcGIS shapefiles. "
            "Install it with: conda install -c conda-forge gdal"
        ) from exc
    return gdal, ogr


def sidecar_warnings(shp_path: Path) -> list[str]:
    warnings: list[str] = []
    for suffix in REQUIRED_SHP_SIDECARS:
        sidecar = shp_path.with_suffix(suffix)
        if not sidecar.exists():
            warnings.append(f"missing shapefile sidecar: {sidecar}")
    return warnings


def layer_feature_count(layer) -> int:
    count = layer.GetFeatureCount()
    return int(count) if count is not None and count >= 0 else 0


def pixel_space_geotransform(layer, width: int, height: int) -> tuple[float, float, float, float, float, float] | None:
    extent = layer.GetExtent()
    if not extent:
        return None
    min_x, max_x, min_y, max_y = extent
    tol_x = max(2.0, width * 0.02)
    tol_y = max(2.0, height * 0.02)

    x_matches = -tol_x <= min_x <= width + tol_x and -tol_x <= max_x <= width + tol_x
    y_down_matches = -tol_y <= min_y <= height + tol_y and -tol_y <= max_y <= height + tol_y
    y_up_matches = -height - tol_y <= min_y <= tol_y and -height - tol_y <= max_y <= tol_y

    if x_matches and y_down_matches:
        return (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if x_matches and y_up_matches:
        return (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    return None


def rasterize_shapefile(
    image_path: Path,
    shp_path: Path,
    *,
    all_touched: bool = False,
) -> tuple[np.ndarray, int]:
    import numpy as np

    from geocore_mask.utils.image_io import read_image

    gdal, ogr = require_gdal()

    image = read_image(image_path)
    height, width = image.array.shape[:2]
    vector_ds = ogr.Open(str(shp_path))
    if vector_ds is None:
        raise RuntimeError(f"cannot open shapefile: {shp_path}")

    layer = vector_ds.GetLayer(0)
    if layer is None:
        raise RuntimeError(f"shapefile has no readable layer: {shp_path}")
    polygon_count = layer_feature_count(layer)

    target_ds = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
    if image.geotransform:
        target_ds.SetGeoTransform(image.geotransform)
    else:
        inferred_gt = pixel_space_geotransform(layer, width, height)
        if inferred_gt is None:
            raise RuntimeError(
                f"{image_path.name} has no geotransform and {shp_path.name} does not look like pixel-space annotation. "
                "Open the image and shapefile in ArcGIS and ensure they overlap, or provide georeferenced images."
            )
        target_ds.SetGeoTransform(inferred_gt)

    if image.projection:
        target_ds.SetProjection(image.projection)

    options = [f"ALL_TOUCHED={'TRUE' if all_touched else 'FALSE'}"]
    err = gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[1], options=options)
    if err != 0:
        raise RuntimeError(f"GDAL rasterization failed for {shp_path}")

    mask = target_ds.GetRasterBand(1).ReadAsArray().astype(bool)
    vector_ds = None
    target_ds = None
    return mask, polygon_count


def write_split(path: Path, samples: list[PreparedSample], dataset_root: Path) -> None:
    lines = []
    for sample in samples:
        image_rel = sample.image_path.relative_to(dataset_root).as_posix()
        mask_rel = sample.mask_path.relative_to(dataset_root).as_posix()
        lines.append(f"{image_rel} {mask_rel}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def split_samples(
    samples: list[PreparedSample],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[PreparedSample], list[PreparedSample], list[PreparedSample]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("train/val/test ratios must sum to a positive number.")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_n = int(round(n * train_ratio / total_ratio))
    val_n = int(round(n * val_ratio / total_ratio))
    if train_n + val_n > n:
        val_n = max(0, n - train_n)

    train = shuffled[:train_n]
    val = shuffled[train_n : train_n + val_n]
    test = shuffled[train_n + val_n :]
    return train, val, test


def write_training_config(
    *,
    base_config_path: Path,
    output_path: Path,
    dataset_root: Path,
    splits_dir: Path,
    experiment_name: str | None,
    image_size: int | None,
    batch_size: int | None,
    epochs: int | None,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[skip] config exists: {output_path}")
        return

    config = json.loads(base_config_path.read_text(encoding="utf-8"))
    if experiment_name:
        config["experiment_name"] = experiment_name
        config.setdefault("output", {})["work_dir"] = f"runs/{experiment_name}"

    def rel(path: Path) -> str:
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return str(path)

    config.setdefault("data", {})["train_list"] = rel(splits_dir / "train.txt")
    config.setdefault("data", {})["val_list"] = rel(splits_dir / "val.txt")
    config.setdefault("data", {})["test_list"] = rel(splits_dir / "test.txt")
    config["data"]["dataset_root"] = rel(dataset_root)

    if image_size is not None:
        config["data"]["image_size"] = image_size
    if batch_size is not None:
        config.setdefault("training", {})["batch_size"] = batch_size
    if epochs is not None:
        config.setdefault("training", {})["epochs"] = epochs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote config: {output_path}")


def prepare_dataset(args: argparse.Namespace) -> int:
    import numpy as np

    from geocore_mask.utils.image_io import read_image, save_mask_png, save_overlay_png

    dataset_root = resolve_path(args.dataset_root)
    assert dataset_root is not None
    image_dir = dataset_root / args.image_dir
    shp_dir = dataset_root / args.shp_dir
    mask_dir = dataset_root / args.mask_dir
    overlay_dir = dataset_root / args.overlay_dir
    splits_dir = dataset_root / args.splits_dir

    if not image_dir.exists():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")
    if not shp_dir.exists():
        raise FileNotFoundError(f"shapefile directory does not exist: {shp_dir}")

    mask_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(image_dir, args.image_pattern)
    if not images:
        raise RuntimeError(f"no images found in {image_dir}")

    prepared: list[PreparedSample] = []
    warnings: list[str] = []
    missing: list[Path] = []

    for image_path in images:
        shp_path = shp_dir / f"{image_path.stem}.shp"
        if not shp_path.exists():
            missing.append(shp_path)
            if args.allow_missing_shp:
                print(f"[skip] missing shapefile for {image_path.name}")
                continue
            continue

        for warning in sidecar_warnings(shp_path):
            warnings.append(warning)

        mask_path = mask_dir / f"{image_path.stem}.png"
        overlay_path = None if args.skip_overlays else overlay_dir / f"{image_path.stem}_overlay.png"

        if mask_path.exists() and not args.overwrite:
            print(f"[skip] mask exists: {mask_path}")
            mask_image = read_image(mask_path).array
            mask_bool = np.squeeze(mask_image).astype(np.uint8) > 0
            polygon_count = -1
        else:
            mask_bool, polygon_count = rasterize_shapefile(image_path, shp_path, all_touched=args.all_touched)
            save_mask_png(mask_bool, mask_path)
            print(f"[ok] wrote mask: {mask_path}")

        if overlay_path is not None:
            if overlay_path.exists() and not args.overwrite:
                print(f"[skip] overlay exists: {overlay_path}")
            else:
                image = read_image(image_path)
                save_overlay_png(image.array, mask_bool, overlay_path)
                print(f"[ok] wrote overlay: {overlay_path}")

        foreground_ratio = float(mask_bool.mean())
        if foreground_ratio <= 0:
            warnings.append(f"empty mask: {mask_path}")
        elif foreground_ratio > 0.95:
            warnings.append(f"very large foreground ratio ({foreground_ratio:.3f}): {mask_path}")

        prepared.append(
            PreparedSample(
                image_path=image_path,
                mask_path=mask_path,
                shp_path=shp_path,
                overlay_path=overlay_path,
                foreground_ratio=foreground_ratio,
                polygon_count=polygon_count,
            )
        )

    if missing and not args.allow_missing_shp:
        missing_text = "\n".join(str(path) for path in missing[:20])
        extra = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise RuntimeError(f"missing shapefiles:\n{missing_text}{extra}")

    if not prepared:
        raise RuntimeError("no samples were prepared.")

    train, val, test = split_samples(
        prepared,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    write_split(splits_dir / "train.txt", train, dataset_root)
    write_split(splits_dir / "val.txt", val, dataset_root)
    write_split(splits_dir / "test.txt", test, dataset_root)
    print(f"[ok] wrote splits: train={len(train)}, val={len(val)}, test={len(test)}")

    report = {
        "dataset_root": str(dataset_root),
        "sample_count": len(prepared),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "samples": [
            {
                "image": sample.image_path.relative_to(dataset_root).as_posix(),
                "shp": sample.shp_path.relative_to(dataset_root).as_posix(),
                "mask": sample.mask_path.relative_to(dataset_root).as_posix(),
                "overlay": sample.overlay_path.relative_to(dataset_root).as_posix() if sample.overlay_path else None,
                "foreground_ratio": sample.foreground_ratio,
                "polygon_count": sample.polygon_count,
            }
            for sample in prepared
        ],
        "warnings": warnings,
    }
    report_path = dataset_root / "prepare_report.json"
    if not report_path.exists() or args.overwrite:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[ok] wrote report: {report_path}")
    else:
        print(f"[skip] report exists: {report_path}")

    if args.config_out:
        base_config = resolve_path(args.base_config)
        config_out = resolve_path(args.config_out)
        assert base_config is not None and config_out is not None
        write_training_config(
            base_config_path=base_config,
            output_path=config_out,
            dataset_root=dataset_root,
            splits_dir=splits_dir,
            experiment_name=args.experiment_name,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            overwrite=args.overwrite,
        )

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    return 0


def main() -> int:
    args = parse_args()
    try:
        return prepare_dataset(args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
