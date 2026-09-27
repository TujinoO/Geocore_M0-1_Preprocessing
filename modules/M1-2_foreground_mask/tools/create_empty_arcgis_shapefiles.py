from __future__ import annotations

import argparse
from pathlib import Path


CORE_SIDECARS = (".shp", ".shx", ".dbf")
ALL_SIDECARS = CORE_SIDECARS + (".cpg",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one same-stem, empty ArcGIS Polygon Shapefile beside each PNG. "
            "Existing shapefiles are never overwritten."
        )
    )
    parser.add_argument("image_dirs", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verify-existing-empty",
        action="store_true",
        help="Also verify that every skipped complete shapefile is an empty Polygon layer.",
    )
    return parser.parse_args()


def require_ogr():
    try:
        from osgeo import ogr
    except Exception as exc:
        raise RuntimeError("GDAL/OGR is required to create shapefiles.") from exc
    return ogr


def target_paths(image_path: Path) -> tuple[Path, ...]:
    return tuple(image_path.with_suffix(suffix) for suffix in ALL_SIDECARS)


def validate_empty_polygon_shapefile(shp_path: Path, ogr) -> None:
    for required_path in tuple(shp_path.with_suffix(suffix) for suffix in ALL_SIDECARS):
        if not required_path.is_file() or required_path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty shapefile sidecar: {required_path}")
    dataset = ogr.Open(str(shp_path), 0)
    if dataset is None:
        raise RuntimeError(f"OGR cannot open generated shapefile: {shp_path}")
    layer = dataset.GetLayer(0)
    if layer is None:
        raise RuntimeError(f"Generated shapefile has no layer: {shp_path}")
    if ogr.GT_Flatten(layer.GetGeomType()) != ogr.wkbPolygon:
        raise RuntimeError(f"Generated layer is not Polygon: {shp_path}")
    if layer.GetFeatureCount() != 0:
        raise RuntimeError(f"Generated shapefile is not empty: {shp_path}")
    definition = layer.GetLayerDefn()
    if definition.GetFieldCount() != 1 or definition.GetFieldDefn(0).GetName() != "Id":
        raise RuntimeError(f"Generated shapefile schema does not match the ArcGIS examples: {shp_path}")
    dataset = None


def create_shapefile(image_path: Path, ogr) -> None:
    shp_path = image_path.with_suffix(".shp")
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if driver is None:
        raise RuntimeError("OGR ESRI Shapefile driver is unavailable.")

    dataset = driver.CreateDataSource(str(shp_path))
    if dataset is None:
        raise RuntimeError(f"Cannot create shapefile: {shp_path}")
    layer = dataset.CreateLayer(
        image_path.stem,
        srs=None,
        geom_type=ogr.wkbPolygon,
        options=["ENCODING=UTF-8"],
    )
    if layer is None:
        raise RuntimeError(f"Cannot create Polygon layer: {shp_path}")
    field = ogr.FieldDefn("Id", ogr.OFTInteger)
    field.SetWidth(6)
    if layer.CreateField(field) != 0:
        raise RuntimeError(f"Cannot create Id field: {shp_path}")
    dataset = None

    for required_path in target_paths(image_path):
        if not required_path.is_file() or required_path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty generated sidecar: {required_path}")
    validate_empty_polygon_shapefile(shp_path, ogr)


def main() -> int:
    args = parse_args()
    ogr = require_ogr()
    planned: list[Path] = []
    skipped: list[Path] = []

    for image_dir in args.image_dirs:
        image_dir = image_dir.resolve()
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")
        if not images:
            raise RuntimeError(f"No PNG images found: {image_dir}")

        for image_path in images:
            core_paths = target_paths(image_path)[:3]
            existing_core = [path for path in core_paths if path.exists()]
            if not existing_core:
                planned.append(image_path)
                continue
            if len(existing_core) != len(core_paths):
                names = ", ".join(str(path) for path in existing_core)
                raise RuntimeError(f"Partial shapefile already exists; refusing to overwrite: {names}")
            if args.verify_existing_empty:
                validate_empty_polygon_shapefile(image_path.with_suffix(".shp"), ogr)
            skipped.append(image_path)

    print(f"PNG images: {len(planned) + len(skipped)}")
    print(f"To create: {len(planned)}")
    print(f"Existing complete shapefiles skipped: {len(skipped)}")
    if args.verify_existing_empty:
        print(f"Existing empty shapefiles validated: {len(skipped)}")
    if args.dry_run:
        print("Dry run only; no files created.")
        return 0

    created = 0
    for image_path in planned:
        create_shapefile(image_path, ogr)
        created += 1
        if created % 100 == 0:
            print(f"Created and validated: {created}/{len(planned)}")

    print(f"Created and validated: {created}/{len(planned)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
