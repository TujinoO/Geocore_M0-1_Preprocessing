"""Minimal ENVI .hdr/.dat reader and writer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ENVI_DTYPE_TO_NUMPY: dict[int, str] = {
    1: "u1",
    2: "i2",
    3: "i4",
    4: "f4",
    5: "f8",
    12: "u2",
    13: "u4",
    14: "i8",
    15: "u8",
}

NUMPY_DTYPE_TO_ENVI: dict[str, int] = {
    "uint8": 1,
    "int16": 2,
    "int32": 3,
    "float32": 4,
    "float64": 5,
    "uint16": 12,
    "uint32": 13,
    "int64": 14,
    "uint64": 15,
}


@dataclass(slots=True)
class EnviMetadata:
    samples: int
    lines: int
    bands: int
    data_type: int
    interleave: str
    byte_order: int = 0
    header_offset: int = 0
    wavelengths: list[float] | None = None
    raw: dict[str, Any] | None = None

    @property
    def shape_yxb(self) -> tuple[int, int, int]:
        return (self.lines, self.samples, self.bands)

    @property
    def dtype(self) -> np.dtype:
        base = ENVI_DTYPE_TO_NUMPY.get(self.data_type)
        if base is None:
            raise ValueError(f"Unsupported ENVI data type: {self.data_type}")
        byte_prefix = "<" if self.byte_order == 0 else ">"
        if base == "u1":
            return np.dtype(base)
        return np.dtype(byte_prefix + base)


def parse_envi_header(hdr_path: str | Path) -> EnviMetadata:
    """Parse enough ENVI header fields for this module."""

    path = Path(hdr_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.lstrip().lower().startswith("envi"):
        raise ValueError(f"{path} is not an ENVI header")

    fields = _parse_header_fields(text)
    required = ["samples", "lines", "bands", "data type", "interleave"]
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"Missing ENVI header fields in {path}: {missing}")

    wavelengths = None
    if "wavelength" in fields:
        wavelengths = _parse_float_list(fields["wavelength"])

    return EnviMetadata(
        samples=int(fields["samples"]),
        lines=int(fields["lines"]),
        bands=int(fields["bands"]),
        data_type=int(fields["data type"]),
        interleave=str(fields["interleave"]).strip().lower(),
        byte_order=int(fields.get("byte order", 0)),
        header_offset=int(fields.get("header offset", 0)),
        wavelengths=wavelengths,
        raw=fields,
    )


def infer_dat_path(hdr_path: str | Path) -> Path:
    """Infer the binary data path from an ENVI header path."""

    hdr = Path(hdr_path)
    if hdr.suffix.lower() == ".hdr":
        candidate = hdr.with_suffix(".dat")
        if candidate.exists():
            return candidate
        candidate = hdr.with_suffix("")
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot infer ENVI data file for {hdr}")


def read_envi(
    hdr_path: str | Path,
    dat_path: str | Path | None = None,
    *,
    mmap: bool = True,
) -> tuple[np.ndarray, EnviMetadata]:
    """Read an ENVI image as y, x, band.

    For memory-mapped reads, the returned array can be a non-contiguous view.
    """

    meta = parse_envi_header(hdr_path)
    data_path = Path(dat_path) if dat_path is not None else infer_dat_path(hdr_path)
    expected = meta.samples * meta.lines * meta.bands * meta.dtype.itemsize + meta.header_offset
    actual = data_path.stat().st_size
    if actual < expected:
        raise ValueError(
            f"ENVI data file is too small: expected at least {expected} bytes, got {actual}"
        )

    count = meta.samples * meta.lines * meta.bands
    if mmap:
        raw = np.memmap(
            data_path,
            dtype=meta.dtype,
            mode="r",
            offset=meta.header_offset,
            shape=(count,),
        )
    else:
        raw = np.fromfile(data_path, dtype=meta.dtype, count=count, offset=meta.header_offset)

    interleave = meta.interleave.lower()
    if interleave == "bil":
        arr = raw.reshape(meta.lines, meta.bands, meta.samples).transpose(0, 2, 1)
    elif interleave == "bip":
        arr = raw.reshape(meta.lines, meta.samples, meta.bands)
    elif interleave == "bsq":
        arr = raw.reshape(meta.bands, meta.lines, meta.samples).transpose(1, 2, 0)
    else:
        raise ValueError(f"Unsupported ENVI interleave: {meta.interleave}")
    return arr, meta


def write_envi(
    cube_yxb: np.ndarray,
    hdr_path: str | Path,
    dat_path: str | Path | None = None,
    *,
    wavelengths: list[float] | None = None,
    interleave: str = "bil",
    description: str = "Geo-Core M0-1 fusion output",
) -> tuple[Path, Path]:
    """Write an ENVI .hdr/.dat pair."""

    hdr = Path(hdr_path)
    dat = Path(dat_path) if dat_path is not None else hdr.with_suffix(".dat")
    hdr.parent.mkdir(parents=True, exist_ok=True)
    dat.parent.mkdir(parents=True, exist_ok=True)

    cube = np.asarray(cube_yxb)
    if cube.ndim != 3:
        raise ValueError("ENVI writer expects a y,x,band cube")
    dtype_name = str(cube.dtype)
    envi_type = NUMPY_DTYPE_TO_ENVI.get(dtype_name)
    if envi_type is None:
        cube = cube.astype(np.float32, copy=False)
        dtype_name = str(cube.dtype)
        envi_type = NUMPY_DTYPE_TO_ENVI[dtype_name]

    interleave = interleave.lower()
    if interleave == "bil":
        raw = cube.transpose(0, 2, 1)
    elif interleave == "bip":
        raw = cube
    elif interleave == "bsq":
        raw = cube.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported ENVI interleave: {interleave}")
    np.ascontiguousarray(raw).tofile(dat)

    lines, samples, bands = cube.shape
    wave_text = ""
    if wavelengths is not None:
        values = ", ".join(f"{float(w):.6g}" for w in wavelengths)
        wave_text = f"wavelength = {{{values}}}\n"
    hdr.write_text(
        "ENVI\n"
        f"description = {{{description}}}\n"
        f"samples = {samples}\n"
        f"lines = {lines}\n"
        f"bands = {bands}\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        f"data type = {envi_type}\n"
        f"interleave = {interleave}\n"
        "byte order = 0\n"
        f"{wave_text}",
        encoding="utf-8",
    )
    return hdr, dat


def _parse_header_fields(text: str) -> dict[str, str]:
    """Parse ENVI key-value fields, including brace blocks."""

    fields: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.lower() == "envi" or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if value.startswith("{") and not value.endswith("}"):
            block = [value]
            depth = value.count("{") - value.count("}")
            while i < len(lines) and depth > 0:
                nxt = lines[i].strip()
                block.append(nxt)
                depth += nxt.count("{") - nxt.count("}")
                i += 1
            value = "\n".join(block)
        if value.startswith("{") and value.endswith("}"):
            value = value[1:-1].strip()
        fields[key] = value
    return fields


def _parse_float_list(value: str) -> list[float]:
    tokens = re.split(r"[,\s]+", value.strip())
    out: list[float] = []
    for token in tokens:
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out
