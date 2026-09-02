from __future__ import annotations

import inspect
from typing import Callable, Dict

from geocore_mask.models.core_unet import CoreMaskUNet, UNet


def _load_registry() -> Dict[str, Callable]:
    return {
        "CoreMaskUNet": CoreMaskUNet,
        "UNet": UNet,
    }


def available_models() -> list[str]:
    return sorted(_load_registry().keys())


def get_model_class(name: str) -> Callable:
    registry = _load_registry()
    try:
        return registry[name]
    except KeyError as exc:
        supported = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported model '{name}'. Supported models: {supported}") from exc


def build_model(name: str, input_channels: int, num_classes: int):
    model_cls = get_model_class(name)
    signature = inspect.signature(model_cls)
    kwargs = {}
    if "band_num" in signature.parameters:
        kwargs["band_num"] = input_channels
    if "num_classes" in signature.parameters:
        kwargs["num_classes"] = num_classes
    if kwargs:
        return model_cls(**kwargs)
    return model_cls(band_num=input_channels, num_classes=num_classes)
