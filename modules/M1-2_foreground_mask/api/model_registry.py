from typing import Callable, Dict

import predict


MODEL_REGISTRY: Dict[str, Callable] = {
    "DLinkNet34": predict.DLinkNet34,
    "DLinkNet50": predict.DLinkNet50,
    "DLinkNet101": predict.DLinkNet101,
    "UNet_Light": predict.UNet_Light,
    "UNet": predict.UNet,
    "DUNet": predict.DUNet,
    "DeepLabv3_plus": predict.DeepLabv3_plus,
    "FCN8S": predict.FCN8S,
    "DABNet": predict.DABNet,
    "Segformer": predict.Segformer,
    "RS_Segformer": predict.RS_Segformer,
    "DE_Segformer": predict.DE_Segformer,
    "HRNet": predict.HRNet,
    "UNetFormer": predict.UNetFormer,
    "FCN_ResNet50": predict.FCN_ResNet50,
    "FCN_ResNet101": predict.FCN_ResNet101,
    "U_MobileNet": predict.U_MobileNet,
    "SegNet": predict.SegNet,
    "U_ConvNeXt": predict.U_ConvNeXt,
    "GeoAwareUNet": predict.GeoAwareUNet,
    "GeoUNet": predict.GeoUNet,
    "UNetWithElevationAttention": predict.UNetWithElevationAttention,
    "MAEViTSegmentation": predict.MAEViTSegmentation,
}


def get_model(name: str) -> Callable:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model_name '{name}'. Supported models: {supported}") from exc
