"""Shared bounded-window RGB inference for validation and production rasters."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def positions(length, size, step):
    if length <= size:
        return [0]
    values = list(range(0, length - size + 1, step))
    if values[-1] != length - size:
        values.append(length - size)
    return values


def foreground_probability(output, activation):
    if isinstance(output, (tuple, list)):
        output = output[0]
    if activation == 'sigmoid':
        return output[:, 1:2] if output.shape[1] > 1 else output
    if activation == 'logits':
        return torch.sigmoid(output) if output.shape[1] == 1 else torch.softmax(output, dim=1)[:, 1:2]
    if activation == 'legacy_softmax':
        return torch.softmax(output, dim=1)[:, 1:2]
    raise ValueError('Unsupported output activation: ' + activation)


@torch.inference_mode()
def predict_window(model, patch, manifest, device):
    cfg = manifest['input']
    size = int(cfg.get('image_size', 512))
    height, width = patch.shape[:2]
    rgb = np.asarray(Image.fromarray(patch).resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    mean = np.asarray(cfg['mean'], dtype=np.float32)
    std = np.asarray(cfg['std'], dtype=np.float32)
    if (std <= 0).any():
        raise ValueError('Normalization standard deviation must be positive')
    tensor = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1).copy()[None]).to(device)
    amp = str(device).startswith('cuda') and manifest.get('inference', {}).get('amp', False)
    with torch.autocast(device_type='cuda', enabled=amp):
        output = model(tensor)
    prob = foreground_probability(output, manifest.get('output_activation', 'sigmoid')).float()
    return F.interpolate(prob, size=(height, width), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()


def iter_probability_stripes(model, read_window, width, height, manifest, device):
    """read_window(x,y,w,h) -> uint8 HWC RGB; yield non-overlapping y, probability.

    Only a source-window-height accumulator is retained. Shared with ndarray
    validation to avoid a separate deployment resize/overlap contract.
    """
    cfg = manifest.get('inference', {})
    size = min(int(cfg.get('source_tile_size', 2048)), width, height)
    overlap = float(cfg.get('overlap', 0.25))
    if size <= 0 or not 0 <= overlap < 0.8:
        raise ValueError('Invalid source tile size or overlap')
    step = max(1, int(size * (1 - overlap)))
    xs, ys = positions(width, size, step), positions(height, size, step)
    # A floor prevents the tiny-Hann denominator bug at the outermost pixels.
    axis = np.maximum(np.hanning(size), 0.05).astype(np.float32)
    weight = np.outer(axis, axis)
    accum = np.zeros((size, width), np.float32)
    denom = np.zeros_like(accum)
    model.eval()
    for i, y in enumerate(ys):
        for x in xs:
            patch = read_window(x, y, size, size)
            if patch.dtype != np.uint8 or patch.shape != (size, size, 3):
                raise ValueError('Input window must be uint8 RGB with requested dimensions')
            prob = predict_window(model, patch, manifest, device)
            accum[:, x:x+size] += prob * weight
            denom[:, x:x+size] += weight
        end = ys[i+1] if i+1 < len(ys) else height
        count = end - y
        yield y, accum[:count] / np.maximum(denom[:count], 1e-12)
        if count < size:
            accum[:size-count] = accum[count:].copy()
            denom[:size-count] = denom[count:].copy()
        accum[size-count:] = 0
        denom[size-count:] = 0


def predict_array(model, image, manifest, device):
    height, width = image.shape[:2]
    result = np.empty((height, width), dtype=np.float32)
    read = lambda x, y, w, h: image[y:y+h, x:x+w]
    for y, stripe in iter_probability_stripes(model, read, width, height, manifest, device):
        result[y:y+len(stripe)] = stripe
    return result
