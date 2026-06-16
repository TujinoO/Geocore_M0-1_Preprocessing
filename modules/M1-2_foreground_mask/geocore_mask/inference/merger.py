from __future__ import annotations

import numpy as np


def cosine_weight(size: int) -> np.ndarray:
    axis = np.hanning(size)
    if np.all(axis == 0):
        axis = np.ones(size, dtype=np.float32)
    weight = np.outer(axis, axis).astype(np.float32)
    min_positive = weight[weight > 0].min() if np.any(weight > 0) else 1.0
    weight[weight == 0] = min_positive
    return weight / weight.max()


class ProbabilityMerger:
    def __init__(self, height: int, width: int, tile_size: int):
        self.prob_sum = np.zeros((height, width), dtype=np.float32)
        self.weight_sum = np.zeros((height, width), dtype=np.float32)
        self.weight = cosine_weight(tile_size)

    def add(self, probability: np.ndarray, x: int, y: int) -> None:
        h, w = probability.shape
        weight = self.weight[:h, :w]
        self.prob_sum[y : y + h, x : x + w] += probability * weight
        self.weight_sum[y : y + h, x : x + w] += weight

    def result(self) -> np.ndarray:
        return self.prob_sum / np.maximum(self.weight_sum, 1e-6)
