"""Experimental column-profile learner for physical slot dividers.

This does not replace M1-3. It tests whether 16 labelled boxes support a
scene-held-out RGB divider model before any deployment decision is made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier


def _unit(values: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(values, [10, 90])
    return np.clip((values - lo) / max(1e-6, hi - lo), 0, 1)


def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    return np.convolve(values, np.ones(radius * 2 + 1) / (radius * 2 + 1), mode="same")


def _profile(sample: dict, target_width: int = 512) -> tuple[np.ndarray, np.ndarray]:
    box = sample["box_bbox_xyxy"]
    with Image.open(Path(sample["image"])) as source:
        x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
        x1, y1 = min(source.width, int(box[2]) + 1), min(source.height, int(box[3]) + 1)
        crop = source.crop((x0, y0, x1, y1)).convert("RGB")
        crop.thumbnail((target_width, 1024), Image.Resampling.BOX)
        crop = crop.resize((target_width, min(1024, crop.height)), Image.Resampling.BOX)
        rgb = np.asarray(crop, dtype=np.float32)
    rgb = rgb[int(len(rgb) * 0.10):int(len(rgb) * 0.90)]
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    dx = np.zeros_like(gray)
    dx[:, 2:-2] = np.abs(gray[:, 4:] - gray[:, :-4])
    dy = np.abs(gray[4:] - gray[:-4])
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    coloured = (red > 95) & (red > 1.20 * green) & (green > 1.12 * blue) & (red - blue > 45)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    base = [
        _unit(np.percentile(dx, 55, axis=0)), _unit(np.percentile(dx, 80, axis=0)),
        _unit(dx.mean(axis=0)), _unit(coloured.mean(axis=0)),
        _unit(gray.mean(axis=0)), _unit(np.percentile(gray, 20, axis=0)),
        _unit(np.percentile(gray, 80, axis=0)), _unit(gray.std(axis=0)),
        _unit(dy.mean(axis=0)), _unit(saturation.mean(axis=0)),
    ]
    features = []
    for profile in base:
        for radius in (0, 3, 9):
            features.append(profile if radius == 0 else _smooth(profile, radius))
    truth_dividers = np.asarray(sample["m13_baseline"]["truth_dividers_x_in_box"])
    truth_scaled = truth_dividers / max(1, box[2] - box[0]) * target_width
    truth_labels = np.zeros(target_width, dtype=np.uint8)
    for x in truth_scaled:
        truth_labels[max(0, int(round(x)) - 4):min(target_width, int(round(x)) + 5)] = 1
    return np.stack(features, axis=1), truth_labels


def _model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=120, max_depth=8, min_samples_leaf=10,
                                class_weight="balanced", max_features=0.7, n_jobs=4, random_state=28)


def _choose(sample: dict, probabilities: np.ndarray) -> tuple[int, float]:
    candidates = sample["m13_baseline"]["candidate_rankings"]
    width = max(1, sample["box_bbox_xyxy"][2] - sample["box_bbox_xyxy"][0])
    scores = []
    for candidate in candidates:
        xs = np.asarray(candidate["separator_x"]) / width * len(probabilities)
        internal = xs[1:-1]
        near = [probabilities[max(0, int(round(x)) - 4):min(len(probabilities), int(round(x)) + 5)].max()
                for x in internal if 0 <= x < len(probabilities)]
        inside = (xs[:-1] + xs[1:]) / 2
        misses = [probabilities[max(0, int(round(x)) - 4):min(len(probabilities), int(round(x)) + 5)].max()
                  for x in inside if 0 <= x < len(probabilities)]
        score = (np.mean(near) if near else 0) - 0.30 * (np.mean(misses) if misses else 0)
        score += 0.20 * candidate["score"]
        scores.append(float(score))
    winner = int(np.argmax(scores))
    return int(candidates[winner]["count"]), scores[winner]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    samples = [sample for sample in json.loads(args.baseline.read_text(encoding="utf-8"))
               if sample.get("m13_baseline", {}).get("candidate_rankings")]
    for sample in samples:
        sample["image"] = str(args.queue / "images" / f"{sample['sample_id']}.png")
    profiles = [_profile(sample) for sample in samples]
    predictions = []
    for source in sorted({sample["source"] for sample in samples}):
        train = [i for i, sample in enumerate(samples) if sample["source"] != source]
        test = [i for i, sample in enumerate(samples) if sample["source"] == source]
        model = _model()
        model.fit(np.concatenate([profiles[i][0] for i in train]),
                  np.concatenate([profiles[i][1] for i in train]))
        for i in test:
            probabilities = model.predict_proba(profiles[i][0])[:, 1]
            chosen, score = _choose(samples[i], probabilities)
            record = {"sample_id": samples[i]["sample_id"], "source": source,
                      "truth": samples[i]["slot_count"],
                      "baseline": samples[i]["m13_baseline"]["candidate_count"],
                      "profile_model": chosen, "score": score,
                      "positive_column_probability_max": float(probabilities.max())}
            predictions.append(record)
            print(record["sample_id"], "truth", record["truth"], "baseline", record["baseline"],
                  "profile", chosen, flush=True)
    result = {"cv_method": "leave_one_original_envi_source_out", "sample_count": len(samples),
              "baseline_exact": sum(row["baseline"] == row["truth"] for row in predictions),
              "profile_model_exact": sum(row["profile_model"] == row["truth"] for row in predictions),
              "predictions": sorted(predictions, key=lambda row: row["sample_id"]),
              "deployment_status": "research_experiment_only"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Grouped CV:", result["baseline_exact"], result["profile_model_exact"], "/", len(samples))


if __name__ == "__main__":
    main()
