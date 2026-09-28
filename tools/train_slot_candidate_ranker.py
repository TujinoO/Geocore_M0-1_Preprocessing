"""Train a small *advisory* RGB physical-slot hypothesis ranker.

The unit of cross-validation is the original ENVI acquisition, never a PNG
crop or a candidate hypothesis. The resulting JSON can be scored without a
scikit-learn runtime. This tool does not modify source annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = ["score", "score_gap_from_best", "weakest_separator", "span_fraction", "spacing_fraction", "count"]


def _features(sample: dict, candidate: dict) -> list[float]:
    all_candidates = sample["m13_baseline"]["candidate_rankings"]
    box = sample["box_bbox_xyxy"]
    width = max(1.0, box[2] - box[0])
    lines = candidate["separator_x"]
    span = (lines[-1] - lines[0]) / width
    return [
        float(candidate["score"]),
        float(candidate["score"] - all_candidates[0]["score"]),
        float(candidate["weakest_separator"]),
        float(span),
        float(span / candidate["count"]),
        float(candidate["count"]),
    ]


def _model() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=28))


def _predict_samples(model: object, rows: list[dict]) -> list[dict]:
    results = []
    for sample in rows:
        candidates = sample["m13_baseline"]["candidate_rankings"]
        probabilities = model.predict_proba(np.asarray([_features(sample, candidate) for candidate in candidates]))[:, 1]
        winner = candidates[int(np.argmax(probabilities))]
        results.append({
            "sample_id": sample["sample_id"], "source": sample["source"],
            "truth": sample["slot_count"], "baseline": sample["m13_baseline"]["candidate_count"],
            "ranker": winner["count"], "ranker_probability": round(float(max(probabilities)), 5),
            "ranker_correct": winner["count"] == sample["slot_count"],
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite model output: {args.output_dir}")
    samples = [sample for sample in json.loads(args.baseline.read_text(encoding="utf-8"))
               if sample.get("m13_baseline", {}).get("candidate_rankings")]
    groups = np.asarray([sample["source"] for sample in samples])
    rows = []
    for sample in samples:
        for candidate in sample["m13_baseline"]["candidate_rankings"]:
            rows.append((_features(sample, candidate), int(candidate["count"] == sample["slot_count"]), sample["source"]))
    X = np.asarray([row[0] for row in rows])
    y = np.asarray([row[1] for row in rows])
    group_rows = np.asarray([row[2] for row in rows])
    predicted = []
    for train_indexes, test_indexes in LeaveOneGroupOut().split(samples, groups=groups):
        train_sources = set(groups[train_indexes])
        test_sources = set(groups[test_indexes])
        train_mask = np.isin(group_rows, list(train_sources))
        model = _model()
        model.fit(X[train_mask], y[train_mask])
        fold = _predict_samples(model, [samples[i] for i in test_indexes])
        for entry in fold:
            entry["train_sources"] = sorted(train_sources)
            entry["heldout_source"] = sorted(test_sources)[0]
        predicted.extend(fold)
    model = _model()
    model.fit(X, y)
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    artifact = {
        "model_type": "slot_candidate_logistic_advisory_v1",
        "feature_names": FEATURE_NAMES,
        "feature_mean": scaler.mean_.tolist(),
        "feature_scale": scaler.scale_.tolist(),
        "coefficient": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "source_baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "sample_count": len(samples),
        "source_count": len(set(groups)),
        "deployment_status": "advisory_only_not_automatic_truth",
    }
    summary = {
        "cv_method": "leave_one_original_envi_source_out",
        "sample_count": len(samples), "source_count": len(set(groups)),
        "baseline_exact": sum(entry["baseline"] == entry["truth"] for entry in predicted),
        "ranker_exact": sum(entry["ranker_correct"] for entry in predicted),
        "predictions": sorted(predicted, key=lambda entry: entry["sample_id"]),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "slot_ranker.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "group_cv.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Grouped CV: baseline {summary['baseline_exact']}/{len(samples)}, ranker {summary['ranker_exact']}/{len(samples)}")


if __name__ == "__main__":
    main()
