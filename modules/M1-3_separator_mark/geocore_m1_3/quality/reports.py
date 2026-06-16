from __future__ import annotations

from geocore_m1_3.segments.core_segment import CoreSegment


def build_quality_report(
    lane_count_expected: int,
    lane_count_detected: int,
    refine_report: dict,
    strip_mapping: dict,
    depth_mapping: dict,
    segments: list[CoreSegment],
    warnings: list[dict],
) -> dict:
    low_coverage_count = sum(1 for segment in segments if "low_core_coverage" in segment.quality_flags)
    missing_count = sum(1 for segment in segments if "contains_missing_interval" in segment.quality_flags)
    return {
        "lane_count_expected": lane_count_expected,
        "lane_count_detected": lane_count_detected,
        "removed_component_count": refine_report.get("removed_component_count", 0),
        "removed_area_ratio": refine_report.get("removed_area_ratio", 0.0),
        "strip_pixel_length": strip_mapping.get("strip_height", 0),
        "strip_pixel_width": strip_mapping.get("strip_width", 0),
        "depth_span_m": depth_mapping["depth_end_m"] - depth_mapping["depth_start_m"],
        "meters_per_pixel": depth_mapping["meters_per_pixel"],
        "segment_count": len(segments),
        "low_coverage_segment_count": low_coverage_count,
        "segments_with_missing_interval_count": missing_count,
        "missing_interval_count": len(depth_mapping.get("missing_intervals", [])),
        "warnings": warnings,
    }
