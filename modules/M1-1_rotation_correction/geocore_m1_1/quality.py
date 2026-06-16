from __future__ import annotations

from pathlib import Path

from .models import M11BatchResult


def write_qa_report(result: M11BatchResult, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# M1-1 QA Report",
        "",
        f"- input: `{result.input_path}`",
        f"- hdr: `{result.hdr_path}`",
        f"- box_count: `{result.box_count}`",
        "",
        "| box_id | angle_deg | confidence | review | output |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for box in result.boxes:
        review = "yes" if box.needs_manual_review else "no"
        lines.append(
            f"| {box.box_id} | {box.angle_deg:.4f} | {box.confidence:.3f} | {review} | `{box.output_image}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)

