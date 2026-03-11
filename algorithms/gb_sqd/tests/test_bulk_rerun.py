from __future__ import annotations

import json
from pathlib import Path

from gb_sqd import bulk_rerun


def test_build_failed_target_overrides_applies_override_only_to_failed_targets():
    summary = {
        "results": [
            {"relative_path": "case_a/atom_1", "status": "success"},
            {"relative_path": "case_b/atom_2", "status": "failed"},
            {"relative_path": "case_c/atom_3", "status": "failed"},
        ]
    }

    result = bulk_rerun.build_failed_target_overrides(
        summary=summary,
        override_parameters={"carryover_threshold": 1.0e-3},
        base_target_overrides={"case_c/atom_3": {"max_time": 1800}},
    )

    assert result == {
        "case_c/atom_3": {
            "max_time": 1800,
            "carryover_threshold": 1.0e-3,
        },
        "case_b/atom_2": {
            "carryover_threshold": 1.0e-3,
        },
    }


def test_bulk_rerun_wrapper_returns_without_rerun_when_initial_run_succeeds(monkeypatch):
    calls: list[dict] = []

    def fake_bulk_gb_sqd_flow(**kwargs):
        calls.append(kwargs)
        return {"failed_targets": 0, "results": []}

    monkeypatch.setattr(bulk_rerun, "bulk_gb_sqd_flow", fake_bulk_gb_sqd_flow)

    result = bulk_rerun.bulk_gb_sqd_flow_with_failed_target_rerun(
        mode="ext_sqd",
        input_root_dir="/tmp/input",
        output_root_dir="/tmp/output",
        failed_target_override_parameters={"carryover_threshold": 1.0e-3},
    )

    assert result["rerun_triggered"] is False
    assert result["rerun_run"] is None
    assert len(calls) == 1


def test_bulk_rerun_wrapper_reruns_only_failed_targets_with_override(tmp_path: Path, monkeypatch):
    output_root_dir = tmp_path / "result"
    summary_path = output_root_dir / "_bulk_summary" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    first_summary = {
        "failed_targets": 1,
        "results": [
            {"relative_path": "case_a/atom_1", "status": "success"},
            {"relative_path": "case_b/atom_2", "status": "failed"},
        ],
    }
    summary_path.write_text(json.dumps(first_summary))

    calls: list[dict] = []

    def fake_bulk_gb_sqd_flow(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("Bulk GB-SQD flow finished with 1 failed targets.")
        return {"failed_targets": 0, "results": [{"relative_path": "case_b/atom_2", "status": "success"}]}

    monkeypatch.setattr(bulk_rerun, "bulk_gb_sqd_flow", fake_bulk_gb_sqd_flow)

    result = bulk_rerun.bulk_gb_sqd_flow_with_failed_target_rerun(
        mode="ext_sqd",
        input_root_dir="/tmp/input",
        output_root_dir=str(output_root_dir),
        skip_completed=False,
        target_overrides={"case_a/atom_1": {"max_time": 900}},
        failed_target_override_parameters={"carryover_threshold": 1.0e-3},
    )

    assert result["rerun_triggered"] is True
    assert result["initial_error"] == "Bulk GB-SQD flow finished with 1 failed targets."
    assert result["rerun_target_overrides"] == {
        "case_a/atom_1": {"max_time": 900},
        "case_b/atom_2": {"carryover_threshold": 1.0e-3},
    }
    assert len(calls) == 2
    assert calls[1]["skip_completed"] is True
    assert calls[1]["target_overrides"] == {
        "case_a/atom_1": {"max_time": 900},
        "case_b/atom_2": {"carryover_threshold": 1.0e-3},
    }
