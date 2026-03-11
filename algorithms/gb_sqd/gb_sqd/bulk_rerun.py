"""Helpers for rerunning failed GB-SQD bulk targets with override parameters."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .bulk import bulk_gb_sqd_flow


def bulk_summary_path(output_root_dir: str | Path) -> Path:
    return Path(output_root_dir).expanduser().resolve() / "_bulk_summary" / "run_summary.json"


def load_bulk_summary(output_root_dir: str | Path) -> dict[str, Any]:
    summary_path = bulk_summary_path(output_root_dir)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Bulk summary not found: {summary_path}")
    return json.loads(summary_path.read_text())


def build_failed_target_overrides(
    *,
    summary: Mapping[str, Any],
    override_parameters: Mapping[str, Any],
    base_target_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    failed_paths = [
        result["relative_path"]
        for result in summary.get("results", [])
        if result.get("status") == "failed" and result.get("relative_path")
    ]

    merged: dict[str, dict[str, Any]] = copy.deepcopy(base_target_overrides or {})
    for relative_path in failed_paths:
        target_override = dict(merged.get(relative_path, {}))
        target_override.update(dict(override_parameters))
        merged[relative_path] = target_override
    return merged


def bulk_gb_sqd_flow_with_failed_target_rerun(
    *,
    failed_target_override_parameters: Mapping[str, Any],
    **bulk_flow_kwargs: Any,
) -> dict[str, Any]:
    """Run bulk GB-SQD once, then rerun only failed targets with override parameters."""

    output_root_dir = bulk_flow_kwargs["output_root_dir"]
    initial_target_overrides = copy.deepcopy(bulk_flow_kwargs.get("target_overrides") or {})

    try:
        initial_summary = bulk_gb_sqd_flow(**bulk_flow_kwargs)
        return {
            "initial_run": initial_summary,
            "initial_error": None,
            "rerun_triggered": False,
            "rerun_target_overrides": {},
            "rerun_run": None,
        }
    except Exception as exc:
        initial_error = str(exc)
        initial_summary = load_bulk_summary(output_root_dir)
        rerun_target_overrides = build_failed_target_overrides(
            summary=initial_summary,
            override_parameters=failed_target_override_parameters,
            base_target_overrides=initial_target_overrides,
        )
        if not rerun_target_overrides:
            raise

    rerun_kwargs = dict(bulk_flow_kwargs)
    rerun_kwargs["skip_completed"] = True
    rerun_kwargs["target_overrides"] = rerun_target_overrides
    rerun_summary = bulk_gb_sqd_flow(**rerun_kwargs)
    return {
        "initial_run": initial_summary,
        "initial_error": initial_error,
        "rerun_triggered": True,
        "rerun_target_overrides": rerun_target_overrides,
        "rerun_run": rerun_summary,
    }
