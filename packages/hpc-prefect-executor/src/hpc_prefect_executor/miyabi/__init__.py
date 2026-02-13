from __future__ import annotations

from typing import Any

from .run import MiyabiRunResult, run_miyabi_job


async def run_miyabi_job_from_blocks(*args: Any, **kwargs: Any):
    # Lazy import avoids cycles with hpc_prefect_executor.from_blocks.
    from .from_blocks import run_miyabi_job_from_blocks as _run_miyabi_job_from_blocks

    return await _run_miyabi_job_from_blocks(*args, **kwargs)


__all__ = ["MiyabiRunResult", "run_miyabi_job", "run_miyabi_job_from_blocks"]
