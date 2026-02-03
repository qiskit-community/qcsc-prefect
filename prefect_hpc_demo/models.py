"""
Pydantic models for run-time parameters (UI/Deployment parameters).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Tuning(BaseModel):
    """
    User-adjustable parameters.
    These are intended for users who understand HPC basics.
    """

    nodes: int | None = Field(default=None, gt=0, description="Override node count")
    walltime: str | None = Field(default=None, description="Override walltime HH:MM:SS")
    ranks_per_node: int | None = Field(default=None, gt=0, description="Override ranks per node")
    threads_per_rank: int | None = Field(default=None, gt=0, description="Override threads per rank")
    mem_gib: int | None = Field(
        default=None,
        gt=0,
        description="Memory intent (GiB). Used for node estimation if nodes is not provided.",
    )
