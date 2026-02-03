"""
Prefect Blocks for an HPC-agnostic execution demo on Miyabi.

Blocks:
- CommandBlock: WHAT to run (logical executable key + default args)
- ExecutionProfileBlock: baseline HOW to run for a specific command
- MiyabiHPCProfileBlock: WHERE/HOW to submit on Miyabi (PBS), including path/queue mapping

This is a minimal prototype intended for a live demo.
"""

from __future__ import annotations

from typing import Literal

from prefect.blocks.core import Block
from pydantic import Field


class CommandBlock(Block):
    """HPC-agnostic command definition (WHAT to run)."""

    _block_type_name = "HPC Command"
    _block_type_slug = "hpc_command"

    command_name: str = Field(title="Command Name", description="Logical command name, e.g., 'diag'.")
    executable_key: str = Field(
        title="Executable Key",
        description="Logical key for the executable. Resolved to an absolute path by the HPC profile.",
    )
    description: str | None = Field(default=None, title="Description")
    default_args: list[str] | None = Field(
        default=None,
        title="Default Args",
        description="Default arguments appended after the executable.",
    )


class ExecutionProfileBlock(Block):
    """
    Baseline execution profile (HOW to run) for a specific command.
    Command-aware by default to avoid ambiguous profiles.
    """

    _block_type_name = "Execution Profile"
    _block_type_slug = "execution_profile"

    profile_name: str = Field(title="Profile Name", description="Human-friendly profile name, e.g., 'diag-n16'.")
    command_name: str = Field(title="Command Name", description="Which command this profile is intended for.")
    resource_class: Literal["cpu", "gpu"] = Field(default="cpu", title="Resource Class")

    # Baseline intent (can be overridden by tuning)
    nodes: int = Field(default=1, gt=0, title="Nodes (Baseline)")
    walltime: str = Field(default="00:30:00", title="Walltime (Baseline)", description="Format HH:MM:SS")
    ranks_per_node: int = Field(default=1, gt=0, title="Ranks per Node (Baseline)")
    threads_per_rank: int = Field(default=1, gt=0, title="Threads per Rank (Baseline)")
    mem_gib: int | None = Field(
        default=None,
        gt=0,
        title="Memory (GiB, optional)",
        description="Optional semantic intent used to estimate nodes if nodes are not provided by user.",
    )

    launcher: Literal["mpirun", "mpiexec", "single"] = Field(default="mpirun", title="Launcher")


class MiyabiHPCProfileBlock(Block):
    """
    Miyabi-specific profile (WHERE/HOW to submit).

    For the demo, we keep it simple:
    - PBS directives generation
    - queue & project mapping
    - executable key -> absolute path mapping
    - basic node/memory estimation (optional)
    """

    _block_type_name = "Miyabi HPC Profile"
    _block_type_slug = "miyabi_hpc_profile"

    queue_cpu: str = Field(default="cpu", title="CPU Queue")
    queue_gpu: str = Field(default="gpu", title="GPU Queue")

    project_cpu: str | None = Field(default=None, title="CPU Project/Account")
    project_gpu: str | None = Field(default=None, title="GPU Project/Account")

    mem_per_node_cpu_gib: int = Field(default=512, gt=0, title="CPU Mem per Node (GiB)")
    mem_per_node_gpu_gib: int = Field(default=256, gt=0, title="GPU Mem per Node (GiB)")

    executable_map: dict[str, str] = Field(
        default_factory=dict,
        title="Executable Map",
        description="Map from executable_key -> absolute path on Miyabi.",
    )
