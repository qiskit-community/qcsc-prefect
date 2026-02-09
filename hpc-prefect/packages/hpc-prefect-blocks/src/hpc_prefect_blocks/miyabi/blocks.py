from __future__ import annotations

from typing import Literal

from prefect.blocks.core import Block
from pydantic import Field


class CommandBlock(Block):
    """Command definition (what to run)."""

    _block_type_name = "HPC Command"
    _block_type_slug = "hpc_command"

    command_name: str = Field(title="Command Name")
    executable_key: str = Field(title="Executable Key")
    description: str | None = Field(default=None, title="Description")
    default_args: list[str] = Field(default_factory=list, title="Default Args")


class ExecutionProfileBlock(Block):
    """Execution baseline (how to run)."""

    _block_type_name = "Execution Profile"
    _block_type_slug = "execution_profile"

    profile_name: str = Field(title="Profile Name")
    command_name: str = Field(title="Command Name")
    resource_class: Literal["cpu", "gpu"] = Field(default="cpu", title="Resource Class")

    num_nodes: int = Field(default=1, gt=0, title="Nodes")
    mpiprocs: int = Field(default=1, gt=0, title="MPI Procs per Node")
    ompthreads: int = Field(default=1, gt=0, title="OMP Threads")
    walltime: str = Field(default="00:05:00", title="Walltime (HH:MM:SS)")
    launcher: Literal["single", "mpirun", "mpiexec"] = Field(default="single", title="Launcher")
    mpi_options: list[str] = Field(default_factory=list, title="MPI Options")
    modules: list[str] = Field(default_factory=list, title="Modules")
    environments: dict[str, str] = Field(default_factory=dict, title="Environment Variables")


class HPCProfileBlock(Block):
    """HPC target-specific resolution (where to submit)."""

    _block_type_name = "HPC Profile"
    _block_type_slug = "hpc_profile"

    hpc_target: Literal["miyabi", "fugaku", "slurm"] = Field(default="miyabi", title="HPC Target")
    queue_cpu: str = Field(default="regular-c", title="CPU Queue")
    queue_gpu: str = Field(default="regular-g", title="GPU Queue")
    project_cpu: str = Field(title="CPU Project")
    project_gpu: str = Field(default="", title="GPU Project")
    executable_map: dict[str, str] = Field(default_factory=dict, title="Executable Map")
