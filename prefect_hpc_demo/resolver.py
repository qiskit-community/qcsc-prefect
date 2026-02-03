"""
Resolution logic: (CommandBlock + ExecutionProfileBlock + Tuning + MiyabiHPCProfileBlock) -> ResolvedRun
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .blocks import CommandBlock, ExecutionProfileBlock, MiyabiHPCProfileBlock
from .models import Tuning


@dataclass(frozen=True)
class ResolvedRun:
    command_name: str
    profile_name: str
    executable: str
    launcher: Literal["mpirun", "mpiexec", "single"]
    queue: str
    project: str | None
    nodes: int
    walltime: str
    ranks_per_node: int
    threads_per_rank: int
    total_ranks: int
    argv: list[str]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def resolve_run(
    *,
    cmd: CommandBlock,
    profile: ExecutionProfileBlock,
    hpc: MiyabiHPCProfileBlock,
    tuning: Tuning | None,
    user_args: list[str] | None,
) -> ResolvedRun:
    if profile.command_name != cmd.command_name:
        raise ValueError(
            f"ExecutionProfile '{profile.profile_name}' is for command '{profile.command_name}', "
            f"but you selected command '{cmd.command_name}'."
        )

    nodes = profile.nodes
    walltime = profile.walltime
    ranks_per_node = profile.ranks_per_node
    threads_per_rank = profile.threads_per_rank
    mem_gib = profile.mem_gib

    if tuning is not None:
        if tuning.walltime is not None:
            walltime = tuning.walltime
        if tuning.ranks_per_node is not None:
            ranks_per_node = tuning.ranks_per_node
        if tuning.threads_per_rank is not None:
            threads_per_rank = tuning.threads_per_rank

        if tuning.nodes is not None:
            nodes = tuning.nodes
        else:
            effective_mem = tuning.mem_gib if tuning.mem_gib is not None else mem_gib
            if effective_mem is not None:
                per_node = hpc.mem_per_node_gpu_gib if profile.resource_class == "gpu" else hpc.mem_per_node_cpu_gib
                nodes = _ceil_div(effective_mem, per_node)

    if profile.resource_class == "gpu":
        queue = hpc.queue_gpu
        project = hpc.project_gpu
    else:
        queue = hpc.queue_cpu
        project = hpc.project_cpu

    if cmd.executable_key not in hpc.executable_map:
        raise KeyError(
            f"Executable key '{cmd.executable_key}' is not registered in MiyabiHPCProfileBlock.executable_map"
        )
    executable = hpc.executable_map[cmd.executable_key]

    total_ranks = nodes * ranks_per_node

    argv = [executable]
    if cmd.default_args:
        argv += cmd.default_args
    if user_args:
        argv += user_args

    return ResolvedRun(
        command_name=cmd.command_name,
        profile_name=profile.profile_name,
        executable=executable,
        launcher=profile.launcher,
        queue=queue,
        project=project,
        nodes=nodes,
        walltime=walltime,
        ranks_per_node=ranks_per_node,
        threads_per_rank=threads_per_rank,
        total_ranks=total_ranks,
        argv=argv,
    )
