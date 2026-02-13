from __future__ import annotations

import inspect
import json
from array import array
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from prefect import task
from prefect.blocks.core import Block
from pydantic import Field

from hpc_prefect_adapters.miyabi.builder import MiyabiJobRequest
from hpc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock
from hpc_prefect_core.models.execution_profile import ExecutionProfile
from hpc_prefect_executor.miyabi.run import run_miyabi_job

BITLEN = 10


async def _resolve_loaded_block(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _resolve_queue_and_project(hpc_block: HPCProfileBlock, resource_class: str) -> tuple[str, str]:
    if resource_class == "gpu":
        return hpc_block.queue_gpu, hpc_block.project_gpu
    return hpc_block.queue_cpu, hpc_block.project_cpu


def _make_job_work_dir(base_work_dir: Path) -> Path:
    base_work_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_dir = base_work_dir / f"job_{timestamp}_{uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


def _read_counts(job_work_dir: Path, bitlen: int) -> dict[str, int]:
    hist_path = job_work_dir / "hist_u64.bin"
    if hist_path.exists():
        hist = array("Q")
        with hist_path.open("rb") as f:
            hist.frombytes(f.read())
        expected = 1 << bitlen
        if len(hist) != expected:
            raise ValueError(f"Unexpected histogram size: expected {expected}, got {len(hist)}")
        return {format(i, f"0{bitlen}b"): int(v) for i, v in enumerate(hist) if v > 0}

    json_path = job_work_dir / "output.json"
    if json_path.exists():
        raw_counts = json.loads(json_path.read_text(encoding="utf-8"))
        return {format(int(k), f"0{bitlen}b"): int(v) for k, v in raw_counts.items()}

    raise FileNotFoundError(
        f"Neither hist_u64.bin nor output.json was generated in {job_work_dir}"
    )


class BitCounter(Block):
    """
    Facade block for legacy tutorial style.

    `BitCounter.load(...).get(bitstrings)` internally resolves:
    - CommandBlock
    - ExecutionProfileBlock
    - HPCProfileBlock
    """

    _block_type_name = "Bit Counter"
    _block_type_slug = "bit-counter"

    root_dir: str = Field(title="Root Directory")
    command_block_name: str = Field(default="cmd-bitcount-hist", title="Command Block Name")
    execution_profile_block_name: str = Field(default="exec-bitcount-mpi", title="Execution Profile Block Name")
    hpc_profile_block_name: str = Field(default="hpc-miyabi-bitcount", title="HPC Profile Block Name")
    script_filename: str = Field(default="bitcount_facade.pbs", title="Script Filename")
    metrics_artifact_key: str = Field(default="miyabi-bitcount-facade-metrics", title="Metrics Artifact Key")
    bitlen: int = Field(default=BITLEN, gt=0, title="Bit Length")
    user_args: list[str] = Field(default_factory=list, title="Command Args")

    async def get(
        self,
        bitstrings: list[str],
    ) -> dict[str, int]:
        return await _get_inner(self, bitstrings)


@task(name="get_counts_mpi")
async def _get_inner(
    counter: BitCounter,
    bitstrings: list[str],
) -> dict[str, int]:
    cmd = await _resolve_loaded_block(CommandBlock.load(counter.command_block_name))
    profile_block = await _resolve_loaded_block(
        ExecutionProfileBlock.load(counter.execution_profile_block_name)
    )
    hpc_block = await _resolve_loaded_block(HPCProfileBlock.load(counter.hpc_profile_block_name))

    if profile_block.command_name != cmd.command_name:
        raise ValueError(
            f"ExecutionProfileBlock '{profile_block.profile_name}' is for command "
            f"'{profile_block.command_name}', but command block is '{cmd.command_name}'."
        )

    executable = hpc_block.executable_map.get(cmd.executable_key)
    if not executable:
        raise KeyError(
            f"Executable key '{cmd.executable_key}' was not found in "
            f"HPCProfileBlock '{counter.hpc_profile_block_name}'."
        )

    queue, project = _resolve_queue_and_project(hpc_block, profile_block.resource_class)
    if not project:
        raise ValueError("Project is empty. Update HPCProfileBlock project_cpu/project_gpu.")

    base_work_dir = Path(counter.root_dir).expanduser().resolve()
    job_work_dir = _make_job_work_dir(base_work_dir)

    u32_values = array("I", (int(bits, 2) for bits in bitstrings))
    with (job_work_dir / "input.bin").open("wb") as f:
        u32_values.tofile(f)

    arguments = list(cmd.default_args)
    if counter.user_args:
        arguments.extend(counter.user_args)

    exec_profile = ExecutionProfile(
        command_key=cmd.command_name,
        num_nodes=profile_block.num_nodes,
        mpiprocs=profile_block.mpiprocs,
        ompthreads=profile_block.ompthreads,
        walltime=profile_block.walltime,
        launcher=profile_block.launcher,
        mpi_options=list(profile_block.mpi_options),
        modules=list(profile_block.modules),
        environments=dict(profile_block.environments),
        arguments=arguments,
    )

    req = MiyabiJobRequest(
        queue_name=queue,
        project=project,
        executable=executable,
    )

    result = await run_miyabi_job(
        work_dir=job_work_dir,
        script_filename=counter.script_filename,
        exec_profile=exec_profile,
        req=req,
        watch_poll_interval=5.0,
        timeout_seconds=1800,
        metrics_artifact_key=counter.metrics_artifact_key,
    )
    if result.exit_status != 0:
        raise RuntimeError(f"BitCounter job failed: exit_status={result.exit_status}")

    return _read_counts(job_work_dir, counter.bitlen)
