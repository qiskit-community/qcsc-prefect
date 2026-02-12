from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from hpc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock

from examples.miyabi_prefect_bitcount_demo.wrapper_block import BitCounterWrapperBlock


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _split_csv(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_default_work_dir(project: str) -> str:
    user = os.getenv("USER", "").strip() or "your_user"
    return f"/work/{project}/{user}/miyabi_bitcount_tutorial"


def _resolve_example_path() -> Path:
    return Path(__file__).resolve().parent


def _register_block_types() -> None:
    block_types = [
        CommandBlock,
        ExecutionProfileBlock,
        HPCProfileBlock,
        BitCounterWrapperBlock,
    ]
    for block_cls in block_types:
        register = getattr(block_cls, "register_type_and_schema", None)
        if callable(register):
            register()


def _set_variable(variable_name: str, shots: int) -> None:
    value = json.dumps({"params": {"shots": shots}})
    subprocess.run(
        ["prefect", "variable", "set", variable_name, value, "--overwrite"],
        check=True,
    )


def main() -> None:
    project = os.getenv("MIYABI_PBS_PROJECT", "").strip()
    if not project:
        raise RuntimeError("Set MIYABI_PBS_PROJECT before running create_blocks.py.")

    queue = os.getenv("MIYABI_PBS_QUEUE", "regular-c").strip()
    launcher = os.getenv("MIYABI_LAUNCHER", "mpiexec").strip()
    walltime = os.getenv("MIYABI_WALLTIME", "00:10:00").strip()

    num_nodes = _env_int("MIYABI_NUM_NODES", 2)
    mpiprocs = _env_int("MIYABI_MPIPROCS", 5)
    ompthreads = _env_int("MIYABI_OMPTHREADS", 1)
    shots = _env_int("BITCOUNT_SHOTS", 100000)

    modules = _split_csv("MIYABI_MODULES", ["intel/2023.2.0", "impi/2021.10.0"])
    mpi_options = _split_csv("MIYABI_MPI_OPTIONS", [])

    example_dir = _resolve_example_path()
    default_wrapper_exec = str((example_dir / "bin/get_counts_json").resolve())
    default_optimized_exec = str((example_dir / "bin/get_counts_hist").resolve())

    wrapper_exec = os.getenv("BITCOUNT_WRAPPER_EXECUTABLE", default_wrapper_exec).strip()
    optimized_exec = os.getenv("BITCOUNT_OPT_EXECUTABLE", default_optimized_exec).strip()

    work_dir = os.getenv("MIYABI_BITCOUNT_WORK_DIR", _resolve_default_work_dir(project)).strip()

    wrapper_block_name = os.getenv("BITCOUNT_WRAPPER_BLOCK_NAME", "bit-counter-wrapper-demo").strip()
    cmd_block_name = os.getenv("BITCOUNT_CMD_BLOCK_NAME", "cmd-bitcount-hist").strip()
    exec_block_name = os.getenv("BITCOUNT_EXEC_PROFILE_BLOCK_NAME", "exec-bitcount-mpi").strip()
    hpc_block_name = os.getenv("BITCOUNT_HPC_PROFILE_BLOCK_NAME", "hpc-miyabi-bitcount").strip()
    options_variable_name = os.getenv("BITCOUNT_OPTIONS_VARIABLE", "miyabi-bitcount-options").strip()

    _register_block_types()

    BitCounterWrapperBlock(
        root_dir=work_dir,
        executable=wrapper_exec,
        queue_name=queue,
        project=project,
        num_nodes=num_nodes,
        num_mpi_processes=mpiprocs,
        ompthreads=ompthreads,
        walltime=walltime,
        launcher=launcher,
        load_modules=modules,
        mpi_options=mpi_options,
        environments={},
        script_filename="bitcount_wrapper.pbs",
    ).save(wrapper_block_name, overwrite=True)

    CommandBlock(
        command_name="bitcount-hist",
        executable_key="bitcount_hist",
        description="Optimized bitcount executable (binary histogram output)",
        default_args=[],
    ).save(cmd_block_name, overwrite=True)

    ExecutionProfileBlock(
        profile_name="bitcount-mpi",
        command_name="bitcount-hist",
        resource_class="cpu",
        num_nodes=num_nodes,
        mpiprocs=mpiprocs,
        ompthreads=ompthreads,
        walltime=walltime,
        launcher=launcher,
        mpi_options=mpi_options,
        modules=modules,
        environments={},
    ).save(exec_block_name, overwrite=True)

    HPCProfileBlock(
        hpc_target="miyabi",
        queue_cpu=queue,
        queue_gpu="regular-g",
        project_cpu=project,
        project_gpu=project,
        executable_map={"bitcount_hist": optimized_exec},
    ).save(hpc_block_name, overwrite=True)

    _set_variable(options_variable_name, shots)

    print("Saved blocks and variable for Miyabi BitCount demo")
    print(f"  Wrapper block: {wrapper_block_name}")
    print(f"  Command block: {cmd_block_name}")
    print(f"  Execution profile block: {exec_block_name}")
    print(f"  HPC profile block: {hpc_block_name}")
    print(f"  Options variable: {options_variable_name}")
    print(f"  Work directory: {work_dir}")
    print(f"  Wrapper executable: {wrapper_exec}")
    print(f"  Optimized executable: {optimized_exec}")


if __name__ == "__main__":
    main()
