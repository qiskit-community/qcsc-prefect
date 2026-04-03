"""Create Prefect blocks and variables required for the SKQD Z2LGT workflow."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

from prefect.variables import Variable

from qcsc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock

from .block_defaults import (
    DEFAULT_COMMAND_BLOCK_NAMES,
    DEFAULT_COMMAND_NAMES,
    DEFAULT_EXECUTION_PROFILE_BLOCK_NAMES,
    DEFAULT_HPC_PROFILE_BLOCK_NAME,
    DEFAULT_OPTIONS_VARIABLE_NAME,
    DEFAULT_PROFILE_NAMES,
)


def _register_block_types() -> None:
    for block_cls in (CommandBlock, ExecutionProfileBlock, HPCProfileBlock):
        register = getattr(block_cls, "register_type_and_schema", None)
        if callable(register):
            register()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Prefect blocks for the SKQD Z2LGT workflow."
    )
    parser.add_argument("--config", type=Path, help="Path to TOML or JSON config file.")

    parser.add_argument("--hpc-target", choices=["miyabi", "fugaku"])
    parser.add_argument("--project")
    parser.add_argument("--project-cpu")
    parser.add_argument("--project-gpu")
    parser.add_argument("--queue")
    parser.add_argument("--queue-cpu")
    parser.add_argument("--queue-gpu")

    parser.add_argument("--python")
    parser.add_argument("--python-cpu")
    parser.add_argument("--python-gpu")

    parser.add_argument("--cpu-modules", nargs="*")
    parser.add_argument("--gpu-modules", nargs="*")
    parser.add_argument("--cpu-pre-commands", nargs="*")
    parser.add_argument("--gpu-pre-commands", nargs="*")
    parser.add_argument("--cpu-environments-json")
    parser.add_argument("--gpu-environments-json")
    parser.add_argument("--cpu-ompthreads", type=int)
    parser.add_argument("--gpu-ompthreads", type=int)
    parser.add_argument("--cpu-mpi-options", nargs="*")
    parser.add_argument("--gpu-mpi-options", nargs="*")

    parser.add_argument("--dmrg-walltime")
    parser.add_argument("--preprocess-walltime")
    parser.add_argument("--train-walltime")
    parser.add_argument("--diagonalize-walltime")
    parser.add_argument("--dmrg-num-nodes", type=int)
    parser.add_argument("--dmrg-mpiprocs", type=int)

    parser.add_argument("--options-variable-name")
    parser.add_argument("--runtime-options-json")

    parser.add_argument("--hpc-profile-block-name")
    parser.add_argument("--dmrg-command-block-name")
    parser.add_argument("--preprocess-command-block-name")
    parser.add_argument("--train-command-block-name")
    parser.add_argument("--diagonalize-command-block-name")
    parser.add_argument("--dmrg-execution-profile-block-name")
    parser.add_argument("--preprocess-execution-profile-block-name")
    parser.add_argument("--train-execution-profile-block-name")
    parser.add_argument("--diagonalize-execution-profile-block-name")
    return parser.parse_args()


def _load_config_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file was not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as f:
        return tomllib.load(f)


def _pick_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _require(name: str, value: Any) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise RuntimeError(f"Missing required setting: {name}")
    return value


def _normalize_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"Expected list[str] or CSV string, got: {type(value)}")


def _normalize_str_dict(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _normalize_str_dict(json.loads(value))
    if isinstance(value, dict):
        normalized = {
            str(key).strip(): str(val).strip()
            for key, val in value.items()
            if str(key).strip()
        }
        return normalized or None
    raise ValueError(f"Expected dict[str, str], got: {type(value)}")


def _default_hpc_profile_name(hpc_target: str) -> str:
    if hpc_target == "miyabi":
        return DEFAULT_HPC_PROFILE_BLOCK_NAME
    return f"hpc-{hpc_target}-skqd-z2lgt"


def _default_cpu_launcher(hpc_target: str) -> str:
    return "mpiexec.hydra" if hpc_target == "miyabi" else "mpiexec"


def _default_gpu_launcher(hpc_target: str) -> str:
    return "mpirun"


def _save_command_block(
    *,
    document_name: str,
    command_name: str,
    executable_key: str,
    module_name: str,
) -> None:
    CommandBlock(
        command_name=command_name,
        executable_key=executable_key,
        description=f"Python entrypoint for {command_name}",
        default_args=["-u", "-m", module_name],
    ).save(document_name, overwrite=True)


def _save_execution_profile_block(
    *,
    document_name: str,
    profile_name: str,
    command_name: str,
    resource_class: str,
    num_nodes: int,
    mpiprocs: int,
    ompthreads: int | None,
    walltime: str,
    launcher: str,
    mpi_options: list[str] | None,
    modules: list[str] | None,
    pre_commands: list[str] | None,
    environments: dict[str, str] | None,
) -> None:
    ExecutionProfileBlock(
        profile_name=profile_name,
        command_name=command_name,
        resource_class=resource_class,
        num_nodes=num_nodes,
        mpiprocs=mpiprocs,
        ompthreads=ompthreads,
        walltime=walltime,
        launcher=launcher,
        mpi_options=list(mpi_options or []),
        modules=list(modules or []),
        pre_commands=list(pre_commands or []),
        environments=dict(environments or {}),
    ).save(document_name, overwrite=True)


def main() -> None:
    args = _parse_args()
    config = _load_config_file(args.config)
    _register_block_types()

    hpc_target = str(_pick_value(args.hpc_target, config.get("hpc_target"), "miyabi")).strip().lower()
    if hpc_target not in {"miyabi", "fugaku"}:
        raise RuntimeError("'hpc_target' must be either 'miyabi' or 'fugaku'.")

    project_common = _pick_value(args.project, config.get("project"))
    queue_common = _pick_value(args.queue, config.get("queue"))
    python_common = _pick_value(args.python, config.get("python"))

    project_cpu = str(_require(
        "project_cpu/project",
        _pick_value(args.project_cpu, config.get("project_cpu"), project_common),
    )).strip()
    project_gpu = str(_pick_value(args.project_gpu, config.get("project_gpu"), project_cpu)).strip()
    queue_cpu = str(_require(
        "queue_cpu/queue",
        _pick_value(args.queue_cpu, config.get("queue_cpu"), queue_common),
    )).strip()
    queue_gpu = str(_pick_value(args.queue_gpu, config.get("queue_gpu"), queue_cpu)).strip()

    python_cpu = str(Path(str(_require(
        "python_cpu/python",
        _pick_value(args.python_cpu, config.get("python_cpu"), python_common),
    ))).expanduser().resolve())
    python_gpu_value = _pick_value(args.python_gpu, config.get("python_gpu"), python_common, python_cpu)
    python_gpu = str(Path(str(_require("python_gpu/python", python_gpu_value))).expanduser().resolve())

    cpu_modules = _normalize_str_list(_pick_value(args.cpu_modules, config.get("cpu_modules")))
    gpu_modules = _normalize_str_list(_pick_value(args.gpu_modules, config.get("gpu_modules")))
    cpu_pre_commands = _normalize_str_list(_pick_value(args.cpu_pre_commands, config.get("cpu_pre_commands")))
    gpu_pre_commands = _normalize_str_list(_pick_value(args.gpu_pre_commands, config.get("gpu_pre_commands")))
    cpu_environments = _normalize_str_dict(
        _pick_value(args.cpu_environments_json, config.get("cpu_environments"))
    )
    gpu_environments = _normalize_str_dict(
        _pick_value(args.gpu_environments_json, config.get("gpu_environments"))
    )

    cpu_ompthreads = _pick_value(args.cpu_ompthreads, config.get("cpu_ompthreads"))
    gpu_ompthreads = _pick_value(args.gpu_ompthreads, config.get("gpu_ompthreads"))
    cpu_mpi_options = _normalize_str_list(_pick_value(args.cpu_mpi_options, config.get("cpu_mpi_options")))
    gpu_mpi_options = _normalize_str_list(_pick_value(args.gpu_mpi_options, config.get("gpu_mpi_options")))

    dmrg_walltime = str(_pick_value(args.dmrg_walltime, config.get("dmrg_walltime"), "01:00:00")).strip()
    preprocess_walltime = str(
        _pick_value(args.preprocess_walltime, config.get("preprocess_walltime"), "00:10:00")
    ).strip()
    train_walltime = str(_pick_value(args.train_walltime, config.get("train_walltime"), "01:00:00")).strip()
    diagonalize_walltime = str(
        _pick_value(args.diagonalize_walltime, config.get("diagonalize_walltime"), "02:00:00")
    ).strip()

    dmrg_num_nodes = int(_pick_value(args.dmrg_num_nodes, config.get("dmrg_num_nodes"), 1))
    dmrg_mpiprocs = int(_pick_value(args.dmrg_mpiprocs, config.get("dmrg_mpiprocs"), 1))

    hpc_profile_block_name = str(
        _pick_value(args.hpc_profile_block_name, config.get("hpc_profile_block_name"), _default_hpc_profile_name(hpc_target))
    ).strip()
    options_variable_name = str(
        _pick_value(args.options_variable_name, config.get("options_variable_name"), DEFAULT_OPTIONS_VARIABLE_NAME)
    ).strip()

    command_block_names = {
        "dmrg": str(_pick_value(
            args.dmrg_command_block_name,
            config.get("dmrg_command_block_name"),
            DEFAULT_COMMAND_BLOCK_NAMES["dmrg"],
        )).strip(),
        "preprocess": str(_pick_value(
            args.preprocess_command_block_name,
            config.get("preprocess_command_block_name"),
            DEFAULT_COMMAND_BLOCK_NAMES["preprocess"],
        )).strip(),
        "train": str(_pick_value(
            args.train_command_block_name,
            config.get("train_command_block_name"),
            DEFAULT_COMMAND_BLOCK_NAMES["train"],
        )).strip(),
        "diagonalize": str(_pick_value(
            args.diagonalize_command_block_name,
            config.get("diagonalize_command_block_name"),
            DEFAULT_COMMAND_BLOCK_NAMES["diagonalize"],
        )).strip(),
    }
    execution_profile_block_names = {
        "dmrg": str(_pick_value(
            args.dmrg_execution_profile_block_name,
            config.get("dmrg_execution_profile_block_name"),
            DEFAULT_EXECUTION_PROFILE_BLOCK_NAMES["dmrg"],
        )).strip(),
        "preprocess": str(_pick_value(
            args.preprocess_execution_profile_block_name,
            config.get("preprocess_execution_profile_block_name"),
            DEFAULT_EXECUTION_PROFILE_BLOCK_NAMES["preprocess"],
        )).strip(),
        "train": str(_pick_value(
            args.train_execution_profile_block_name,
            config.get("train_execution_profile_block_name"),
            DEFAULT_EXECUTION_PROFILE_BLOCK_NAMES["train"],
        )).strip(),
        "diagonalize": str(_pick_value(
            args.diagonalize_execution_profile_block_name,
            config.get("diagonalize_execution_profile_block_name"),
            DEFAULT_EXECUTION_PROFILE_BLOCK_NAMES["diagonalize"],
        )).strip(),
    }

    _save_command_block(
        document_name=command_block_names["dmrg"],
        command_name=DEFAULT_COMMAND_NAMES["dmrg"],
        executable_key="python_cpu",
        module_name="skqd_z2lgt.tasks.dmrg",
    )
    _save_command_block(
        document_name=command_block_names["preprocess"],
        command_name=DEFAULT_COMMAND_NAMES["preprocess"],
        executable_key="python_cpu",
        module_name="skqd_z2lgt.tasks.preprocess",
    )
    _save_command_block(
        document_name=command_block_names["train"],
        command_name=DEFAULT_COMMAND_NAMES["train"],
        executable_key="python_gpu",
        module_name="skqd_z2lgt.tasks.train_generator",
    )
    _save_command_block(
        document_name=command_block_names["diagonalize"],
        command_name=DEFAULT_COMMAND_NAMES["diagonalize"],
        executable_key="python_gpu",
        module_name="skqd_z2lgt.tasks.diagonalize",
    )

    _save_execution_profile_block(
        document_name=execution_profile_block_names["dmrg"],
        profile_name=DEFAULT_PROFILE_NAMES["dmrg"],
        command_name=DEFAULT_COMMAND_NAMES["dmrg"],
        resource_class="cpu",
        num_nodes=dmrg_num_nodes,
        mpiprocs=dmrg_mpiprocs,
        ompthreads=cpu_ompthreads,
        walltime=dmrg_walltime,
        launcher="single",
        mpi_options=[],
        modules=cpu_modules,
        pre_commands=cpu_pre_commands,
        environments=cpu_environments,
    )
    _save_execution_profile_block(
        document_name=execution_profile_block_names["preprocess"],
        profile_name=DEFAULT_PROFILE_NAMES["preprocess"],
        command_name=DEFAULT_COMMAND_NAMES["preprocess"],
        resource_class="cpu",
        num_nodes=1,
        mpiprocs=1,
        ompthreads=cpu_ompthreads,
        walltime=preprocess_walltime,
        launcher=_default_cpu_launcher(hpc_target),
        mpi_options=cpu_mpi_options,
        modules=cpu_modules,
        pre_commands=cpu_pre_commands,
        environments=cpu_environments,
    )
    _save_execution_profile_block(
        document_name=execution_profile_block_names["train"],
        profile_name=DEFAULT_PROFILE_NAMES["train"],
        command_name=DEFAULT_COMMAND_NAMES["train"],
        resource_class="gpu",
        num_nodes=1,
        mpiprocs=1,
        ompthreads=gpu_ompthreads,
        walltime=train_walltime,
        launcher=_default_gpu_launcher(hpc_target),
        mpi_options=gpu_mpi_options,
        modules=gpu_modules,
        pre_commands=gpu_pre_commands,
        environments=gpu_environments,
    )
    _save_execution_profile_block(
        document_name=execution_profile_block_names["diagonalize"],
        profile_name=DEFAULT_PROFILE_NAMES["diagonalize"],
        command_name=DEFAULT_COMMAND_NAMES["diagonalize"],
        resource_class="gpu",
        num_nodes=1,
        mpiprocs=1,
        ompthreads=gpu_ompthreads,
        walltime=diagonalize_walltime,
        launcher=_default_gpu_launcher(hpc_target),
        mpi_options=gpu_mpi_options,
        modules=gpu_modules,
        pre_commands=gpu_pre_commands,
        environments=gpu_environments,
    )

    HPCProfileBlock(
        hpc_target=hpc_target,
        queue_cpu=queue_cpu,
        queue_gpu=queue_gpu,
        project_cpu=project_cpu,
        project_gpu=project_gpu,
        executable_map={
            "python_cpu": python_cpu,
            "python_gpu": python_gpu,
        },
    ).save(hpc_profile_block_name, overwrite=True)

    runtime_options = _pick_value(args.runtime_options_json, config.get("runtime_options_json"))
    if runtime_options is not None:
        variable_value = json.loads(str(runtime_options))
    else:
        variable_value = config.get("runtime_options", {})
    Variable.set(options_variable_name, variable_value, overwrite=True)

    print(f"Saved blocks and variable for SKQD Z2LGT workflow (target={hpc_target})")
    print(f"  HPC profile block: {hpc_profile_block_name}")
    print(f"  Command blocks: {command_block_names}")
    print(f"  Execution profile blocks: {execution_profile_block_names}")
    print(f"  Options variable: {options_variable_name}")
    print(f"  CPU Python: {python_cpu}")
    print(f"  GPU Python: {python_gpu}")


if __name__ == "__main__":
    main()
