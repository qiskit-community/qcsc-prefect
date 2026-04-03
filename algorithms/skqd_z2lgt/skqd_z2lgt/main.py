"""Definition of the main Prefect flow for Z2LGT SKQD."""
import os
from pathlib import Path
import logging
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import h5py
from prefect import flow, task, get_run_logger
from prefect.client.schemas.filters import (ArtifactFilter, ArtifactFilterKey,
                                            ArtifactFilterTaskRunId)
from prefect.client.orchestration import get_client
from prefect.variables import Variable
from prefect.runtime import flow_run, task_run
from prefect_qiskit.runtime import QuantumRuntime
from prefect_qiskit.primitives import PrimitiveJobRun
from qcsc_prefect_executor.from_blocks import run_job_from_blocks

from skqd_z2lgt.block_defaults import (
    DEFAULT_COMMAND_BLOCK_NAMES,
    DEFAULT_EXECUTION_PROFILE_BLOCK_NAMES,
    DEFAULT_HPC_PROFILE_BLOCK_NAME,
    DEFAULT_METRICS_ARTIFACT_KEYS,
    DEFAULT_OPTIONS_VARIABLE_NAME,
    DEFAULT_RUNTIME_NAME,
    DEFAULT_SCRIPT_FILENAMES,
)
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.tasks.open_output import open_output
from skqd_z2lgt.tasks.dmrg import dmrg_flow
from skqd_z2lgt.tasks.sample_quantum import sample_quantum_flow
from skqd_z2lgt.tasks.preprocess import preprocess_flow
from skqd_z2lgt.tasks.train_generator import train_generator_flow
from skqd_z2lgt.tasks.diagonalize import check_saved_result


def _prepare_pkgpath(parameters: Parameters, root_dir: str | None) -> None:
    if parameters.pkgpath:
        parameters.pkgpath = str(Path(parameters.pkgpath).expanduser().resolve())
        return

    if not root_dir:
        raise ValueError(
            "parameters.pkgpath is empty. Pass a shared filesystem path via "
            "'parameters.pkgpath' or the flow argument 'root_dir'."
        )

    try:
        run_id = flow_run.id
    except Exception:  # pragma: no cover - defensive only
        run_id = None

    if not run_id:
        run_id = uuid4().hex

    base_dir = Path(root_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    parameters.pkgpath = str(base_dir / run_id)


async def _run_hpc_script_job(
    *,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    user_args: list[str],
    script_filename: str,
    metrics_artifact_key: str,
    task_label: str,
    execution_profile_overrides: dict[str, object] | None = None,
):
    result = await run_job_from_blocks(
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        work_dir=Path(user_args[0]).expanduser().resolve().parent
        if user_args and user_args[0].endswith("parameters.json")
        else Path(user_args[0]).expanduser().resolve(),
        script_filename=script_filename,
        user_args=user_args,
        watch_poll_interval=5.0,
        metrics_artifact_key=metrics_artifact_key,
        execution_profile_overrides=execution_profile_overrides,
    )
    if result.exit_status != 0:
        raise RuntimeError(f"HPC job for {task_label} failed with exit_status={result.exit_status}.")
    return result


@flow
async def skqd_z2lgt(
    parameters: Parameters,
    runtime_name: str = DEFAULT_RUNTIME_NAME,
    option_name: str = DEFAULT_OPTIONS_VARIABLE_NAME,
    root_dir: str | None = None,
    hpc_profile_block_name: str = DEFAULT_HPC_PROFILE_BLOCK_NAME,
    dmrg_command_block_name: str = DEFAULT_COMMAND_BLOCK_NAMES["dmrg"],
    dmrg_execution_profile_block_name: str = DEFAULT_EXECUTION_PROFILE_BLOCK_NAMES["dmrg"],
    preprocess_command_block_name: str = DEFAULT_COMMAND_BLOCK_NAMES["preprocess"],
    preprocess_execution_profile_block_name: str = DEFAULT_EXECUTION_PROFILE_BLOCK_NAMES["preprocess"],
    train_command_block_name: str = DEFAULT_COMMAND_BLOCK_NAMES["train"],
    train_execution_profile_block_name: str = DEFAULT_EXECUTION_PROFILE_BLOCK_NAMES["train"],
    diagonalize_command_block_name: str = DEFAULT_COMMAND_BLOCK_NAMES["diagonalize"],
    diagonalize_execution_profile_block_name: str = DEFAULT_EXECUTION_PROFILE_BLOCK_NAMES["diagonalize"],
) -> float:
    """Calculation of ground-state energy of Z2 LGT using SKQD.

    Args:
        parameters: Configuration parameters.
        runtime_name: Name of the QuantumRuntime block.
        option_name: Name of the Prefect Variable storing sampler primitive options.
        root_dir: Shared filesystem root used when ``parameters.pkgpath`` is empty.
        hpc_profile_block_name: Name of the shared HPCProfileBlock.
    """
    logger = get_run_logger()
    logger.setLevel(logging.INFO)

    _prepare_pkgpath(parameters, root_dir)
    if not parameters.runtime.options_name:
        parameters.runtime.options_name = option_name
    open_output(parameters, logger)

    if parameters.dmrg:
        dmrg_future = dmrg.submit(
            parameters,
            command_block_name=dmrg_command_block_name,
            execution_profile_block_name=dmrg_execution_profile_block_name,
            hpc_profile_block_name=hpc_profile_block_name,
        )
    sample_quantum_future = sample_quantum.submit(parameters,
                                                  runtime_name=runtime_name,
                                                  option_name=option_name)
    preprocess_future = preprocess.submit(parameters,
                                          command_block_name=preprocess_command_block_name,
                                          execution_profile_block_name=preprocess_execution_profile_block_name,
                                          hpc_profile_block_name=hpc_profile_block_name,
                                          wait_for=[sample_quantum_future])
    train_generator_future = train_generator.submit(parameters,
                                                    command_block_name=train_command_block_name,
                                                    execution_profile_block_name=train_execution_profile_block_name,
                                                    hpc_profile_block_name=hpc_profile_block_name,
                                                    wait_for=[preprocess_future])
    diagonalize_future = diagonalize.submit(
        parameters,
        command_block_name=diagonalize_command_block_name,
        execution_profile_block_name=diagonalize_execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        wait_for=[train_generator_future],
    )
    energy_cr, energy_rn = diagonalize_future.result()

    if parameters.dmrg:
        dmrg_energy = dmrg_future.result()
        logger.info('DMRG energy: %f', dmrg_energy)
    logger.info('SKQD energy (random bit flips): %f', energy_rn)
    logger.info('SKQD energy (full conf. recovery): %f', energy_cr)

    return energy_cr


@task
async def dmrg(
    parameters: Parameters,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
) -> float:
    """Run DMRG and MPS sampling."""
    logger = get_run_logger()
    logger.info('Estimating ground-state energy via DMRG')

    parameters_path = Path(parameters.pkgpath) / "parameters.json"

    def dmrg_fn():
        async def run_dmrg_job():
            await _run_hpc_script_job(
                command_block_name=command_block_name,
                execution_profile_block_name=execution_profile_block_name,
                hpc_profile_block_name=hpc_profile_block_name,
                user_args=[str(parameters_path)],
                script_filename=DEFAULT_SCRIPT_FILENAMES["dmrg"],
                metrics_artifact_key=DEFAULT_METRICS_ARTIFACT_KEYS["dmrg"],
                task_label="skqd_z2lgt dmrg",
            )
            dmrg_path = Path(parameters.pkgpath) / "dmrg.h5"
            with h5py.File(dmrg_path, "r", libver="latest") as source:
                return (
                    source["energy"][()],
                    source["mps_states"][()],
                    source["mps_probs"][()],
                )

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(lambda: asyncio.run(run_dmrg_job())).result()

    return dmrg_flow(parameters, dmrg_fn, logger)


@task
async def sample_quantum(
    parameters: Parameters,
    runtime_name: str = DEFAULT_RUNTIME_NAME,
    option_name: str = DEFAULT_OPTIONS_VARIABLE_NAME,
):
    """Run the circuits on a backend and return the sampler results.

    Args:
        parameters: Workflow parameters.
        runtime_name: Name of QuantumRuntime block.
        option_name: Name of the Variable storing sampler primitive options.
    """
    logger = get_run_logger()
    logger.info('Sampling Trotter circuit final state bitstrings')

    resolved_option_name = parameters.runtime.options_name or option_name

    async with asyncio.TaskGroup() as tg:
        runtime_task = tg.create_task(QuantumRuntime.load(runtime_name))
        if resolved_option_name:
            options_task = tg.create_task(Variable.get(resolved_option_name))
        else:
            options_task = None

    runtime = runtime_task.result()
    options = (
        options_task.result() if options_task is not None else None
    ) or dict(parameters.runtime.options)
    task_id = task_run.id

    def fetch_result_fn():
        def fn():
            job = PrimitiveJobRun(job_id=parameters.runtime.job_id, credentials=runtime.credentials)
            return asyncio.run(job.fetch_result())

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(fn).result()

    # Cannot pass runtime.get_target directly as get_target_fn to sample_quantum_flow because of
    # async_dispatch (get_target would be called in a thread running an event loop and will
    # therefore return a coroutine)
    def get_target_fn():
        def fn():
            return runtime.get_target()

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(fn).result()

    def sample_fn(pubs):
        def fn(pubs):
            return runtime.sampler(sampler_pubs=pubs, options=options)

        with ThreadPoolExecutor(1) as executor:
            pub_result = executor.submit(fn, pubs).result()

        prefect_client = get_client(sync_client=True)
        artifacts = prefect_client.read_artifacts(
            limit=1,
            artifact_filter=ArtifactFilter(
                key=ArtifactFilterKey(any_=['job-metrics']),
                task_run_id=ArtifactFilterTaskRunId(any_=[task_id])
            )
        )
        if artifacts:
            keys, values = json.loads(artifacts[0].data)
            job_id = values[keys.index('job_id')]
        else:
            job_id = ''

        return pub_result, job_id

    sample_quantum_flow(parameters, fetch_result_fn, get_target_fn, sample_fn, logger)


@task
async def preprocess(
    parameters: Parameters,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
):
    """Correct the link-state bitstrings with MWPM and convert to plaquette-state bitstrings.

    Args:
        parameters: Configuration parameters.
        command_block_name: Name of the CommandBlock that executes preprocess.py.
    """
    logger = get_run_logger()
    logger.info('Correcting and converting link states to plaquette states')

    async def run_preprocess_job(task_specs):
        arguments = [
                parameters.pkgpath,
                '--mpi',
                '--etype', ','.join(task[0] for task in task_specs),
                '--idt', ','.join(str(task[1]) for task in task_specs),
                '--ikrylov', ','.join(str(task[2]) for task in task_specs)
            ]
        await _run_hpc_script_job(
            command_block_name=command_block_name,
            execution_profile_block_name=execution_profile_block_name,
            hpc_profile_block_name=hpc_profile_block_name,
            user_args=arguments,
            script_filename=DEFAULT_SCRIPT_FILENAMES["preprocess"],
            metrics_artifact_key=DEFAULT_METRICS_ARTIFACT_KEYS["preprocess"],
            task_label="skqd_z2lgt preprocess",
            execution_profile_overrides={
                "num_nodes": len(task_specs),
                "mpiprocs": 1,
                "walltime": "00:10:00",
            },
        )

    def convert_fn(task_specs):
        with ThreadPoolExecutor(1) as executor:
            executor.submit(lambda: asyncio.run(run_preprocess_job(task_specs))).result()

    preprocess_flow(parameters, convert_fn, logger=logger)


@task
async def train_generator(
    parameters: Parameters,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
):
    """Train a CRBM per Trotter step.

    Args:
        parameters: Configuration parameters.
        command_block_name: Name of the CommandBlock that executes train_generator.py.
    """
    logger = get_run_logger()
    logger.info('Training conditional restricted Boltzmann machines')

    async def run_train_job(task_specs):
        arguments = [
                parameters.pkgpath,
                '--mpi',
                '--idt', ','.join(str(task[0]) for task in task_specs),
                '--ikrylov', ','.join(str(task[1]) for task in task_specs)
            ]
        await _run_hpc_script_job(
            command_block_name=command_block_name,
            execution_profile_block_name=execution_profile_block_name,
            hpc_profile_block_name=hpc_profile_block_name,
            user_args=arguments,
            script_filename=DEFAULT_SCRIPT_FILENAMES["train"],
            metrics_artifact_key=DEFAULT_METRICS_ARTIFACT_KEYS["train"],
            task_label="skqd_z2lgt train_generator",
            execution_profile_overrides={
                "num_nodes": len(task_specs),
                "mpiprocs": 1,
                "walltime": "01:00:00",
            },
        )

    def train_fn(task_specs):
        with ThreadPoolExecutor(1) as executor:
            return executor.submit(lambda: asyncio.run(run_train_job(task_specs))).result()

    train_generator_flow(parameters, train_fn, logger=logger)


@task
async def diagonalize(
    parameters: Parameters,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
) -> tuple[float, float]:
    """Perform SQD with iterative configuration recovery.

    Args:
        parameters: Configuration parameters.
        command_block_name: Name of the CommandBlock that executes diagonalize.py.
    """
    logger = get_run_logger()

    gen_modes = []
    energies = []
    for gen_mode in ['cr', 'rn']:
        if (res := check_saved_result(parameters, f'skqd_{gen_mode}')) is None:
            gen_modes.append(gen_mode)
        else:
            energies.append(res[1])

    if not gen_modes:
        logger.info('All SQD results found on disk.')
        return tuple(energies)

    await _run_hpc_script_job(
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        user_args=[
            parameters.pkgpath,
            '--mpi',
            '--mode', ','.join(gen_modes)
        ],
        script_filename=DEFAULT_SCRIPT_FILENAMES["diagonalize"],
        metrics_artifact_key=DEFAULT_METRICS_ARTIFACT_KEYS["diagonalize"],
        task_label="skqd_z2lgt diagonalize",
        execution_profile_overrides={
            "num_nodes": len(gen_modes),
            "mpiprocs": 1,
            "walltime": "02:00:00",
        },
    )

    energies = []
    for gen_mode in ['cr', 'rn']:
        res = check_saved_result(parameters, f'skqd_{gen_mode}')
        energies.append(res[1])
    return tuple(energies)


def deploy():
    """Deploy workflow with a local worker."""
    # Prefect deploys with relative path.
    # Workflow is now installed in site-packages.
    os.chdir(Path(__file__).parent)

    # Deploy the workflow with specified options.
    skqd_z2lgt.with_options(version=os.getenv("WF_VERSION", "unknown")).serve(
        name="skqd_z2lgt",
        description="SKQD experiment for Z2 LGT."
    )


if __name__ == '__main__':
    from argparse import ArgumentParser
    import yaml

    parser = ArgumentParser(prog='skqd_z2lgt')
    parser.add_argument('parameters', metavar='PATH',
                        help='Path to a yaml file containing the workflow parameters.')
    parser.add_argument('--runtime-name', default=DEFAULT_RUNTIME_NAME,
                        help='Prefect QuantumRuntime block name.')
    parser.add_argument('--option-name', default=DEFAULT_OPTIONS_VARIABLE_NAME,
                        help='Prefect Variable name for runtime sampler options.')
    parser.add_argument('--root-dir',
                        help='Shared filesystem root directory used when parameters.pkgpath is empty.')
    parser.add_argument('--log-level', metavar='LEVEL', default='INFO', help='Logging level.')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format='%(asctime)s:%(name)s:%(levelname)s %(message)s')

    if os.path.isdir(args.parameters):
        with open(Path(args.parameters) / 'parameters.json', 'r', encoding='utf-8') as src:
            params = Parameters.model_validate_json(src.read())
    else:
        with open(args.parameters, 'r', encoding='utf-8') as src:
            params = Parameters(**yaml.load(src, yaml.Loader))

    asyncio.run(
        skqd_z2lgt(
            params,
            runtime_name=args.runtime_name,
            option_name=args.option_name,
            root_dir=args.root_dir,
        )
    )
