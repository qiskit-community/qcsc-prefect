from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qcsc_prefect_adapters.base.jinja_env import make_env
from qcsc_prefect_core.models.execution_profile import ExecutionProfile

_ENV = make_env("qcsc_prefect_adapters.slurm")
_TEMPLATE = "batch.slurm.j2"


@dataclass(frozen=True)
class SlurmJobRequest:
    """Target-specific request fields required to build a Slurm batch job."""

    partition: str
    executable: str
    account: str | None = None
    qpu: str | None = None


def to_slurm_template_kwargs(*, exec_profile: ExecutionProfile, req: SlurmJobRequest) -> dict:
    """Build template variables for the Slurm job script."""

    kw: dict = {
        "partition": req.partition,
        "executable": req.executable,
        "num_nodes": exec_profile.num_nodes,
        "launcher": exec_profile.launcher,
    }
    if req.account:
        kw["account"] = req.account
    if req.qpu:
        kw["qpu"] = req.qpu
    if exec_profile.mpiprocs is not None:
        kw["mpiprocs"] = exec_profile.mpiprocs
    if exec_profile.ompthreads is not None:
        kw["ompthreads"] = exec_profile.ompthreads
    if exec_profile.walltime is not None:
        kw["walltime"] = exec_profile.walltime
    if exec_profile.modules:
        kw["modules"] = list(exec_profile.modules)
    if exec_profile.pre_commands:
        kw["pre_commands"] = list(exec_profile.pre_commands)
    if exec_profile.environments:
        kw["environments"] = dict(exec_profile.environments)
    if exec_profile.mpi_options:
        kw["mpi_options"] = list(exec_profile.mpi_options)
    if exec_profile.arguments:
        kw["arguments"] = list(exec_profile.arguments)
    return kw


def render_script(*, work_dir: Path, exec_profile: ExecutionProfile, req: SlurmJobRequest) -> str:
    """Render a Slurm job script text from the configured Jinja template."""

    template = _ENV.get_template(_TEMPLATE)
    kwargs = to_slurm_template_kwargs(exec_profile=exec_profile, req=req)
    return template.render(work_dir=str(work_dir), **kwargs)


def write_script_file(*, work_dir: Path, filename: str, text: str) -> Path:
    """Write a rendered job script into the work directory."""

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / filename
    path.write_text(text)
    return path
