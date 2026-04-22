# How to Test Slurm Locally with `slurm-docker-cluster`

This guide explains how to spin up a small Slurm cluster locally with
[`giovtorres/slurm-docker-cluster`](https://github.com/giovtorres/slurm-docker-cluster)
and use it to smoke-test the Slurm support in `qcsc-prefect`.

The intended use case is:

- one Docker host
- one Slurm controller container
- multiple Slurm compute-node containers (`c1`, `c2`, `c3`, ...)
- testing `sbatch`, `sacct`, `scancel`, and `srun` behavior locally

> [!IMPORTANT]
> This setup is "single Docker host, multiple Slurm nodes".
> It is very useful for local development, but it is not the same as a real
> multi-machine cluster.

## What this guide does

This guide covers:

1. starting a local Slurm cluster
2. copying `qcsc-prefect` into the Slurm controller container
3. installing the local Python packages there
4. running a small `run_slurm_job()` smoke test
5. verifying that a multi-node job actually ran on multiple worker containers

## Prerequisites

- Docker
- Docker Compose / `docker compose`
- `make`
- a local checkout of this repository

The upstream project's quick start and scaling notes are here:

- [Upstream README](https://github.com/giovtorres/slurm-docker-cluster)
- [Upstream `docker-compose.yml`](https://raw.githubusercontent.com/giovtorres/slurm-docker-cluster/main/docker-compose.yml)
- [Upstream `Dockerfile`](https://github.com/giovtorres/slurm-docker-cluster/blob/main/Dockerfile)

---

## Step 1. Start a local Slurm cluster

Clone the upstream repository:

```bash
git clone https://github.com/giovtorres/slurm-docker-cluster.git
cd slurm-docker-cluster
cp .env.example .env
```

Set the CPU worker count to 3 so that multi-node jobs have multiple targets:

```bash
perl -0pi -e 's/^CPU_WORKER_COUNT=.*/CPU_WORKER_COUNT=3/m' .env
```

Pull the prebuilt image and tag it with the version expected by the compose file:

```bash
docker pull giovtorres/slurm-docker-cluster:latest
docker tag giovtorres/slurm-docker-cluster:latest slurm-docker-cluster:25.11.4
```

Start the cluster:

```bash
make up
make status
```

> [!NOTE]
> If `sinfo` later shows only `c1` even though you expect multiple CPU workers,
> explicitly scale the worker service from the host machine:
>
> ```bash
> cd slurm-docker-cluster
> make scale-cpu-workers N=2
> make status
> ```
>
> This can happen if `.env` was updated after an earlier startup and the running
> Compose application still has only one CPU worker.

Open a shell in the controller:

```bash
make shell
```

Inside the controller, confirm that Slurm sees the worker nodes:

```bash
sinfo
```

Remember the partition name shown in the first column, such as `cpu` or
`normal`. You will use that exact partition name in the smoke test below.

Example output:

```text
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
normal*      up 5-00:00:00      3   idle c[1-3]
```

---

## Step 2. Copy `qcsc-prefect` into the controller

From your host machine, copy this repository into the `slurmctld` container:

```bash
docker cp /Users/hitomi/Project/qcsc-prefect/. slurmctld:/data/qcsc-prefect
```

If your local checkout lives elsewhere, replace the source path accordingly.

---

## Step 3. Install `qcsc-prefect` in the controller

Open a shell in the controller if you are not already inside it:

```bash
docker exec -it slurmctld bash
```

Move to the copied repository:

```bash
cd /data/qcsc-prefect
```

The upstream image includes `python3.12`, but `pip` may not be initialized yet.
Enable it, create a virtual environment, and install the local packages:

```bash
python3 -m ensurepip --upgrade
mkdir -p .venv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install \
  -e packages/qcsc-prefect-core \
  -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-blocks \
  -e packages/qcsc-prefect-executor \
  pytest
```

---

## Step 4. Run a Slurm smoke test

Run the following script inside the controller:

```bash
cd /data/qcsc-prefect
. .venv/bin/activate

python3 - <<'PY'
import asyncio
from pathlib import Path

from qcsc_prefect_adapters.slurm.builder import SlurmJobRequest
from qcsc_prefect_core.models.execution_profile import ExecutionProfile
from qcsc_prefect_executor.slurm import run as run_mod


class Logger:
    def info(self, msg):
        print(msg)

    def error(self, msg):
        print(msg)


async def fake_create_table_artifact(*, table, key):
    print("artifact key:", key)
    print(table)


run_mod.get_run_logger = lambda: Logger()
run_mod.create_table_artifact = fake_create_table_artifact

work_dir = Path("/data/qcsc-prefect/.slurm-test")
work_dir.mkdir(parents=True, exist_ok=True)

exe = work_dir / "hello.sh"
exe.write_text("#!/bin/sh\necho slurm-integration-ok\nhostname\n")
exe.chmod(0o755)

profile = ExecutionProfile(
    command_key="slurm-integration",
    num_nodes=2,
    mpiprocs=1,
    launcher="srun",
    walltime="00:05:00",
)
# Use the partition name reported by `sinfo` on your cluster.
# In many setups this is `cpu`; in others it may be `normal`.
req = SlurmJobRequest(
    partition="cpu",
    account=None,
    executable=str(exe),
)

result = asyncio.run(
    run_mod.run_slurm_job(
        work_dir=work_dir,
        script_filename="integration_job.slurm",
        exec_profile=profile,
        req=req,
        watch_poll_interval=2.0,
        timeout_seconds=120,
        metrics_artifact_key="slurm-integration-metrics",
    )
)

print(result)
PY
```

What this test checks:

- `sbatch` can submit a job
- `sacct` can observe final job state
- `srun` is used as the in-job launcher
- stdout and stderr files are collected
- the job can run across multiple compute-node containers

---

## Step 5. Verify the output

Inspect the generated files:

```bash
cd /data/qcsc-prefect/.slurm-test
ls -l
cat output.out
cat output.err
```

Expected `output.out` content should include:

- `slurm-integration-ok`
- two hostnames such as `c1` and `c2`

If you requested `num_nodes=2`, seeing two different worker names is the simplest
proof that the job really executed across multiple Slurm nodes.

You can also inspect Slurm directly:

```bash
squeue
sacct
```

---

## Step 6. Scale the worker count if needed

From the host machine:

```bash
cd slurm-docker-cluster
make scale-cpu-workers N=4
make status
```

If you expected 2 CPU workers but `sinfo` still shows only `c1`, run
`make scale-cpu-workers N=2` first to reconcile the active worker count with
your intended setup.

Then rerun the smoke test with a larger `num_nodes` value if needed.

---

## Next step

If you want to run the BitCount demo on top of this local Slurm cluster, follow
[`docs/tutorials/create_qcsc_workflow_for_local_slurm.md`](../tutorials/create_qcsc_workflow_for_local_slurm.md).

---

## Optional: rerun local unit tests inside the controller

If you want to run the current local Slurm unit tests inside the controller:

```bash
cd /data/qcsc-prefect
. .venv/bin/activate
pytest \
  packages/qcsc-prefect-adapters/tests/test_slurm_builder.py \
  packages/qcsc-prefect-adapters/tests/test_slurm_runtime.py \
  packages/qcsc-prefect-executor/tests/test_run_slurm_job_local.py
```

These tests are still mocked tests.
They are useful for basic regression checking, but they do not replace the real
Slurm smoke test above.

---

## Future improvement

The current repository does not yet have a real Slurm integration test file
equivalent to:

- `packages/qcsc-prefect-executor/tests/test_run_miyabi_job_miyabi_integration.py`
- `packages/qcsc-prefect-executor/tests/test_run_fugaku_job_fugaku_integration.py`

If you want repeatable CI-style testing later, the next step is to add:

- `packages/qcsc-prefect-executor/tests/test_run_slurm_job_slurm_integration.py`

and run it inside the `slurmctld` container or a dedicated Slurm-enabled test environment.
