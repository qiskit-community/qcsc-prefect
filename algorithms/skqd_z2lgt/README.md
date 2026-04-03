# Sample-based Krylov Quantum Diagonalization of Two-Dimensional Z2 Lattice Gauge Theory

This package provides an implementation of sample-based Krylov quantum diagonalization (SKQD)
applied to pure two-dimensional Z2 lattice gauge theory (LGT) problems. Samples are drawn from
Trotterized time evolution of the LGT model, and its Hamiltonian is projected and diagonalized onto
the sub-Hilbert space spanned by the sampled bitstrings.

## Getting Started

Before starting, make sure:

- You have completed [How to Set Up IBM Quantum Access Credentials for Prefect](../../docs/howto/howto_setup_prefect_qiskit.md).
- You have completed [How to Set Up the MDX Workflow Server for QCSC Execution](../../docs/howto/howto_setup_mdx_server.md).

`skqd_z2lgt` uses Prefect for orchestration and submits CPU/GPU Python script jobs through the
current `qcsc-prefect` block/executor stack. In a typical Miyabi setup you will prepare two Python
environments:

- a CPU-side environment for DMRG and preprocessing
- a GPU-side environment for CRBM training and diagonalization

Both environments must have `algorithms/skqd_z2lgt` installed, because the batch jobs execute
`python -m skqd_z2lgt.tasks.*`.

From the repository root, install the local qcsc-prefect packages and the workflow package:

```bash
cd /path/to/qcsc-prefect

uv pip install --no-deps \
  -e packages/qcsc-prefect-core \
  -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-blocks \
  -e packages/qcsc-prefect-executor

uv pip install -e algorithms/skqd_z2lgt
```

## Create Blocks

Create the HPC blocks and the Prefect variable that stores runtime options:

```bash
cp algorithms/skqd_z2lgt/skqd_z2lgt_blocks.example.toml \
  algorithms/skqd_z2lgt/skqd_z2lgt_blocks.toml

python algorithms/skqd_z2lgt/create_blocks.py \
  --config algorithms/skqd_z2lgt/skqd_z2lgt_blocks.toml
```

The example TOML expects:

- `python_cpu`: absolute path to the CPU-side Python executable
- `python_gpu`: absolute path to the GPU-side Python executable
- `project`, `queue_cpu`, `queue_gpu`: Miyabi project/queue settings

If you want to override the sampler options variable directly, set `runtime_options` in the TOML
file or pass `--runtime-options-json`.

## Deploy

Deploy the workflow in a Prefect server with:

```bash
skqd-z2lgt-deploy
```

## Run Notes

When running the flow, either:

- set `parameters.pkgpath` to a shared filesystem directory, or
- pass the flow argument `root_dir` so the workflow can create a per-run output directory

The sub-jobs read and write intermediate files under `parameters.pkgpath`, so this path must be
visible from the nodes that execute the CPU/GPU batch jobs.
