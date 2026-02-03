# Prefect Miyabi Demo (HPC-agnostic) — Quick Start

This is a minimal demo that uses **Prefect Blocks** to separate:
- **Command** (WHAT to run)
- **Execution Profile** (baseline HOW to run)
- **HPC Profile** (Miyabi PBS specifics)

The flow stays HPC-agnostic and is controlled by run-time parameters.

## 1) Register block types

Run from the directory that contains this package:

```bash
prefect block register -m prefect_hpc_demo.blocks
```

## 2) Create Blocks (admin step)

Run the helper script and edit the executable path:

```bash
python admin_create_blocks.py
```

Or create blocks manually (see `admin_create_blocks.py`).

## 3) Run flow locally (generate script only)

```bash
python -c "from prefect_hpc_demo.flow_demo import miyabi_demo_flow; print(miyabi_demo_flow(submit=False))"
```

## 4) Pass tuning overrides

```python
from prefect_hpc_demo.flow_demo import miyabi_demo_flow
from prefect_hpc_demo.models import Tuning

miyabi_demo_flow(
    exec_profile_block_name="exec-diag-n16",
    tuning=Tuning(nodes=32, threads_per_rank=4),
    submit=False,
)
```

## 5) Submit to PBS (optional)

Set `submit=True` (requires `qsub`):

```python
miyabi_demo_flow(submit=True)
```

## Notes

- PBS resource syntax (`select`, `ncpus`) should be adjusted to Miyabi site policy.
- This prototype intentionally keeps validation simple.
- Next step: plug in your existing Miyabi executor and metrics collection.
