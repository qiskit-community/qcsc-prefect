# Fugaku Prefect Block Hello Demo

This example uses `run_fugaku_job` to submit a real `pjsub` job from Prefect Blocks.

## Prerequisites

- Run on a Fugaku login node
- `pjsub` and `pjstat` are available in `PATH`
- Prefect API is reachable

## 1) Sync dependencies

```bash
cd /Users/hitomi/Project/qcsc-prefect
uv sync
```

## 2) Register block types

```bash
cd /Users/hitomi/Project/qcsc-prefect
uv run prefect block register -m qcsc_prefect_blocks.miyabi.blocks
```

Note: The blocks are currently defined in `qcsc_prefect_blocks.miyabi.blocks` but work for both Miyabi and Fugaku.

## 3) Create demo Blocks

```bash
cd /Users/hitomi/Project/qcsc-prefect
export FUGAKU_GROUP=your_group_id
export FUGAKU_RSCGRP=small
export FUGAKU_GFSCACHE=/vol0002
uv run python examples/fugaku_prefect_hello_demo/create_blocks.py
```

This script creates:
- `cmd-fugaku-hello-demo`
- `exec-fugaku-hello-single`
- `hpc-fugaku` (instance of `HPCProfileBlock`)

To use an executable other than `hello_demo.sh`:

```bash
export FUGAKU_DEMO_EXECUTABLE=/path/to/your/executable
```

## 4) Run the flow

```bash
cd /Users/hitomi/Project/qcsc-prefect
uv run python -c "import asyncio; from examples.fugaku_prefect_hello_demo.flow import fugaku_prefect_block_hello_flow; print(asyncio.run(fugaku_prefect_block_hello_flow()))"
```

Example return value:

```text
{'job_id': '12345', 'exit_status': 0, 'state': 'EXT', 'work_dir': '/work/.../fugaku_prefect_block_hello'}
```

## Environment Variables

- `FUGAKU_GROUP`: Preferred. Your Fugaku Group ID (e.g., `ra000000`)
- `FUGAKU_PROJECT`: Legacy alias still accepted for compatibility
- `FUGAKU_RSCGRP`: Resource group (default: `small`). Options: `small`, `large`, `huge`, etc.
- `FUGAKU_GFSCACHE`: GFS cache path (default: `/vol0002`). Used for `PJM_LLIO_GFSCACHE`
- `FUGAKU_DEMO_EXECUTABLE`: Path to executable (default: `hello_demo.sh` in this directory)

## Notes

- The `PJM_LLIO_GFSCACHE` directive is automatically added to the job script when `gfscache` is specified
- Job output files are named: `{script_basename}.{job_name}.out`, `.err`, `.stats`
- The demo uses a single node with `launcher="single"` for simplicity
