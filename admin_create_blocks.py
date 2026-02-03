"""
Admin helper to create demo Blocks in Prefect.

Edit the executable path and optional queue/project to match Miyabi.
"""
from prefect_hpc_demo.blocks import CommandBlock, ExecutionProfileBlock, MiyabiHPCProfileBlock

CommandBlock(
    command_name="diag",
    executable_key="diag",
    description="Demo diag command",
).save("cmd-diag", overwrite=True)

ExecutionProfileBlock(
    profile_name="diag-n16",
    command_name="diag",
    resource_class="cpu",
    nodes=16,
    walltime="01:00:00",
    ranks_per_node=4,
    threads_per_rank=2,
    launcher="mpirun",
).save("exec-diag-n16", overwrite=True)

ExecutionProfileBlock(
    profile_name="diag-n2",
    command_name="diag",
    resource_class="cpu",
    nodes=2,
    walltime="00:20:00",
    ranks_per_node=4,
    threads_per_rank=2,
    launcher="mpirun",
).save("exec-diag-n2", overwrite=True)

MiyabiHPCProfileBlock(
    queue_cpu="cpu",
    queue_gpu="gpu",
    project_cpu=None,
    executable_map={
        "diag": "/path/to/your/diag",
    },
).save("hpc-miyabi", overwrite=True)

print("Blocks created: cmd-diag, exec-diag-n16, exec-diag-n2, hpc-miyabi")
