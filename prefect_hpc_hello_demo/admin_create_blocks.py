from prefect_hpc_demo.blocks import CommandBlock, ExecutionProfileBlock, MiyabiHPCProfileBlock

CommandBlock(
    command_name="mpi-hello",
    executable_key="hello_mpi",
    description="MPI Hello World demo (prints rank/size/hostname)",
).save("cmd-mpi-hello", overwrite=True)

ExecutionProfileBlock(
    profile_name="hello-n2",
    command_name="mpi-hello",
    resource_class="cpu",
    nodes=2,
    walltime="00:05:00",
    ranks_per_node=4,
    threads_per_rank=1,
    launcher="mpiexec.hydra",
    modules=None,  
).save("exec-hello-n2", overwrite=True)

ExecutionProfileBlock(
    profile_name="hello-n8",
    command_name="mpi-hello",
    resource_class="cpu",
    nodes=8,
    walltime="00:05:00",
    ranks_per_node=4,
    threads_per_rank=1,
    launcher="mpiexec.hydra",
).save("exec-hello-n8", overwrite=True)

ExecutionProfileBlock(
    profile_name="hello-n1",
    command_name="mpi-hello",
    resource_class="cpu",
    nodes=1,
    walltime="00:05:00",
    ranks_per_node=1,
    threads_per_rank=1,
    launcher="mpiexec.hydra",
).save("exec-hello-n1", overwrite=True)

MiyabiHPCProfileBlock(
    queue_cpu="regular-c",
    queue_gpu="regular-g",
    project_cpu=None,      
    module_init=None,      
    spack_setup=None,
    executable_map={
        "hello_mpi": "/work/gz09/z30541/hpc-execution-profiles/prefect_hpc_hello_demo/src/hello_mpi",
    },
).save("hpc-miyabi", overwrite=True)

print("Blocks created: cmd-mpi-hello, exec-hello-n2, exec-hello-n8, hpc-miyabi")
