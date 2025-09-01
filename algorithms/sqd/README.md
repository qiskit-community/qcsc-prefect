# SQD Experiment with DICE Solver

This package provides the SQD experiment demonstrated in the paper [Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric Superomputer](https://arxiv.org/abs/2405.05068).

The experiment requires a [custom DICE solver](https://github.com/caleb-johnson/Dice) to diagonalize a molecular Hamiltonian using MPI. This package also includes a Prefect integration via the `DiceSHCISolverJob` block, which is a Prefect block document. This block implements the `.run` method that accepts molecular properties and returns an `SCIResult` object from [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd), enabling seamless integration of MPI-enabled diagonalization into your quantum chemistry workflow.

---

## Getting Started

### 1. Install DICE Solver

First, load `openmpi` or an equivalent module to enable a C++ compiler with MPI support (e.g., `mpicxx`):

```bash
module load mpi/openmpi-x86_64
```

If the module is not available (`module avail`), install it manually. The module name may vary depending on your environment.

After loading the module, build the solver:

```bash
sh build_dice.sh
```

This process may take some time. Upon successful build, the `bin` directory will contain the `Dice` binary:

```bash
ls -l bin
```

Example output:

```
-rwxr-xr-x. 1 prefectuser prefectuser 47132672 Aug 31 19:07 Dice
-rwxr-xr-x. 1 prefectuser prefectuser   227216 Aug 31 19:07 libboost_mpi.so
-rwxr-xr-x. 1 prefectuser prefectuser   227216 Aug 31 19:07 libboost_mpi.so.1.85.0
-rwxr-xr-x. 1 prefectuser prefectuser   479816 Aug 31 19:07 libboost_serialization.so
-rwxr-xr-x. 1 prefectuser prefectuser   479816 Aug 31 19:07 libboost_serialization.so.1.85.0
-rwxr-xr-x. 1 prefectuser prefectuser   334912 Aug 31 19:07 libboost_wserialization.so
-rwxr-xr-x. 1 prefectuser prefectuser   334912 Aug 31 19:07 libboost_wserialization.so.1.85.0
```

Remember the absolute path of the executable for later use in the Prefect block:

```bash
realpath bin/Dice
```

Example output:

```
/home/prefectuser/qii-miyabi-kawasaki/algorithms/sqd/bin/Dice
```

### 2. Set Up Python Environment

Create and activate a virtual environment:

```bash
uv venv -p 3.12 && source .venv/bin/activate
```

Install the project in editable mode:

```bash
uv pip install -e .
```

Verify installation:

```bash
uv pip list | grep qii-miyabi-kawasaki
```

Example output:

```
prefect-miyabi            0.1.0       /home/prefectuser/qii-miyabi-kawasaki/framework/prefect-miyabi
sqd-dice                  0.1.0       /home/prefectuser/qii-miyabi-kawasaki/algorithms/sqd
```

> [!NOTE]
> `prefect-miyabi` is an upstream dependency that provides core functionality for integrating the DICE solver into the Miyabi supercomputer environment.

### 3. Configure Prefect Server

In the root directory of this package, you'll find a `prefect.toml` file specifying the Prefect server endpoint. Check the configuration:

```bash
prefect config view
```

Example output:

```
🚀 you are connected to:
https://qii-kawasaki-miyabi-serv.cspp.cc.u-tokyo.ac.jp/prefect
PREFECT_PROFILE='ephemeral'
PREFECT_API_URL='https://qii-kawasaki-miyabi-serv.cspp.cc.u-tokyo.ac.jp/prefect/api' (from prefect.toml)
PREFECT_SERVER_ALLOW_EPHEMERAL_MODE='False' (from prefect.toml)
```

By default, the endpoint connects to `qii-kawasaki-miyabi-serv`, a virtual machine hosted by [mdx](https://mdx.jp/). Modify `PREFECT_API_URL` to connect to your own Prefect server if needed.

Register the data schema for dependency blocks so they can be configured via the Prefect console GUI:

```bash
prefect block register -m prefect_qiskit
prefect block register -m prefect_qiskit.vendors
prefect block register -m sqd_dice.dice_job
```

Refer to the [Prefect Qiskit tutorial](https://qiskit-community.github.io/prefect-qiskit/tutorials/01_getting_started/) for guidance on setting up the `QuantumRuntime` block for primitive executions.

Similarly, configure the `DiceSHCISolverJob` block to specify the path to the `Dice` executable and the working directory. In the Miyabi environment, both must reside in shared storage (e.g., `/work`) so compute nodes can access them.

Ensure the following environment variable is set:

```yaml
{"LD_LIBRARY_PATH": "path/to/bin:$LD_LIBRARY_PATH"}
```

To run mpi launchers, make sure the necessary module is loaded:

```yaml
["mpi/openmpi-x86_64"]
```

Use the following block names:

- `QuantumRuntime`: `"sqd-runner-{$USER}"`
- `DiceSHCISolverJob`: `"sqd-solver-{$USER}"`

where `{$USER}` indicates a login user name, e.g. `sqd-runner-prefectuser`.
Optionally, you can define primitive execution options using the Prefect Variable `sampler_options`.
If you use the `local` executor, please read the [Special Tips for Local Shell](#special-tips-for-local-shell) carefully.

Once all blocks are configured, deploy the workflow and trigger it from the Prefect console GUI:

```bash
sqd-deploy
```

Example output:

```
Your flow 'sqd-2405-05068' is being served and polling for scheduled runs!

To trigger a run for this flow, use the following command:

        $ prefect deployment run 'sqd-2405-05068/sqd_2405_05068'

You can also run your flow via the Prefect UI: ...
```

Ensure the deployed URL matches your Prefect server endpoint.

---

## Special Tips for Local Shell

When running experiments in a local shell (e.g., on a virtual machine or laptop), there are several important considerations:

### ⚠️ Avoiding Race Conditions in Local Execution

If you set `sqd_num_batches` (Batch Number) > 1, the executors of `DiceSHCISolverJob` may invoke `mpirun` asynchronously, resulting in near-simultaneous launches. This can lead to race conditions during MPI initialization, potentially causing errors like:
```
terminate called after throwing an instance of 'std::out_of_range'
  what():  vector::_M_range_check: __n (which is 22) >= this->size() (which is 22)
```
To avoid this, you can limit the number of concurrently running jobs using Prefect's concurrency control:
```bash
prefect concurrency-limit create "res: local" 1
```
This forces local jobs to run sequentially, which may reduce performance but is acceptable for code validation. For high-performance execution, use the `pbs` executor on the Miyabi environment.

### 🛠️ Preventing Segmentation Faults from UCX

By default, `mpirun` may use the UCX (Unified Communication X) backend for MPI communication. This can cause segmentation faults or bus errors when running multiple Dice jobs concurrently on a single node, especially in environments with shared memory or RDMA-enabled hardware.

To avoid these issues, explicitly disable UCX and use the TCP backend by setting the following environment variables:
```yaml
{
  "LD_LIBRARY_PATH": "path/to/bin:$LD_LIBRARY_PATH",
  "OMPI_MCA_pml": "ob1",
  "OMPI_MCA_btl": "tcp,self",
  "OMPI_MCA_btl_tcp_if_include": "lo"
}
```

**Explanation of Each Setting:**
- `OMPI_MCA_pml="ob1"`: Selects the basic and stable point-to-point messaging layer, avoiding UCX.
- `OMPI_MCA_btl="tcp,self"`: Uses TCP for inter-process communication and self for intra-process communication.
- `OMPI_MCA_btl_tcp_if_include="lo"`: Restricts TCP communication to the loopback interface, ideal for single-node testing.

These settings ensure mpirun uses a stable and isolated communication backend, making parallel execution reliable in local environments.

---

## Contribution Guidelines

This package serves as a Prefect-style reference implementation of the SQD experiment described in the publication. It is considered feature-complete and is now in the maintenance phase.

No further contributions are expected except for bug fixes and improvements to Prefect usage patterns. Workflow developers may use this experiment as a test vehicle for workflow technology research.
