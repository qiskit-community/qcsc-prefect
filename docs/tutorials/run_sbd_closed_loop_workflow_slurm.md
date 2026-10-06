# Run SBD Closed-loop Workflow on Slurm (qcsc-prefect)

This tutorial walks through running a Sample-based Quantum
Diagonalization (SQD) closed-loop workflow with `qcsc-prefect` on a
Slurm-based HPC system.

The workflow combines quantum sampling and classical processing with an
SBD Davidson diagonalization job submitted through Slurm. Prefect
orchestrates the workflow and stores the reusable execution
configuration in Blocks and Variables.

The goal is to compute the ground-state energy of the N2-Mo state while
demonstrating how the same QCSC workflow can target a generic Slurm
environment.

## Prerequisites

Before starting, make sure:

-   You have access to a Slurm cluster and can submit jobs with
    `sbatch`.
-   An MPI C++ compiler wrapper such as `mpicxx` or `mpic++` is
    available.
-   OpenBLAS development headers and libraries are installed and
    linkable with `-lopenblas`.
-   Git and Python are available.
-   You have a Python environment with Prefect and the `qcsc-prefect`
    packages.
-   You have configured access to the Prefect server used for the
    workflow.
-   If you plan to use a real quantum device, configure the required IBM
    Quantum credentials/runtime Block.

> **Important**
>
> Slurm systems differ in partition names, account/project requirements,
> filesystem layout, MPI configuration, and available modules. Replace
> the example values in this tutorial with values appropriate for your
> cluster.

------------------------------------------------------------------------

## 0. What changes for Slurm?

The workflow architecture remains the same across HPC backends. The main
difference is how the classical SBD solver is built, configured, and
submitted.

The SBD solver uses the following reusable Blocks:

-   `CommandBlock` --- **what** executable to run.
-   `ExecutionProfileBlock` --- **how** to run it, including MPI
    launcher, process count, walltime, modules, and related execution
    settings.
-   `HPCProfileBlock` --- **where** to run it, including the HPC target,
    partition/queue, and project/account.
-   `SBDSolverJob` --- the SBD-specific wrapper that references the
    three Blocks above and stores solver-specific parameters.

For a Slurm target, `qcsc-prefect` generates a Slurm batch script and
submits the SBD computation through Slurm.

------------------------------------------------------------------------

## 1. Big picture: where the workflow runs

A typical Slurm setup contains three execution contexts:

-   **Workflow environment** --- runs the Prefect Flow and Python
    workflow logic.
-   **Slurm compute nodes** --- execute the compiled SBD `diag` binary.
-   **IBM Quantum** --- executes quantum sampling when
    `quantum_source="real-device"` is selected.

On systems where the workflow environment and Slurm login environment
are the same machine or share the same filesystem, the repository and
SBD executable can be accessed directly.

### Prefect concepts used by the workflow

#### Flow

The complete SQD experiment is represented as a Prefect Flow, including
quantum sampling, configuration recovery, SBD diagonalization,
iterations, and result collection.

#### Tasks

Individual stages include:

-   quantum sampling,
-   subsampling/configuration recovery,
-   Davidson diagonalization on Slurm,
-   result collection and artifact generation.

#### Blocks

Blocks store reusable configuration such as:

-   quantum credentials/runtime configuration,
-   executable path,
-   MPI execution settings,
-   Slurm partition/account settings,
-   SBD solver parameters.

#### Variables

Runtime options such as quantum sampler settings can be stored as
Prefect Variables.

#### Deployment

A deployment makes the Flow runnable by name from the Prefect UI or CLI.

------------------------------------------------------------------------

## 2. Tutorial steps

### Step 1. Clone the repository and activate the Python environment

Clone or enter your `qcsc-prefect` checkout:

``` bash
git clone https://github.com/qiskit-community/qcsc-prefect.git
cd qcsc-prefect
```

Activate the Python environment you use for `qcsc-prefect`.

For example:

``` bash
source /path/to/venv/bin/activate
```

Confirm that your Prefect profile points to the intended server:

``` bash
prefect profile ls
prefect config view
```

------------------------------------------------------------------------

### Step 2. Install the required packages

From the repository root, install the local packages:

``` bash
uv pip install --no-deps \
  -e packages/qcsc-prefect-core \
  -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-blocks \
  -e packages/qcsc-prefect-executor
```

Install the workflow utility and SBD packages:

``` bash
uv pip install -e algorithms/qcsc_workflow_utility
uv pip install -e algorithms/sbd
```

Check the installation:

``` bash
uv pip list | grep -E "(qcsc-prefect|sbd|qcsc)"
```

When developing from a local checkout, it is useful to confirm that
Python is importing the package from the checkout you intend to test.

For example:

``` bash
python -c "import qcsc_prefect_adapters; print(qcsc_prefect_adapters.__file__)"
```

------------------------------------------------------------------------

### Step 3. Build the SBD solver for the Slurm environment

The SBD solver is a native C++ executable. Build it in an environment
compatible with the Slurm compute nodes where it will run.

Navigate to the native solver directory:

``` bash
cd algorithms/sbd/native
```

Run the Slurm build script:

``` bash
bash build_sbd_slurm.sh
```

The script:

1.  selects `mpicxx` or `mpic++`,
2.  clones the upstream SBD repository if needed,
3.  checks out the SBD revision used by this integration,
4.  compiles `main.cc` with C++17, OpenMP, and optimization enabled,
5.  links the executable against OpenBLAS.

The upstream SBD revision is pinned by the build script so that the
build is reproducible.

After a successful build, you should see output similar to:

``` text
Build completed: /path/to/qcsc-prefect/algorithms/sbd/native/diag
```

Verify the executable:

``` bash
ls -lh diag
file diag
```

Record its absolute path:

``` bash
realpath diag
```

You will use this path as `sbd_executable` in the Slurm configuration.

> **Note**
>
> The default build expects OpenBLAS to be available through
> `-lopenblas`. Cluster-specific compiler or CPU optimization flags can
> be added when appropriate, but the portable build should be tested
> first.

------------------------------------------------------------------------

### Step 4. Create the Slurm SBD configuration

Return to the repository root:

``` bash
cd /path/to/qcsc-prefect
```

Create a working directory for SBD jobs. The directory must be
accessible from the Slurm compute nodes.

For example:

``` bash
mkdir -p /path/to/sbd_jobs
```

Copy the Slurm example configuration:

``` bash
cp algorithms/sbd/sbd_blocks.slurm.example.toml \
   algorithms/sbd/sbd_blocks.toml
```

Edit the local configuration:

``` bash
vim algorithms/sbd/sbd_blocks.toml
```

At minimum, configure:

  ------------------------------------------------------------------------------------------------------
  Parameter               Example                                                Description
  ----------------------- ------------------------------------------------------ -----------------------
  `hpc_target`            `"slurm"`                                              Selects the Slurm
                                                                                 backend

  `project`               `"default"`                                            Slurm account/project,
                                                                                 if required

  `queue`                 `"normal"`                                             Slurm partition

  `work_dir`              `"/path/to/sbd_jobs"`                                  Shared job working
                                                                                 directory

  `sbd_executable`        `"/path/to/qcsc-prefect/algorithms/sbd/native/diag"`   Absolute path to the
                                                                                 compiled solver
  ------------------------------------------------------------------------------------------------------

A minimal configuration looks like:

``` toml
hpc_target = "slurm"
project = "default"
queue = "normal"
work_dir = "/path/to/sbd_jobs"
sbd_executable = "/path/to/qcsc-prefect/algorithms/sbd/native/diag"

launcher = "srun"
walltime = "00:15:00"
num_nodes = 1
mpiprocs = 8
modules = []
mpi_options = []
```

The example configuration also exposes SBD solver parameters:

``` toml
task_comm_size = 1
adet_comm_size = 1
bdet_comm_size = 1

block = 4
iteration = 1
tolerance = 0.01
carryover_ratio = 0.1
solver_mode = "cpu"
```

For a quick functional test, the low-iteration settings in the example
configuration are recommended before scaling the calculation.

> **Important**
>
> `sbd_blocks.toml` is a local runtime configuration and may contain
> machine-specific paths and scheduler settings. Use
> `sbd_blocks.slurm.example.toml` as the shareable configuration
> template.

------------------------------------------------------------------------

### Step 5. Generate the Prefect Blocks

Run the block creation script using the Slurm configuration:

``` bash
python algorithms/sbd/create_blocks.py \
  --config algorithms/sbd/sbd_blocks.toml
```

The script creates the reusable configuration needed by `SBDSolverJob`,
including:

-   a `CommandBlock` for the `diag` executable,
-   an `ExecutionProfileBlock` for MPI/Slurm execution,
-   an `HPCProfileBlock` with `hpc_target="slurm"`,
-   an `SBDSolverJob`,
-   the SQD runtime options Variable.

If you use the block-name overrides shown in the Slurm example
configuration, the resulting names are:

``` text
CommandBlock:          cmd-sbd-diag
ExecutionProfileBlock: exec-sbd-mpi
HPCProfileBlock:       hpc-slurm-sbd
SBD Solver Job:        davidson-solver
Prefect Variable:      sqd_options
```

The `davidson-solver` Block is the workflow-facing solver preset.
Internally, it references the command, execution profile, and HPC
profile Blocks.

You can inspect the registered Blocks with Prefect:

``` bash
prefect block ls
```

------------------------------------------------------------------------

### Step 6. Understand the generated Slurm job

When the SBD solver runs, the Slurm adapter generates a batch script
containing scheduler directives derived from the Blocks.

Conceptually, the generated script resembles:

``` bash
#!/bin/bash
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --nodes=<nodes>
#SBATCH --ntasks-per-node=<mpi-processes>
#SBATCH --time=<walltime>
#SBATCH --output=<work-dir>/output.out
#SBATCH --error=<work-dir>/output.err

cd <work-dir>

srun /absolute/path/to/diag <solver-arguments>
```

Optional execution settings can also provide:

-   environment modules,
-   pre-run shell commands,
-   environment variables,
-   OpenMP thread counts,
-   additional MPI launcher options.

The generated job performs an executable preflight check before
launching the solver so that an invalid or inaccessible executable path
fails with a clear error.

------------------------------------------------------------------------

### Step 7. Deploy the SBD workflow

From the repository root, activate the Python environment and start the
SBD deployment:

``` bash
cd /path/to/qcsc-prefect
source /path/to/venv/bin/activate
sbd-deploy
```

For a long-running serving process, use the process-management mechanism
appropriate for your environment, such as `screen`, `tmux`, or a
service.

For example:

``` bash
screen -S sbd-workflow
sbd-deploy
```

Detach from `screen` with `<Ctrl-a>` followed by `d`.

The serving process must remain active so that it can pick up Flow Runs
created from the Prefect UI or CLI.

List deployments with:

``` bash
prefect deployment ls
```

------------------------------------------------------------------------

### Step 8. Provide workflow parameters

From the Prefect UI, select the SBD deployment and choose **Run → Custom
run**.

For an initial test, use parameters similar to:

  ---------------------------------------------------------------------------------------------------
  Field                               Value / Example
  ----------------------------------- ---------------------------------------------------------------
  FCIDump File                        `/path/to/qcsc-prefect/algorithms/sbd/data/fcidump_N2_MO.txt`

  SQD Subspace Dimension              `1000000`

  Differential Evolution Iterations   `1`

  Quantum Source                      `random` or `real-device`

  Random Seed                         `24`

  Solver Block Ref                    `sbd_solver_job/davidson-solver`
  ---------------------------------------------------------------------------------------------------

`Solver Block Ref` selects the `SBDSolverJob` preset used by the
workflow.

For the first Slurm integration test, `random` is useful because it
allows the classical workflow and Slurm execution path to be validated
independently of quantum-device access.

When `real-device` is selected, the workflow uses the configured quantum
runtime Block and submits the sampling workload to IBM Quantum.

------------------------------------------------------------------------

### Step 9. Execute and monitor the workflow

Submit the Flow Run from the Prefect UI.

The workflow will:

1.  obtain or generate quantum samples,
2.  perform configuration recovery,
3.  prepare the SBD input files,
4.  generate a Slurm batch script,
5.  submit the Davidson diagonalization job,
6.  wait for the Slurm job to finish,
7.  parse the SBD output,
8.  continue the SQD loop and record telemetry.

You can also monitor the classical job directly with Slurm:

``` bash
squeue -u "$USER"
```

Inspect the generated job directory and Slurm output files when
troubleshooting:

``` bash
ls -la /path/to/sbd_jobs
```

The generated job directories contain the solver inputs, Slurm script,
and output/error files associated with the SBD execution.

After the workflow completes, inspect the `sqd-telemetry` artifact in
Prefect. It contains intermediate energy information produced during the
SQD workflow.

For the N2 example, the final energy is expected to approach
approximately `-134.94` Hartree when using settings sufficient for
convergence.

------------------------------------------------------------------------

## 3. What happens when the workflow submits an SBD job?

The Slurm execution path keeps the scientific workflow independent of
cluster-specific scheduler configuration.

At runtime:

1.  The workflow loads the selected `SBDSolverJob`.
2.  `SBDSolverJob` resolves its `CommandBlock`, `ExecutionProfileBlock`,
    and `HPCProfileBlock`.
3.  The solver prepares files such as `fcidump.txt` and `AlphaDets.bin`.
4.  The executor resolves `hpc_target="slurm"`.
5.  A Slurm job request is constructed from the reusable Blocks.
6.  The Slurm adapter renders the `.slurm` batch script.
7.  The job is submitted to the configured partition/account.
8.  The workflow waits for the scheduler job to reach a final state.
9.  SBD output files are parsed and returned to the workflow.
10. Prefect stores the resulting SQD telemetry.

This separation allows the workflow logic to remain stable while
scheduler-specific settings are controlled through configuration and
Prefect Blocks.

------------------------------------------------------------------------

## 4. Troubleshooting

### `diag` does not exist

Rebuild the native solver:

``` bash
cd algorithms/sbd/native
bash build_sbd_slurm.sh
```

Then verify:

``` bash
ls -lh diag
file diag
```

### OpenBLAS cannot be linked

Verify that the OpenBLAS development package is installed and that the
linker can resolve:

``` text
-lopenblas
```

The exact installation procedure depends on the operating system and
cluster software environment.

### `sbatch` or `srun` is unavailable

The workflow must execute in an environment with access to the Slurm
client commands used to submit and monitor jobs.

Check:

``` bash
which sbatch
which srun
sinfo
```

### Slurm rejects the partition or account

Verify the values configured as:

``` toml
project = "..."
queue = "..."
```

These map to the Slurm account/project and partition used by the
generated batch job.

### The executable exists on the login node but fails on the compute node

Ensure that:

-   `sbd_executable` uses an absolute path,
-   the filesystem is mounted on the compute nodes,
-   the executable has execute permission,
-   required dynamic libraries are available on the compute nodes.

### Prefect cannot find the solver Block

Check the configured Prefect profile/server:

``` bash
prefect profile ls
prefect config view
prefect block ls
```

Then recreate the Blocks if necessary:

``` bash
python algorithms/sbd/create_blocks.py \
  --config algorithms/sbd/sbd_blocks.toml
```

------------------------------------------------------------------------

## 5. Files introduced for Slurm support

The Slurm SBD integration uses:

``` text
algorithms/sbd/create_blocks.py
algorithms/sbd/native/build_sbd_slurm.sh
algorithms/sbd/sbd_blocks.slurm.example.toml
packages/qcsc-prefect-adapters/src/qcsc_prefect_adapters/slurm/templates/batch.slurm.j2
```

The local runtime file:

``` text
algorithms/sbd/sbd_blocks.toml
```

is intended for machine-specific configuration and should not be
committed with cluster-specific paths or settings.

------------------------------------------------------------------------

## 6. Recommended validation sequence

Before scaling the SQD calculation or enabling a real quantum device,
validate the Slurm integration in stages:

1.  Build `diag` successfully.
2.  Confirm the executable runs on the target Slurm compute environment.
3.  Create the Slurm Prefect Blocks.
4.  Run the workflow with a small SQD subspace and
    `quantum_source="random"`.
5.  Confirm Slurm submission, job completion, and SBD result parsing.
6.  Inspect the Prefect telemetry artifact.
7.  Enable `quantum_source="real-device"` after the classical Slurm path
    is working.
8.  Increase SQD/solver parameters only after the end-to-end workflow is
    stable.

This staged approach separates scheduler/build problems from
quantum-access problems and makes the integration easier to debug.

------------------------------------------------------------------------

END OF TUTORIAL
