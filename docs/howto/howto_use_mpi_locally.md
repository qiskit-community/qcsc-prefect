# How to Run MPI Programs Locally on a Linux VM

This guide walks you through building and running MPI programs written in C or C++ on a Linux virtual machine. It is intended for testing before deploying to the Miyabi supercomputer environment.

## Prerequisites

- This guide assumes a Red Hat-based Linux distribution (e.g., RHEL, CentOS, Fedora).
- You need administrative privileges to install packages.

## Instructions

### Step 1. Install Required Packages

Install the necessary compilers and MPI libraries:

<img src="../images/icon-pc.png" alt="pc" width="50"/><br>
```bash
dnf install gcc
dnf install gcc-c++
dnf install openmpi openmpi-devel 
```

Initialize the environment module system:

<img src="../images/icon-pc.png" alt="pc" width="50"/><br>
```bash
source /etc/profile.d/modules.sh 
```

Load the OpenMPI module:

<img src="../images/icon-pc.png" alt="pc" width="50"/><br>
```bash
module load mpi/openmpi-x86_64 
```

### Step 2. Compile Your MPI Program

Use the OpenMPI compiler wrapper to compile your source code:

<img src="../images/icon-pc.png" alt="pc" width="50"/><br>
```bash
mpicxx my-program.cpp -o my-program
```

If you're using a third-party library, it may provide its own build script. 
Refer to its documentation for details.

### Step 3. Run the Program

Launch your program using `mpirun` with the desired number of processes:

<img src="../images/icon-pc.png" alt="pc" width="50"/><br>
```bash
mpirun -np 4 ./my-program
```

## 🛠️ Troubleshooting

### Running Multiple MPI Programs Simultaneously

If you run multiple MPI programs at the same time, you may encounter errors like:

```text
terminate called after throwing an instance of 'std::out_of_range'
  what():  vector::_M_range_check: __n (which is 22) >= this->size() (which is 22)
```

This is likely caused by a race condition. If you're using the `prefect-miyabi` integration, limit concurrency to avoid conflicts:

<img src="../images/icon-pc.png" alt="pc" width="50"/><br>
```bash
prefect concurrency-limit create "res: local" 1
```

### UCX-Related Segmentation Faults

In shared-memory environments, UCX (Unified Communication X) may cause segmentation faults. 
To avoid this, switch to TCP-based communication.
Export the following environment variables to the shell:

<img src="../images/icon-pc.png" alt="pc" width="50"/><br>
```bash
OMPI_MCA_pml="ob1"
OMPI_MCA_btl="tcp,self"
OMPI_MCA_btl_tcp_if_include="lo"
```

---
*END OF GUIDE*
