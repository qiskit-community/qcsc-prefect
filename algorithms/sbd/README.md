# RIKEN SQD workflow as model case

This workflow iteratively updates LUCJ parameter with the differential evoluation optimization.
Each round of optimization runs a single configuration recovery loop per walker.
We use the SBD solver to compute carryover bitstrings in HPC space.

## Blocks

To run the workflow, you need to setup the following blocks.

### Qiskit Integrations

```bash
prefect block register -m prefect_qiskit
prefect block register -m prefect_qiskit.vendors
```

- `IBM Quantum Credentials` (block name: arbitrary)
- `Quantum Runtime` (block name: `sqd-runtime`)

When the `sqd-runtime` block doesn't exist, the bit sampling switchs to the random uniform sampling.

### SBD Integration

```bash
prefect block register -m prefect_sbd
```

- `SBD Solver Job` (block name: `davidson-solver`)

Example (`-np 384`):
```
adet comm size: 4
bdet comm size: 4
task comm size: 4
# node: 64
# mpiproc: 6
# omp threads: 18
# block: 10
# iteration: 10
```

### S3 Integration

We use builtin S3 integration.

- `MinIO Credentials` (block name: arbitrary)
- `S3 Bucket` (block name: `s3-sqd`)

On the MDX platform, we host a dedicated S3 service:
- Endpoint URL: `https://qii-kawasaki-miyabi-serv.cspp.cc.u-tokyo.ac.jp`
- MinIO Root User: `minioroot`
- MinIO Root Password: `*******` (ask Hitomi Takahashi)

## Options

We can optionally provide sampler execution options: `sqd_options`

Example:
```bash
prefect variable set sqd_options '{"params": {"shots": 500000, "options": {"dynamical_decoupling": {"enable": true, "sequence_type": "XY4", "skip_reset_qubits": true}}}}' --overwrite
```

(500k shots, DD enabled / XY4)
