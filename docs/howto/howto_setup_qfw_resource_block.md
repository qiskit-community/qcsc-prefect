# Create and Register a QFw Resource Block

The `QFwResource` Prefect block stores the configuration needed to access a quantum device through QFw.

Use this block when the SQD workflow is configured with:

```text
quantum_source = "qfw"
```

## 1. Prerequisites

Before creating the `QFwResource` block:

- The QFw-SLURM container environment must be running.
- QFw/QRMI and the target quantum device must be configured at `/etc/openqse/qfw/device/device-access.yaml`.
- A Prefect server must be running and the active Prefect profile must point to that server.
- The QFw device-access configuration and provider credentials must already be available.

For Prefect server setup in the QFw-SLURM environment, see:

[`howto_setup_prefect_server_slurm.md`](howto_setup_prefect_server_slurm.md)

Provider credentials should remain in the QFw credential configuration rather than being duplicated directly in the Prefect block.

## 2. Register the QFw Resource Block Type

The `QFwResource` block contains the information needed to locate and access a QFw device resource.

The block implementation is located at:

```text
/path/to/qcsc-prefect/algorithms/sbd/sbd/qfw_resource.py
```
Register the block type with Prefect:

```bash
prefect block register -f /path/to/qcsc-prefect/algorithms/sbd/sbd/qfw_resource.py
```

## 3. Create the QFw Resource Block Instance

Create and save the `qfw-runner` block instance using:

```bash
python /path/to/qcsc-prefect/algorithms/sbd/create_qfw_block.py
```

## 4. Verify the Block Instance

Check that the named block instance was created:

```bash
prefect block ls
```

The block should appear with the name:

```text
qfw-runner
```
