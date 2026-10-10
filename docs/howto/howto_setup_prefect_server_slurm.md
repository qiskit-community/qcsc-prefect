# How to Run and Access a Prefect Server on a Slurm Login Node

This guide describes how to run a Prefect server on a Slurm login node and access the Prefect UI from your local computer using SSH port forwarding.

It also includes notes for running Prefect inside the `QFw-SLURM` container environment.

This setup is intended for small-scale development, testing and tutorial environments. For shared or production environments, use the Prefect deployment appropriate for your infrastructure.

## Prerequisites

Before starting, make sure that:

- You have access to a Slurm cluster.
- You can SSH to the Slurm login node.
- Prefect is installed in the Python environment used for the workflow.

For the QFw-SLURM environment, also make sure that:

- The QFw-SLURM container environment is running.
- QFw is installed and configured.
- The target quantum device is configured in:

```text
/etc/openqse/qfw/device/device-access.yaml
```

## Step 1. Log in to the Slurm login node

From your local computer, connect to the Slurm login node:

```bash
ssh <username>@<login-node>
```

### Generic Slurm environment

Activate the Python environment containing Prefect:

```bash
cd /path/to/qcsc-prefect
source .venv/bin/activate
```

### QFw-SLURM environment

In the QFw-SLURM setup, run the Prefect server on the `slurmctld` container.

Enter the container using the mechanism appropriate for your QFw-SLURM deployment, then activate the QFw environment:

```bash
source /opt/openqse/qfw/bin/qfw-activate --venv /opt/openqse/qfw-venv
```

The QFw environment contains the QFw and QRMI libraries needed by the quantum execution path.

## Step 2. Start the Prefect server

Start the Prefect server:

```bash
prefect server start --host 0.0.0.0 --background
```

The Prefect server listens on port `4200`.

Using `--host 0.0.0.0` allows the server to be reached through port forwarding from outside the login node or container.

## Step 3. Configure the Prefect client

Configure the Prefect client in the same environment to connect to the local Prefect server:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

Verify the active Prefect configuration:

```bash
prefect profile inspect
```

The active profile should show:

```text
PREFECT_API_URL='http://127.0.0.1:4200/api'
```

Verify that the server is accessible:

```bash
prefect server status
```

You can also verify the API directly:

```bash
curl http://127.0.0.1:4200/api/health
```

A successful response indicates that the Prefect server is running and accessible from the login node or container.

## Step 4. Access the Prefect UI from your local computer

The Prefect UI is served on port `4200`.

### Generic Slurm environment

If the Prefect server is running directly on the Slurm login node, open a new terminal on your local computer and run:

```bash
ssh -L 4200:127.0.0.1:4200 <username>@<login-node>
```

Keep this SSH connection open while using the Prefect UI.

Then open:

```text
http://127.0.0.1:4200
```

in your local web browser.

### QFw-SLURM environment

If the Prefect server is running inside the `slurmctld` container, forward the local port to the container IP through the host running the QFw-SLURM environment.

For example:

```bash
ssh -L 4200:<slurmctld-container-ip>:4200 <username>@<remote-host>
```

Then open:

```text
http://127.0.0.1:4200
```

in your local browser.

The exact command depends on how the QFw-SLURM containers are deployed and accessed.

If needed, the container IP can be obtained from the container runtime on the remote host.

## Step 5. Verify Prefect from the workflow environment

Before creating Prefect blocks or running the SQD workflow, verify that the active environment is connected to the expected Prefect server:

```bash
prefect profile inspect
```

and:

```bash
curl http://127.0.0.1:4200/api/health
```

For QFw-SLURM, these commands should be run inside the `slurmctld` container after activating the QFw environment.


## Step 6. Stop the Prefect server

When the server is no longer needed, stop it in the environment where it was started:

```bash
prefect server stop
```

You can start it again later with:

```bash
prefect server start --host 0.0.0.0 --background
```
