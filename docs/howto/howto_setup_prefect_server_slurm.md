# How to Run and Access a Prefect Server on a Slurm Login Node

This guide describes how to run a Prefect server on a Slurm login node and access the Prefect UI from your local computer using SSH port forwarding.

This setup is intended for small-scale development, testing and tutorial environments. For shared or production environments, use the Prefect deployment appropriate for your infrastructure.

## Prerequisites

Before starting, make sure that:

- You have access to a Slurm cluster.
- You can SSH to the Slurm login node.
- Prefect is installed in your Python environment on the login node.

## Step 1. Log in to the Slurm login node

From your local computer, connect to the Slurm login node:

```bash
ssh <username>@<login-node>
```

Activate the Python environment containing Prefect:

```bash
cd /path/to/qcsc-prefect
source .venv/bin/activate
```

## Step 2. Start the Prefect server

Start the Prefect server on the login node:

```bash
prefect server start --host 0.0.0.0 --background
```

The Prefect server listens on port `4200`.

## Step 3. Configure the Prefect client

Configure the Prefect client on the login node to connect to the local Prefect server:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

Verify that the server is accessible:

```bash
prefect server status
```

You can also verify the API directly:

```bash
curl http://127.0.0.1:4200/api/health
```

A successful response indicates that the Prefect server is running and accessible from the login node.

## Step 4. Access the Prefect UI from your local computer

The Prefect UI is running on the remote Slurm login node. To access it from your local computer, create an SSH tunnel that forwards port `4200` on your local computer to port `4200` on the login node.

Open a new terminal **on your local computer** and run:

```bash
ssh -L 4200:127.0.0.1:4200 <username>@<login-node>
```

Keep this SSH connection open while using the Prefect UI.

Then open the following address in a web browser on your local computer:

```text
http://127.0.0.1:4200
```

The Prefect UI running on the Slurm login node should now be accessible from your local browser.

## Step 5. Stop the Prefect server

When the server is no longer needed, stop it on the Slurm login node:

```bash
prefect server stop
```

You can start it again later with:

```bash
prefect server start --host 0.0.0.0 --background
```
