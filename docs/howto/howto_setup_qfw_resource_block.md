# Create and Register a QFw Resource Block

The `QFwResource` Prefect block stores the configuration needed to access a quantum device through QFw.

Use this block when the SQD workflow is configured with:

```text
quantum_source = "qfw"
```

## 1. Prerequisites

The QFw-SLURM container environment should already be configured with:

- QFw installed and activated
- A valid `device-access.yaml`
- QRMI support for the target device
- Provider credentials available to QFw

For example:

```text
/etc/openqse/qfw/device/device-access.yaml
```

Provider credentials should remain in the QFw credential configuration rather than being duplicated directly in the Prefect block.

## 2. Register QFw Resource Block

The block contains the information needed to locate and access the QFw device resource.

Example:

```python
from prefect.blocks.core import Block
from pydantic import Field


class QFwResource(Block):
    device_id: str = Field(
        description="QFw device identifier, for example ibm_fez."
    )

    device_access_config: str = Field(
        default="/etc/openqse/qfw/device/device-access.yaml",
        description="Path to the QFw device-access configuration.",
    )

    interface: str = Field(
        default="qrmi",
        description="Preferred QFw quantum interface.",
    )
```

The python file is located at /path/to/qcsc-prefect/algorithms/sbd/sbd/qfw_resource.py

Register the block type with Prefect:

```bash
prefect block register -f /path/to/qcsc-prefect/algorithms/sbd/sbd/qfw_resource.py
```

## 3. Create QFw Resource Block

Create a small registration script, for example:

```python
python /path/to/qcsc-prefect/algorithms/sbd/create_qfw_block.py
```

## 4. Verify the Block

Check that the block was registered:

```bash
prefect block ls
```

The block should appear with the name:

```text
qfw-runner
```
