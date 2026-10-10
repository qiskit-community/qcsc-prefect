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
