from sbd.qfw_resource import QFwResource

qfw = QFwResource(
    device_id="ibm_fez",
    device_access_config="/etc/openqse/qfw/device/device-access.yaml",
    interface="qrmi",
)

qfw.save(
    "qfw-runner",
    overwrite=True,
)

print("Created QFwResource block: qfw-runner")
