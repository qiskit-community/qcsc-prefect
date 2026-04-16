from __future__ import annotations

import argparse
import asyncio

from prefect import flow
from prefect.artifacts import create_table_artifact
from prefect.variables import Variable
from prefect_qiskit import QuantumRuntime
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

try:
    from get_counts_integration import BITLEN, BitCounter
    from options_resolver import resolve_sampler_options_and_work_dir
except ModuleNotFoundError:
    from examples.prefect_bitcount_demo.get_counts_integration import BITLEN, BitCounter
    from examples.prefect_bitcount_demo.options_resolver import resolve_sampler_options_and_work_dir

@flow(name="miyabi_tutorial")
async def main(
    runtime_block_name: str = "ibm-runner",
    bitcounter_block_name: str = "miyabi-tutorial",
    options_variable_name: str = "miyabi-tutorial",
    hpc_profile_block_override: str | None = None,
):
    runtime = await QuantumRuntime.load(runtime_block_name)
    counter = await BitCounter.load(bitcounter_block_name)
    if hpc_profile_block_override:
        counter = counter.model_copy(
            update={"hpc_profile_block_name": hpc_profile_block_override}
        )
    options_raw = await Variable.get(options_variable_name)
    sampler_options, _ = resolve_sampler_options_and_work_dir(options_raw, default_shots=100000)

    target = await runtime.get_target()
    qc_ghz = QuantumCircuit(BITLEN)
    qc_ghz.h(0)
    qc_ghz.cx(0, range(1, BITLEN))
    qc_ghz.measure_active()

    pm = generate_preset_pass_manager(
        optimization_level=3,
        target=target,
        seed_transpiler=123,
    )
    isa = pm.run(qc_ghz)
    pub_like = (isa,)

    results = await runtime.sampler([pub_like], options=sampler_options)
    bitstrings = results[0].data.meas.get_bitstrings()
    counts = await counter.get(bitstrings)

    await create_table_artifact(
        table=[list(counts.keys()), list(counts.values())],
        key="sampler-count-dict",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run tutorial-style BitCount flow with an optional HPC profile override."
    )
    parser.add_argument("--runtime-block", default="ibm-runner")
    parser.add_argument("--bitcounter-block", default="miyabi-tutorial")
    parser.add_argument("--options-variable", default="miyabi-tutorial")
    parser.add_argument("--hpc-profile-block-override", default=None)
    args = parser.parse_args()

    asyncio.run(
        main(
            runtime_block_name=args.runtime_block,
            bitcounter_block_name=args.bitcounter_block,
            options_variable_name=args.options_variable,
            hpc_profile_block_override=args.hpc_profile_block_override,
        )
    )
