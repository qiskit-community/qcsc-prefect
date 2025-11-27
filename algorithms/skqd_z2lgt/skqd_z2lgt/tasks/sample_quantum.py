"""Compose the Krylov circuits and run the quantum runtime sampler."""
import os
from collections.abc import Callable
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import h5py
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target, PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveIdentityEquivalent
from qiskit.primitives import BitArray, PrimitiveResult
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.circuits import make_step_circuits, compose_trotter_circuits


def get_trotter_circuits(
    parameters: Parameters,
    target: Target,
    logger: Optional[logging.Logger] = None
) -> tuple[list[int], list[QuantumCircuit], list[QuantumCircuit]]:
    """Compose full Trotter simulation circuits.

    We first generate single-step circuit elements for the given lattice and base two-qubit gate,
    compile them, then compose the resulting ISA circuits into multi-step Trotter simulation
    circuits. Both the forward-evolution (Krylov) circuits and forward-backward (reference) circuits
    are returned.

    Args:
        parameters: Workflow parameters.
        target: Backend target.

    Returns:
        Physical qubit layout and lists (length configuration.num_steps) of forward-evolution and
        forward-backward circuits.
    """
    logger = logger or logging.getLogger(__name__)

    lattice = TriangularZ2Lattice(parameters.lgt.lattice)
    layout = lattice.layout_heavy_hex(target=target, basis_2q=parameters.circuit.basis_2q)

    pm = generate_preset_pass_manager(
        optimization_level=parameters.circuit.optimization_level,
        target=target,
        initial_layout=layout,
    )
    pm.post_optimization = PassManager(
        [
            FoldRzzAngle(),
            Optimize1qGatesDecomposition(target=target),  # Cancel added local gates
            RemoveIdentityEquivalent(target=target),  # Remove GlobalPhaseGate
        ]
    )
    circuits = make_step_circuits(lattice, parameters.lgt.plaquette_energy,
                                  parameters.skqd.dt, parameters.circuit.basis_2q,
                                  parameters.lgt.charged_vertices)
    # Somehow the combination of multiprocessing pm.run + prefect task causes the former to hang
    logger.info('Transpiling circuits')
    init, full_step, fwd_step, bkd_step, measure = [pm.run(circuit) for circuit in circuits]

    id_step = fwd_step.compose(bkd_step)
    exp_circuits = compose_trotter_circuits(init, full_step, measure,
                                            parameters.skqd.n_trotter_steps)
    ref_circuits = compose_trotter_circuits(init, id_step, measure, parameters.skqd.n_trotter_steps)
    return layout, exp_circuits, ref_circuits


def save_raw(
    parameters: Parameters,
    pub_result: PrimitiveResult,
    logger: Optional[logging.Logger] = None
):
    logger = logger or logging.getLogger(__name__)
    logger.info('Saving raw link data')
    dirpath = Path(parameters.pkgpath) / 'data' / 'raw'
    try:
        os.makedirs(dirpath)
    except FileExistsError:
        pass

    ires = 0
    for etype in ['exp', 'ref']:
        for istep in range(parameters.skqd.n_trotter_steps):
            path = dirpath / f'{etype}_step{istep}.h5'
            with h5py.File(path, 'w', libver='latest') as out:
                bit_array = pub_result[ires].data.c
                dataset = out.create_dataset('link', data=bit_array.array)
                dataset.attrs['num_bits'] = bit_array.num_bits
                ires += 1


def load_raw(
    parameters: Parameters,
    etype: Optional[str] = None,
    istep: Optional[int] = None
) -> tuple[list[BitArray], list[BitArray]] | list[BitArray] | BitArray:
    def read_bit_array(dataset):
        return BitArray(dataset[()], int(dataset.attrs['num_bits']))
    
    if etype is None:
        etypes = ['exp', 'ref']
    else:
        etypes = [etype]
    if istep is None:
        isteps = list(range(parameters.skqd.n_trotter_steps))
    else:
        isteps = [istep]

    dirpath = Path(parameters.pkgpath) / 'data' / 'raw'
    data = []
    for etype in etypes:
        data.append([])
        for istep in isteps:
            with h5py.File(dirpath / f'{etype}_step{istep}.h5', 'r', libver='latest') as source:
                data[-1].append(read_bit_array(source['link']))

    if etype is None:
        return tuple(data)
    elif istep is None:
        return data[0]
    return data[0][0]


def sample_quantum_flow(
    parameters: Parameters,
    fetch_result_fn: Callable,
    get_target_fn: Callable,
    sample_fn: Callable,
    logger: Optional[logging.Logger] = None
):
    logger = logger or logging.getLogger(__name__)

    try:
        load_raw(parameters)
    except FileNotFoundError:
        pass
    else:
        logger.info('Raw bitstrings already saved to file')
        return

    if parameters.runtime.job_id:
        logger.info('Fetching result of workload %s', parameters.runtime.job_id)
        pub_result = fetch_result_fn()
        layout = None
    else:
        logger.info('Running a new experiment')
        # Transpile and compose the circuits
        target = get_target_fn()
        layout, exp_circuits, ref_circuits = get_trotter_circuits(parameters, target, logger)
        # Run primitive
        logger.info('Submitting a runtime job')
        pub_result, job_id = sample_fn(exp_circuits + ref_circuits)

        parameters.circuit.layout = layout
        parameters.runtime.job_id = job_id
        with open(Path(parameters.pkgpath) / 'parameters.json', 'w', encoding='utf-8') as out:
            out.write(parameters.model_dump_json())

    save_raw(parameters, pub_result, logger)


def sample_quantum(
    parameters: Parameters,
    logger: Optional[logging.Logger] = None
) -> tuple[list[BitArray], list[BitArray]]:
    """Run the circuits on a backend and return the sampler results.

    Args:
        parameters: Workflow parameters.


    Returns:
        Lists of BitArrays for forward-evolution and forward-backward circuits.
    """
    logger = logger or logging.getLogger(__name__)
    service = QiskitRuntimeService(instance=parameters.runtime.instance)

    def fetch_result_fn():
        return service.job(parameters.runtime.job_id).result()

    def get_target_fn():
        backend = service.backend(parameters.runtime.backend, use_fractional_gates=True)
        return backend.target

    def sample_fn(pubs):
        backend = service.backend(parameters.runtime.backend, use_fractional_gates=True)
        options = dict(parameters.runtime.options)
        options['default_shots'] = parameters.runtime.shots
        sampler = Sampler(backend, options=options)
        job = sampler.run(pubs)
        logger.info('Sampler job: %s', job.job_id())
        return job.result(), job.job_id()

    return sample_quantum_flow(parameters, fetch_result_fn, get_target_fn, sample_fn, logger)
