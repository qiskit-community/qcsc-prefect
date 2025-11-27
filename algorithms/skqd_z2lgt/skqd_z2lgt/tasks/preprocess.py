"""Preprocess raw data (link states with errors) and convert them to vertex and plaquette data."""
import os
from collections.abc import Callable
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import h5py
from qiskit.primitives import BitArray
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.mwpm import convert_link_to_plaq
from skqd_z2lgt.utils import read_bits, save_bits
from skqd_z2lgt.tasks.sample_quantum import load_raw
from skqd_z2lgt.tasks.common import make_dual_lattice

RecoData = list[tuple[np.ndarray, np.ndarray]]  # [(vertex data, plaquette data)] * steps


def save_reco(
    parameters: Parameters,
    reco_data: tuple[RecoData, RecoData],
    logger: Optional[logging.Logger] = None
):
    logger = logger or logging.getLogger(__name__)
    logger.info('Saving vertex and plaquette data')
    dirpath = Path(parameters.pkgpath) / 'data' / 'reco'
    try:
        os.makedirs(dirpath)
    except FileExistsError:
        pass

    for etype, arrays in zip(['exp', 'ref'], reco_data):
        for istep in range(parameters.skqd.n_trotter_steps):
            path = dirpath / f'{etype}_step{istep}.h5'
            with h5py.File(path, 'w', libver='latest') as out:
                for name, array in zip(['vtx', 'plaq'], arrays[istep]):
                    save_bits(out, name, array)


def load_reco(
    parameters: Parameters,
    etype: Optional[str] = None,
    istep: Optional[int] = None
) -> tuple[RecoData, RecoData] | RecoData | tuple[np.ndarray, np.ndarray]:
    if etype is None:
        etypes = ['exp', 'ref']
    else:
        etypes = [etype]
    if istep is None:
        isteps = list(range(parameters.skqd.n_trotter_steps))
    else:
        isteps = [istep]

    dirpath = Path(parameters.pkgpath) / 'data' / 'reco'
    data = []
    for etype in etypes:
        data.append([])
        for istep in isteps:
            with h5py.File(dirpath / f'{etype}_step{istep}.h5', 'r', libver='latest') as source:
                data[-1].append((read_bits(source['vtx']), read_bits(source['plaq'])))

    if etype is None:
        return tuple(data)
    elif istep is None:
        return data[0]
    return data[0][0]


def preprocess_flow(
    parameters: Parameters,
    raw_data: tuple[list[BitArray], list[BitArray]],
    convert_fn: Callable,
    logger: Optional[logging.Logger] = None
) -> tuple[RecoData, RecoData]:
    """Correct the link-state bitstrings with MWPM and convert to plaquette-state bitstrings.

    Args:
        parameters: Configuration parameters.
        cpu_pyfuncjob_name: Name of the PyFunctionJob block that runs a python function in an
            interpreter in the current environment.
        bit_arrays: Lists of BitArrays returned by sample_krylov_bitstrings.
    """
    logger = logger or logging.getLogger(__name__)

    try:
        reco_data = load_reco(parameters)
    except FileNotFoundError:
        pass
    else:
        logger.info('Loading existing reco data from output file')
        return reco_data

    logger.info('Correcting the charge sector of link-state bitstrings and converting them to '
                'vertex and plaquette data')

    dual_lattice = make_dual_lattice(parameters)
    reco_data = convert_fn(raw_data, dual_lattice)
    save_reco(parameters, reco_data)

    return reco_data


def preprocess(
    parameters: Parameters,
    raw_data: tuple[RecoData, RecoData],
    logger: Optional[logging.Logger] = None
) -> tuple[RecoData, RecoData]:
    def convert_fn(bit_arrays, dual_lattice):
        batch_size = parameters.runtime.shots // 20
        reco_data = []
        for arrays in bit_arrays:
            reco_data.append([])
            for array in arrays:
                reco_data[-1].append(convert_link_to_plaq(array, dual_lattice,
                                                          batch_size=batch_size))
        return tuple(reco_data)

    return preprocess_flow(parameters, raw_data, convert_fn, logger)


def preprocess_single_array(
    parameters: Parameters,
    etype: str,
    istep: int,
    logger: Optional[logging.Logger] = None
):
    """Preprocess one bit array parallelized over all cores and save the result."""
    logger = logger or logging.getLogger(__name__)
    logger.info('Preprocessing %s step %d', etype, istep)
    
    bit_array = load_raw(parameters, etype, istep)
    dual_lattice = make_dual_lattice(parameters)
    batch_size = bit_array.num_shots // (os.cpu_count() - 1)
    arrays = convert_link_to_plaq(bit_array, dual_lattice, batch_size)

    path = Path(parameters.pkgpath) / 'data' / 'reco' / f'{etype}_step{istep}.h5'
    try:
        os.makedirs(path.parent)
    except FileExistsError:
        pass

    with h5py.File(path, 'w', libver='latest') as out:
        for name, array in zip(['vtx', 'plaq'], arrays):
            save_bits(out, name, array)


if __name__ == '__main__':
    import argparse
    from mpi4py import MPI

    parser = argparse.ArgumentParser()
    parser.add_argument('pkgpath')
    parser.add_argument('--etype')
    parser.add_argument('--istep', type=int)
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()))

    if options.mpi:
        comm = MPI.COMM_WORLD
        mpi_rank = comm.Get_rank()
        nsteps = params.skqd.n_trotter_steps
        if mpi_rank < nsteps:
            et = 'exp'
        else:
            et = 'ref'
        ist = mpi_rank % nsteps
    else:
        et = options.etype
        ist = options.istep

    with open(Path(options.pkgpath) / 'parameters.json', 'r', encoding='utf-8') as src:
        params = Parameters.model_validate_json(src.read())
    preprocess_single_array(params, et, ist)
