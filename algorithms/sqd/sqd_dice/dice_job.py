"""Dice solver integration for Prefect."""
import math
import struct
from pathlib import Path

import jinja2
import numpy as np
from prefect import get_run_logger
from prefect_miyabi import MiyabiJobBlock
from pydantic import Field
from pyscf.tools import fcidump
from qiskit_addon_sqd.fermion import SCIResult, SCIState


class DiceSHCISolverJob(MiyabiJobBlock):
    """Prefect integration of Dice solver for SHCI input and output."""
    
    _block_type_name = "Dice SHCI Solver Job"
    _block_type_slug = "dice_shci_solver_job"
    
    select_cutoff: float = Field(
        default=5e-4,
        description="Cutoff threshold for retaining state vector coefficients.",
        title="Select Cutoff",
    )
    
    davidson_tol: float = Field(
        default=1e-5,
        description="Floating point tolerance for Davidson solver.",
        title="Davidson Tolerance",
    )
    
    energy_tol: float = Field(
        default=1e-10,
        description="Floating point tolerance for SCI energy.",
        title="Energy Tolerance",
    )
    
    max_iter: int = Field(
        default=10,
        description="The maximum number of HCI iterations to perform.",
        title="Maximum Iteration",
    )
    
    async def run(
        self,
        ci_strings: tuple[np.ndarray, np.ndarray],
        one_body_tensor: np.ndarray,
        two_body_tensor: np.ndarray,
        norb: int,
        nelec: tuple[int, int],
        spin_sq: float | None = None,
    ) -> SCIResult:
        """Run SHCI solver.
        
        .. node::
            File I/O code is ported from https://github.com/Qiskit/qiskit-addon-dice-solver.
        
        Args:
            ci_strings: A pair of spin-alpha CI strings and spin-beta CI strings 
                whose Cartesian product give the basis of the subspace 
                in which to perform a diagonalization.
            one_body_tensor: The one-body tensor of the Hamiltonian.
            two_body_tensor: The two-body tensor of the Hamiltonian.
            norb: The number of spatial orbitals.
            nelec: The numbers of alpha and beta electrons.
            spin_sq: Target value for the total spin squared for the ground state. If ``None``, no spin will be imposed.
        
        Returns:
            SCIResult object constructed from DICE solver outputs.
        """
        logger = get_run_logger()
        
        with self.get_executor() as executor:
            self._prepare_input_files(
                work_dir=executor.work_dir,
                ci_strings=ci_strings,
                one_body_tensor=one_body_tensor,
                two_body_tensor=two_body_tensor,
                norb=norb,
                nelec=nelec,
                spin_sq=spin_sq,
            )
            
            exit_code = await executor.execute_job(**self.get_job_variables())

            if exit_code != 0:
                logger.warning(
                    f"Dice solver returned nonzero exit code {exit_code}. "
                    "Dice may return nonzero codes even on successful executions. "
                    "Investigating output files..."
                )
            sci_result = self._read_output_files(
                work_dir=executor.work_dir,
                norb=norb,
                nelec=nelec,
            )

        return sci_result
    
    def _prepare_input_files(
        self, 
        work_dir: Path, 
        ci_strings: tuple[np.ndarray, np.ndarray],
        one_body_tensor: np.ndarray,
        two_body_tensor: np.ndarray,
        norb: int,
        nelec: tuple[int, int],
        spin_sq: float | None = None,
     ):
        # Write PySCF FCI dump file.
        fcidump.from_integrals(
            str(work_dir / "fcidump.txt"),
            one_body_tensor,
            two_body_tensor,
            norb,
            nelec,
        )
        # Write input file.
        template_loader = jinja2.FileSystemLoader(searchpath=Path(__file__).resolve().parent)
        template_env = jinja2.Environment(loader=template_loader)
        input_dat = template_env.get_template("input.dat.j2").render(
            spin_sq=spin_sq,
            select_cutoff=self.select_cutoff,
            davidson_tol=self.davidson_tol,
            energy_tol=self.energy_tol,
            max_iter=self.max_iter,
            dim=min(
                2147483647,
                math.comb(norb, nelec[0]) * math.comb(norb, nelec[1]),
            ),
            num_elec=nelec[0] + nelec[1],            
        )
        with open(work_dir / "input.dat", "w") as fp:
            fp.write(input_dat)
        # Write the determinants.
        # The 16 is hard-coded because that is what the modified Dice branch expects currently.
        with open(work_dir / "AlphaDets.bin", "wb") as fp:
            for ci in ci_strings[0]:
                fp.write(ci.item().to_bytes(16, byteorder="big"))
        with open(work_dir / "BetaDets.bin", "wb") as fp:
            for ci in ci_strings[1]:
                fp.write(ci.item().to_bytes(16, byteorder="big"))
        
    def _read_output_files(
        self, 
        work_dir: Path,
        norb: int,
        nelec: tuple[int, int],
    ):
        # Read in the avg orbital occupancies
        spin1_rdm_dice = np.loadtxt(work_dir / "spin1RDM.0.0.txt", skiprows=1)
        avg_occupancies = np.zeros(2 * norb)
        for i in range(spin1_rdm_dice.shape[0]):
            if spin1_rdm_dice[i, 0] == spin1_rdm_dice[i, 1]:
                orbital_id = spin1_rdm_dice[i, 0]
                parity = orbital_id % 2
                avg_occupancies[int(orbital_id // 2 + parity * norb)] = spin1_rdm_dice[i, 2]
        # Read in the estimated ground state energy
        with open(work_dir / "shci.e", "rb") as fp:
            bytestring_energy = fp.read(8)
            energy = struct.unpack("d", bytestring_energy)[0]
        # Read the wavefunction magnitudes
        occupancy_strs = []
        amplitudes = []
        with open(work_dir / "dets.bin", "rb") as fp:
            num_dets = struct.unpack("i", fp.read(4))[0]
            num_orb = struct.unpack("i", fp.read(4))[0]
            for i in range(2 * num_dets):
                # Read the wave function amplitude
                if i % 2 == 0:
                    # Read the double-precision float describing the amplitude
                    wf_amplitude = struct.unpack("d", fp.read(8))[0]
                    amplitudes.append(wf_amplitude)
                else:
                    occupancy_strs.append(str(fp.read(num_orb))[2:-1])
        # Convert occupancies to CI strings.
        ci_strs = []
        for occ_str in occupancy_strs:
            bitstring = np.zeros(2 * norb, dtype=bool)
            for i, bit in enumerate(occ_str):
                match bit:
                    case "2":
                        bitstring[i] = 1
                        bitstring[i + norb] = 1
                    case "a":
                        bitstring[i] = 1
                    case "b":
                        bitstring[i + norb] = 1
            ci_str_a = sum(b << i for i, b in enumerate(bitstring[:norb]))
            ci_str_b = sum(b << i for i, b in enumerate(bitstring[norb:]))
            ci_strs.append((ci_str_a, ci_str_b))
        # Construct wavefunction amplitudes from CI strings and their associated amplitudes.
        strs_a, strs_b = zip(*ci_strs)
        uniques_a = np.unique(strs_a)
        uniques_b = np.unique(strs_b)
        num_dets_a = len(uniques_a)
        num_dets_b = len(uniques_b)
        sci_coefficients = np.zeros((num_dets_a, num_dets_b))
        ci_strs_a = np.zeros(num_dets_a, dtype=np.int64)
        ci_strs_b = np.zeros(num_dets_b, dtype=np.int64)
        ci_str_map_a = {uni_str: i for i, uni_str in enumerate(uniques_a)}
        ci_str_map_b = {uni_str: i for i, uni_str in enumerate(uniques_b)}
        for amp, ci_str in zip(amplitudes, ci_strs):
            ci_str_a, ci_str_b = ci_str
            i = ci_str_map_a[ci_str_a]
            j = ci_str_map_b[ci_str_b]
            sci_coefficients[i, j] = amp
            ci_strs_a[i] = uniques_a[i]
            ci_strs_b[j] = uniques_b[j]
        sci_state = SCIState(
            amplitudes=sci_coefficients,
            ci_strs_a=ci_strs_a,
            ci_strs_b=ci_strs_b,
            norb=norb,
            nelec=nelec,
        )
        # Construct SCIResult to return.
        # Skip computation of reduced dencity matrix since they are not used in SQD.
        return SCIResult(
            energy=energy,
            sci_state=sci_state,
            orbital_occupancies=(avg_occupancies[:norb], avg_occupancies[norb:]),
        )
