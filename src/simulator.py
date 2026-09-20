"""
src/simulator.py — SolGelSimulator class for Reactive MC (v5.2.1)
==================================================================
[FIX v5.2.1]: Corrected 'self.coords' to 'self.system.coords' in save_structure().
"""

import numpy as np
import json
import logging
from pathlib import Path
from math import exp, sqrt, pi

from .constants import (
    KB_EV, NA_AVOGADRO, KE_COULOMB, ATM_TO_EV_A3,
    TYPE_SI, TYPE_P, TYPE_O, MAX_COORD, BOND_FORM_CUTOFF,
    SKIN_DEFAULT, CUTOFF_DEFAULT, WOLF_ALPHA_DEFAULT, TYPE_TO_ELEM
)
from .numba_kernels import (
    local_energy_reactive, total_energy_reactive, compute_nc_numba,
    find_o_candidates_for_nf_fast, find_nf_candidates_for_o_fast,
    find_bonded_nfs_for_o_fast, energy_decomposition
)
from .neighbor_list import NeighborList
from .bond_network import BondNetwork

logger = logging.getLogger(__name__)


class SolGelSimulator:
    def __init__(self, system, seed=42, cutoff=CUTOFF_DEFAULT,
                 wolf_alpha=WOLF_ALPHA_DEFAULT, npt_enabled=False,
                 pressure_atm=1.0, disp_volume=0.02, debug_energy=False):
        self.system = system
        self.rng = np.random.default_rng(seed)
        self.cutoff = cutoff
        self.wolf_alpha = wolf_alpha
        self.seed = seed
        self.debug_energy = debug_energy
        
        self.npt_enabled = npt_enabled
        self.P_ext = pressure_atm * ATM_TO_EV_A3
        self.disp_volume = disp_volume

        self.bond_network = BondNetwork(system.N_ATOMS)
        self.bond_network.set_types(system.type_indices)

        success_count, fail_count = 0, 0
        for i, j in system.initial_bonds:
            ti = int(system.type_indices[i])
            tj = int(system.type_indices[j])
            if (ti in (TYPE_SI, TYPE_P) and tj == TYPE_O) or \
               (tj in (TYPE_SI, TYPE_P) and ti == TYPE_O):
                bond_type = 0 if TYPE_SI in (ti, tj) else 1
            else:
                fail_count += 1
                continue
            if self.bond_network.add_bond(i, j, bond_type) >= 0:
                success_count += 1
            else:
                fail_count += 1
                
        logger.info(f"Initialized {self.bond_network.n_active_bonds} bonds "
                    f"(success={success_count}, fail={fail_count})")

        self.nl = NeighborList(
            system.coords, system.box, cutoff, SKIN_DEFAULT, self.bond_network)
        
        self.A_mat = system.A_mat
        self.F_mat = system.F_mat
        self.C_mat = system.C_mat
        self.R_HARD_MAT = system.R_HARD_MAT
        self.type_Z = system.type_Z
        self.wolf_self = (-KE_COULOMB * (self.wolf_alpha / sqrt(pi)) *
                          np.sum(self.system.charges ** 2))

        self.step = 0
        self.current_energy = 0.0
        self.accepted = self.attempts = 0
        self.vol_accepted = self.vol_attempts = 0
        self.bond_form_accepted = self.bond_form_attempts = 0
        self.bond_break_accepted = self.bond_break_attempts = 0
        self.bond_switch_accepted = self.bond_switch_attempts = 0

        self.max_disp = {
            'Si': 0.04, 'Ca': 0.08, 'Na': 0.08,
            'P': 0.05, 'O': 0.08, 'Sr': 0.08
        }
        
        self.energy_log, self.nc_log, self.drift_log = [], [], []
        self.log_freq = max(100, system.N_ATOMS // 100)
        
        if self.debug_energy:
            logger.info("[DEBUG] Energy drift monitoring ENABLED")
        self._init_energy()

    def _init_energy(self):
        n, s, bf = self.nl.neighbors, self.nl.starts, self.nl.bond_flags
        U = total_energy_reactive(
            self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            n, s, bf, self.system.box, self.cutoff, self.wolf_alpha,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d)
        self.current_energy = U + self.wolf_self
        logger.info(f"Initial energy: "
                    f"{self.current_energy / self.system.N_ATOMS:.6f} eV/atom")

    def recalculate_energy(self):
        """Recalculate total energy from scratch (prevents drift)."""
        self.nl.coords, self.nl.box = self.system.coords, self.system.box
        self.nl.update(force=True)
        n, s, bf = self.nl.neighbors, self.nl.starts, self.nl.bond_flags
        U = total_energy_reactive(
            self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            n, s, bf, self.system.box, self.cutoff, self.wolf_alpha,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d)
        self.current_energy = U + self.wolf_self
        return self.current_energy

    def check_energy_drift(self):
        """Compare current_energy with full recalculation."""
        old_energy = self.current_energy
        self.nl.coords, self.nl.box = self.system.coords, self.system.box
        self.nl.update(force=True)
        n, s, bf = self.nl.neighbors, self.nl.starts, self.nl.bond_flags
        U = total_energy_reactive(
            self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            n, s, bf, self.system.box, self.cutoff, self.wolf_alpha,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d)
        new_energy = U + self.wolf_self
        drift = (new_energy - old_energy) / self.system.N_ATOMS
        
        if abs(drift) > 0.01:
            logger.warning(f"[DRIFT] Energy drift detected: "
                           f"{drift:+.4f} eV/atom")
        self.current_energy = new_energy
        self.drift_log.append(drift)
        return drift

    def decompose_energy(self):
        """
        Decompose total energy into components for debugging.
        Returns dict with 'bonded', 'buckingham', 'coulomb', 'zbl', 'total'.
        """
        n, s, bf = self.nl.neighbors, self.nl.starts, self.nl.bond_flags
        e_bonded, e_buck, e_coul, e_zbl = energy_decomposition(
            self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            n, s, bf, self.system.box, self.cutoff, self.wolf_alpha,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d)
        
        return {
            'bonded': e_bonded / self.system.N_ATOMS,
            'buckingham': e_buck / self.system.N_ATOMS,
            'coulomb': (e_coul + self.wolf_self) / self.system.N_ATOMS,
            'zbl': e_zbl / self.system.N_ATOMS,
            'total': (e_bonded + e_buck + e_coul + e_zbl + self.wolf_self) / self.system.N_ATOMS,
        }

    def compute_nc(self) -> float:
        """Compute Network Connectivity."""
        bond_i, bond_j, _ = self.bond_network.get_bond_arrays()
        return compute_nc_numba(
            self.system.type_indices, bond_i, bond_j, self.system.N_ATOMS)

    def mc_move(self, T: float) -> bool:
        """Standard MC displacement move."""
        i = int(self.rng.integers(0, self.system.N_ATOMS))
        old_pos = self.system.coords[i].copy()
        maxd = self.max_disp.get('O', 0.06)  # Default fallback
        # Map type index to element for max_disp lookup
        elem = TYPE_TO_ELEM.get(self.system.type_indices[i], 'O')
        maxd = self.max_disp.get(elem, 0.06)
        
        n_arr, s_arr, bf = (self.nl.neighbors, self.nl.starts,
                            self.nl.bond_flags)
        
        old_e = local_energy_reactive(
            i, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.cutoff, self.wolf_alpha,
            n_arr, s_arr, bf,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d)
        
        delta = self.rng.uniform(-maxd, maxd, size=3)
        self.system.coords[i] = (old_pos + delta) % self.system.box
        
        new_e = local_energy_reactive(
            i, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.cutoff, self.wolf_alpha,
            n_arr, s_arr, bf,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d)
        
        de = new_e - old_e
        if de <= 0.0 or self.rng.random() < exp(-de / (KB_EV * T)):
            self.current_energy += de
            self.accepted += 1
            self.attempts += 1
            dr = self.system.coords[i] - self.nl.ref_coords[i]
            dr -= self.system.box * np.round(dr / self.system.box)
            if np.sqrt(np.sum(dr ** 2)) > SKIN_DEFAULT / 3.0:
                self.nl.update(force=True)
            return True
        else:
            self.system.coords[i] = old_pos
            self.attempts += 1
            return False

    def bond_formation_move(self, T: float) -> bool:
        """Attempt to form a new Si-O or P-O bond."""
        self.bond_form_attempts += 1
        
        nf_candidates = [
            i for i in range(self.system.N_ATOMS)
            if self.system.type_indices[i] in (TYPE_SI, TYPE_P)
            and self.bond_network.get_coordination(i) < MAX_COORD[self.system.type_indices[i]]
        ]
        if not nf_candidates:
            return False
        nf = nf_candidates[self.rng.integers(0, len(nf_candidates))]
        
        bond_starts, bond_partners, _, _ = self.bond_network.get_bond_csr()
        cand_dist, cand_idx = find_o_candidates_for_nf_fast(
            nf, self.system.coords, self.system.type_indices,
            self.system.box, self.nl.neighbors, self.nl.starts,
            bond_starts, bond_partners, BOND_FORM_CUTOFF)
        
        if len(cand_idx) == 0:
            return False
        
        best = np.argmin(cand_dist)
        o_target = int(cand_idx[best])
        
        n_arr, s_arr, bf = (self.nl.neighbors, self.nl.starts,
                            self.nl.bond_flags)
        old_e_nf = local_energy_reactive(
            nf, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.cutoff, self.wolf_alpha,
            n_arr, s_arr, bf,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d)
        old_e_o = local_energy_reactive(
            o_target, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.cutoff, self.wolf_alpha,
            n_arr, s_arr, bf,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d)
        
        ti = int(self.system.type_indices[nf])
        bond_type = 0 if ti == TYPE_SI else 1
        bond_id = self.bond_network.add_bond(nf, o_target, bond_type)
        if bond_id < 0:
            return False
        
        self.nl.update_bond_flags_local(nf, o_target, bond_type)
        
        new_e_nf = local_energy_reactive(
            nf, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.cutoff, self.wolf_alpha,
            n_arr, s_arr, self.nl.bond_flags,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d)
        new_e_o = local_energy_reactive(
            o_target, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.cutoff, self.wolf_alpha,
            n_arr, s_arr, self.nl.bond_flags,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d)
        
        de = ((new_e_nf + new_e_o) - (old_e_nf + old_e_o)) / 2.0
        
        if de <= 0.0 or self.rng.random() < exp(-de / (KB_EV * T)):
            self.current_energy += de
            self.bond_form_accepted += 1
            return True
        else:
            self.bond_network.remove_bond(bond_id)
            self.nl.update_bond_flags_local(nf, o_target, -1)
            return False

    def bond_breaking_move(self, T: float) -> bool:
        """Attempt to break an existing bond."""
        self.bond_break_attempts += 1
        all_bonds = self.bond_network.get_all_bonds()
        if not all_bonds:
            return False
        
        idx = self.rng.integers(0, len(all_bonds))
        bond_id, i, j, btype = all_bonds[idx]
        
        n_arr, s_arr, bf = (self.nl.neighbors, self.nl.starts,
                            self.nl.bond_flags)
        old_e_i = local_energy_reactive(
            i, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.cutoff, self.wolf_alpha,
            n_arr, s_arr, bf,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d)
        old_e_j = local_energy_reactive(
            j, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.cutoff, self.wolf_alpha,
            n_arr, s_arr, bf,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d)
        
        self.bond_network.remove_bond(bond_id)
        self.nl.update_bond_flags_local(i, j, -1)
        
        new_e_i = local_energy_reactive(
            i, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.cutoff, self.wolf_alpha,
            n_arr, s_arr, self.nl.bond_flags,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d)
        new_e_j = local_energy_reactive(
            j, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.cutoff, self.wolf_alpha,
            n_arr, s_arr, self.nl.bond_flags,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d)
        
        de = ((new_e_i + new_e_j) - (old_e_i + old_e_j)) / 2.0
        
        if de <= 0.0 or self.rng.random() < exp(-de / (KB_EV * T)):
            self.current_energy += de
            self.bond_break_accepted += 1
            return True
        else:
            self.bond_network.add_bond(i, j, btype)
            self.nl.update_bond_flags_local(i, j, btype)
            return False

    def bond_switch_move(self, T: float) -> bool:
        """
        Attempt to switch a bond from one NF to another.
        Uses recalculate_energy() for accurate delta-E computation.
        """
        self.bond_switch_attempts += 1
        bond_starts, bond_partners, _, _ = self.bond_network.get_bond_csr()
        
        bo_candidates = []
        for i in range(self.system.N_ATOMS):
            if self.system.type_indices[i] != TYPE_O:
                continue
            nf1, nf2 = find_bonded_nfs_for_o_fast(
                i, self.system.type_indices, bond_starts, bond_partners)
            if nf1 >= 0 and nf2 >= 0:
                bid1 = self.bond_network.get_bond_id(i, nf1)
                bid2 = self.bond_network.get_bond_id(i, nf2)
                bo_candidates.append((i, nf1, nf2, bid1, bid2))
        
        if not bo_candidates:
            return False
        
        idx = self.rng.integers(0, len(bo_candidates))
        o_atom, nf1, nf2, bid1, bid2 = bo_candidates[idx]
        
        if self.rng.integers(0, 2) == 0:
            old_nf, old_bid, keep_nf = nf1, bid1, nf2
        else:
            old_nf, old_bid, keep_nf = nf2, bid2, nf1
        
        old_bond_entry = self.bond_network.bonds.get(old_bid)
        if old_bond_entry is None:
            return False
        old_btype = old_bond_entry[2]
        
        new_nf_candidates = find_nf_candidates_for_o_fast(
            o_atom, self.system.coords, self.system.type_indices,
            self.system.box, self.nl.neighbors, self.nl.starts,
            bond_starts, bond_partners, BOND_FORM_CUTOFF, old_nf, keep_nf)
        
        if len(new_nf_candidates) == 0:
            return False
        
        new_nf = int(new_nf_candidates[
            self.rng.integers(0, len(new_nf_candidates))])
        
        old_total = self.recalculate_energy()
        
        removed = self.bond_network.remove_bond(old_bid)
        if removed is None:
            return False
        
        new_bid = self.bond_network.add_bond(o_atom, new_nf, old_btype)
        if new_bid < 0:
            self.bond_network.add_bond(o_atom, old_nf, old_btype)
            return False
        
        self.nl.update_bond_flags_local(o_atom, old_nf, -1)
        self.nl.update_bond_flags_local(o_atom, new_nf, old_btype)
        
        new_total = self.recalculate_energy()
        de = new_total - old_total
        
        if de <= 0.0 or self.rng.random() < exp(-de / (KB_EV * T)):
            self.current_energy = new_total
            self.bond_switch_accepted += 1
            return True
        else:
            self.bond_network.remove_bond(new_bid)
            self.bond_network.add_bond(o_atom, old_nf, old_btype)
            self.nl.update_bond_flags_local(o_atom, new_nf, -1)
            self.nl.update_bond_flags_local(o_atom, old_nf, old_btype)
            self.current_energy = old_total
            return False

    def volume_move(self, T: float) -> bool:
        """NPT volume move with Metropolis acceptance."""
        self.vol_attempts += 1
        N = self.system.N_ATOMS
        V_old = self.system.box ** 3
        
        max_dv = self.disp_volume * (5.0 if self.system.box > 40.0 else 1.0)
        log_dV = self.rng.uniform(-max_dv, max_dv)
        V_new = V_old * np.exp(log_dV)
        L_new = V_new ** (1.0 / 3.0)
        
        MAX_DENSITY = 2.80
        rho_new = (self.system.total_mass * 1.0e24 /
                   (NA_AVOGADRO * L_new ** 3))
        if rho_new > MAX_DENSITY or L_new < 2.0 * self.cutoff:
            return False
        
        old_coords = self.system.coords.copy()
        old_box = self.system.box
        old_energy = self.current_energy
        
        scale = L_new / old_box
        self.system.coords = (old_coords * scale) % L_new
        self.system.box = L_new
        
        self.nl.coords, self.nl.box = self.system.coords, self.system.box
        self.nl.update(force=True)
        
        n, s, bf = self.nl.neighbors, self.nl.starts, self.nl.bond_flags
        new_energy = total_energy_reactive(
            self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            n, s, bf, self.system.box, self.cutoff, self.wolf_alpha,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d) + self.wolf_self
        
        de = new_energy - old_energy
        dV = V_new - V_old
        bias = (self.P_ext * dV -
                N * KB_EV * T * np.log(V_new / V_old))
        delta = de + bias
        
        if delta < 0.0 or self.rng.random() < np.exp(-delta / (KB_EV * T)):
            self.current_energy = new_energy
            self.vol_accepted += 1
            return True
        else:
            self.system.coords = old_coords
            self.system.box = old_box
            self.nl.coords, self.nl.box = self.system.coords, self.system.box
            self.nl.update(force=True)
            self.current_energy = old_energy
            return False

    def _maybe_log(self):
        """Log energy and NC periodically."""
        if self.step % self.log_freq == 0:
            self.energy_log.append(
                self.current_energy / self.system.N_ATOMS)
            self.nc_log.append(self.compute_nc())

    def save_checkpoint(self, path: Path):
        """Save simulation state to checkpoint file."""
        data = {
            'step': self.step,
            'accepted': self.accepted,
            'attempts': self.attempts,
            'current_energy': self.current_energy,
            'bond_form_accepted': self.bond_form_accepted,
            'bond_form_attempts': self.bond_form_attempts,
            'bond_break_accepted': self.bond_break_accepted,
            'bond_break_attempts': self.bond_break_attempts,
            'bond_switch_accepted': self.bond_switch_accepted,
            'bond_switch_attempts': self.bond_switch_attempts,
            'vol_accepted': self.vol_accepted,
            'vol_attempts': self.vol_attempts,
            'box': self.system.box,
            'bond_network': self.bond_network.to_dict(),
            'rng_state': self.rng.bit_generator.state,
            'energy_log': self.energy_log,
            'nc_log': self.nc_log,
            'drift_log': self.drift_log,
        }
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
        np.save(path.with_name('checkpoint_coords.npy'),
                self.system.coords)

    def load_checkpoint(self, path: Path):
        """Load simulation state from checkpoint file."""
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        self.step = data['step']
        self.accepted = data['accepted']
        self.attempts = data['attempts']
        self.current_energy = data['current_energy']
        self.bond_form_accepted = data.get('bond_form_accepted', 0)
        self.bond_form_attempts = data.get('bond_form_attempts', 0)
        self.bond_break_accepted = data.get('bond_break_accepted', 0)
        self.bond_break_attempts = data.get('bond_break_attempts', 0)
        self.bond_switch_accepted = data.get('bond_switch_accepted', 0)
        self.bond_switch_attempts = data.get('bond_switch_attempts', 0)
        self.vol_accepted = data.get('vol_accepted', 0)
        self.vol_attempts = data.get('vol_attempts', 0)
        self.system.box = data['box']
        self.bond_network = BondNetwork.from_dict(data['bond_network'])
        self.bond_network.set_types(self.system.type_indices)
        self.rng.bit_generator.state = data['rng_state']
        self.energy_log = data.get('energy_log', [])
        self.nc_log = data.get('nc_log', [])
        self.drift_log = data.get('drift_log', [])
        coord_path = path.with_name('checkpoint_coords.npy')
        if coord_path.exists():
            self.system.coords = np.load(coord_path)
        self.nl = NeighborList(
            self.system.coords, self.system.box,
            self.cutoff, SKIN_DEFAULT, self.bond_network)
        return True

    def save_structure(self, path: Path, label: str = ""):
        """Save current structure to XYZ file."""
        nc = self.compute_nc()
        rho = (self.system.total_mass * 1.0e24 /
               (NA_AVOGADRO * self.system.box ** 3))
        with open(path, 'w') as f:
            f.write(f"{self.system.N_ATOMS}\n")
            f.write(f"{label} N={self.system.N_ATOMS}, rho={rho:.4f}, "
                    f"box={self.system.box:.4f}, NC={nc:.3f}, "
                    f"bonds={self.bond_network.n_active_bonds}, "
                    f"seed={self.seed}\n")
            # [FIX v5.2.1] Corrected self.coords to self.system.coords
            for i in range(self.system.N_ATOMS):
                elem = TYPE_TO_ELEM[self.system.type_indices[i]]
                f.write(f"{elem:2s} {self.system.coords[i, 0]:12.6f} "
                        f"{self.system.coords[i, 1]:12.6f} "
                        f"{self.system.coords[i, 2]:12.6f}\n")