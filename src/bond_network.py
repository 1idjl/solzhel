"""
src/bond_network.py — Dynamic covalent bond network management
================================================================
Manages the creation, breaking, and tracking of bonds (Si-O, P-O).
Includes CSR-like structure for O(1) lookups in Numba kernels.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Set
from .constants import MAX_COORD, TYPE_SI, TYPE_P, TYPE_O


class BondNetwork:
    """
    Tracks dynamic covalent bonds for Reactive MC.
    Supports coordination limits and fast CSR lookups.
    """
    
    def __init__(self, n_atoms: int):
        self.n_atoms = n_atoms
        self.bonds: Dict[int, Tuple[int, int, int]] = {}  # bid -> (i, j, type)
        self.atom_bonds: List[Set[int]] = [set() for _ in range(n_atoms)]
        self.atom_coord = np.zeros(n_atoms, dtype=np.int32)
        self._pair_to_bond: Dict[Tuple[int, int], int] = {}
        self._next_bond_id = 0
        self.n_active_bonds = 0
        self._types = None

    def set_types(self, types: np.ndarray):
        """Attach atom types for coordination checking."""
        self._types = types.copy() if isinstance(types, np.ndarray) \
            else np.array(types, dtype=np.int32)

    def add_bond(self, i: int, j: int, bond_type: int = -1) -> int:
        """Add a bond. Returns bond_id on success, -1 on failure."""
        key = (min(i, j), max(i, j))
        if key in self._pair_to_bond:
            return -1
            
        # Auto-detect bond type if not provided
        if bond_type == -1:
            if self._types is None: 
                return -1
            ti, tj = int(self._types[i]), int(self._types[j])
            if (ti in (TYPE_SI, TYPE_P) and tj == TYPE_O) or \
               (tj in (TYPE_SI, TYPE_P) and ti == TYPE_O):
                bond_type = 0 if TYPE_SI in (ti, tj) else 1
            else:
                return -1
                
        # Check coordination limits
        if self.atom_coord[i] >= self._max_coord(i): 
            return -1
        if self.atom_coord[j] >= self._max_coord(j): 
            return -1

        bond_id = self._next_bond_id
        self._next_bond_id += 1
        self.bonds[bond_id] = (i, j, bond_type)
        self.atom_bonds[i].add(bond_id)
        self.atom_bonds[j].add(bond_id)
        self.atom_coord[i] += 1
        self.atom_coord[j] += 1
        self._pair_to_bond[key] = bond_id
        self.n_active_bonds += 1
        return bond_id

    def remove_bond(self, bond_id: int) -> Optional[Tuple[int, int, int]]:
        """Remove a bond. Returns (i, j, type) or None."""
        if bond_id not in self.bonds: 
            return None
        i, j, bond_type = self.bonds[bond_id]
        del self.bonds[bond_id]
        self.atom_bonds[i].discard(bond_id)
        self.atom_bonds[j].discard(bond_id)
        self.atom_coord[i] -= 1
        self.atom_coord[j] -= 1
        del self._pair_to_bond[(min(i, j), max(i, j))]
        self.n_active_bonds -= 1
        return (i, j, bond_type)

    def has_bond(self, i: int, j: int) -> bool:
        return (min(i, j), max(i, j)) in self._pair_to_bond

    def get_bond_id(self, i: int, j: int) -> int:
        return self._pair_to_bond.get((min(i, j), max(i, j)), -1)

    def get_bonded_neighbors(self, i: int):
        """Returns list of (neighbor_idx, bond_id, bond_type)."""
        return [(bj if bi == i else bi, bid, btype) 
                for bid, (bi, bj, btype) in self.bonds.items() 
                if bid in self.atom_bonds[i]]

    def get_coordination(self, i: int) -> int:
        return int(self.atom_coord[i])

    def get_all_bonds(self):
        """Returns list of (bond_id, i, j, bond_type)."""
        return [(bid, i, j, bt) for bid, (i, j, bt) in self.bonds.items()]

    def get_bond_arrays(self):
        """Simple arrays for Numba (O(N_bonds) iteration)."""
        n = len(self.bonds)
        bond_i = np.empty(n, dtype=np.int32)
        bond_j = np.empty(n, dtype=np.int32)
        bond_type = np.empty(n, dtype=np.int32)
        for idx, (bid, (i, j, bt)) in enumerate(self.bonds.items()):
            bond_i[idx] = i
            bond_j[idx] = j
            bond_type[idx] = bt
        return bond_i, bond_j, bond_type

    def get_bond_csr(self):
        """
        Build CSR-like arrays for O(1) bond lookup in Numba.
        Returns: bond_starts, bond_partners, bond_types, bond_ids
        """
        counts = np.zeros(self.n_atoms, dtype=np.int32)
        for i, j, _ in self.bonds.values():
            counts[i] += 1
            counts[j] += 1

        bond_starts = np.zeros(self.n_atoms + 1, dtype=np.int32)
        bond_starts[1:] = np.cumsum(counts)

        total_entries = bond_starts[-1]
        bond_partners = np.empty(total_entries, dtype=np.int32)
        bond_types = np.empty(total_entries, dtype=np.int32)
        bond_ids = np.empty(total_entries, dtype=np.int32)

        fill = bond_starts[:-1].copy()
        for bid, (i, j, bt) in self.bonds.items():
            bond_partners[fill[i]] = j
            bond_types[fill[i]] = bt
            bond_ids[fill[i]] = bid
            fill[i] += 1
            
            bond_partners[fill[j]] = i
            bond_types[fill[j]] = bt
            bond_ids[fill[j]] = bid
            fill[j] += 1

        return bond_starts, bond_partners, bond_types, bond_ids

    def build_bond_flags(self, neighbors, starts):
        """Build full bond_flags array for NeighborList."""
        bond_flags = np.full(len(neighbors), -1, dtype=np.int32)
        for i in range(self.n_atoms):
            for p in range(starts[i], starts[i + 1]):
                j = neighbors[p]
                if self.has_bond(i, j):
                    bid = self.get_bond_id(i, j)
                    _, _, btype = self.bonds[bid]
                    bond_flags[p] = btype
        return bond_flags

    # ========================================================================
    # SERIALIZATION (for Checkpointing)
    # ========================================================================
    def to_dict(self):
        return {
            'bonds': [(bid, i, j, bt) for bid, (i, j, bt) in self.bonds.items()],
            'n_atoms': self.n_atoms,
        }

    @classmethod
    def from_dict(cls, data):
        bn = cls(data['n_atoms'])
        for bid, i, j, bt in data['bonds']:
            bn.bonds[bid] = (i, j, bt)
            bn.atom_bonds[i].add(bid)
            bn.atom_bonds[j].add(bid)
            bn.atom_coord[i] += 1
            bn.atom_coord[j] += 1
            bn._pair_to_bond[(min(i, j), max(i, j))] = bid
            bn._next_bond_id = max(bn._next_bond_id, bid + 1)
            bn.n_active_bonds += 1
        return bn

    def _max_coord(self, i: int) -> int:
        if self._types is None: 
            return 100
        return MAX_COORD.get(int(self._types[i]), 100)