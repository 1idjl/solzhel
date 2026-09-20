"""
src/neighbor_list.py — Neighbor list management with Verlet skin
=================================================================
Builds and updates the neighbor list using scipy.spatial.cKDTree.
Supports dynamic rebuilding based on maximum atomic displacement (skin).
"""

import numpy as np
from scipy.spatial import cKDTree
from .constants import SKIN_DEFAULT


class NeighborList:
    """
    Verlet neighbor list for efficient pair interactions.
    Rebuilds when max displacement exceeds skin/3.
    """
    
    def __init__(self, coords, box, cutoff, skin=SKIN_DEFAULT, bond_network=None):
        self.coords = coords
        self.box = box
        self.cutoff = cutoff
        self.skin = skin
        self.sc = cutoff + skin  # Search cutoff
        self.bond_network = bond_network
        
        self.neighbors = None
        self.starts = None
        self.bond_flags = None
        self.ref_coords = None
        self.rebuild_count = 0
        
        self._build()

    def _build(self):
        """Build neighbor list from scratch using cKDTree."""
        tree = cKDTree(self.coords, boxsize=[self.box, self.box, self.box])
        pairs = tree.query_pairs(self.sc, output_type='ndarray')
        
        n = len(self.coords)
        counts = np.zeros(n, dtype=np.int32)
        for i, j in pairs:
            counts[i] += 1
            counts[j] += 1
            
        # CSR-like format for Numba
        starts = np.zeros(n + 1, dtype=np.int32)
        starts[1:] = np.cumsum(counts)
        
        neighbors = np.empty(starts[-1], dtype=np.int32)
        fill = starts[:-1].copy()
        for i, j in pairs:
            neighbors[fill[i]] = j
            fill[i] += 1
            neighbors[fill[j]] = i
            fill[j] += 1
            
        self.neighbors = neighbors
        self.starts = starts
        self.ref_coords = self.coords.copy()
        self.rebuild_count += 1
        
        # Build bond flags if bond network is attached
        if self.bond_network is not None:
            self.bond_flags = self.bond_network.build_bond_flags(
                self.neighbors, self.starts)
        else:
            self.bond_flags = np.full(len(neighbors), -1, dtype=np.int32)
            
        return neighbors, starts, self.bond_flags

    def update(self, force=False):
        """Update neighbor list if needed."""
        if force or self._needs_rebuild():
            return self._build()
        return self.neighbors, self.starts, self.bond_flags

    def update_bond_flags_local(self, i, j, bond_type):
        """
        O(N_neigh) local update instead of O(N_pairs) full rebuild.
        Calls the Numba kernel for speed.
        """
        # Lazy import to avoid circular dependencies before numba_kernels is loaded
        from .numba_kernels import update_bond_flag_pair
        update_bond_flag_pair(
            i, j, bond_type, 
            self.neighbors, self.starts, self.bond_flags
        )

    def _needs_rebuild(self):
        """Check if any atom has moved more than skin/3."""
        if self.ref_coords is None:
            return True
        dr = self.coords - self.ref_coords
        dr -= self.box * np.round(dr / self.box)
        return np.sqrt(np.sum(dr ** 2, axis=1)).max() > self.skin / 3.0