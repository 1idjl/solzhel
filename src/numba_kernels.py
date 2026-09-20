"""
src/numba_kernels.py — Numba-accelerated kernels for energy and search
========================================================================
Contains all @njit functions used by the simulator.
Using Numba here provides a 100-1000x speedup over pure Python loops.
"""

import numpy as np
from numba import njit
from math import erfc, exp, sqrt, pi

# Import constants directly for use inside njit functions
from .constants import (
    KE_COULOMB, SWITCH_DR,
    TYPE_SI, TYPE_P, TYPE_O,
    BOND_HARMONIC_K_SI_O, BOND_HARMONIC_K_P_O,
    BOND_HARMONIC_R0_SI_O, BOND_HARMONIC_R0_P_O,
)

# ============================================================================
# 1. PAIR POTENTIALS
# ============================================================================

@njit(fastmath=True, cache=True)
def zbl_repulsion(r, Zi, Zj, zbl_a0, zbl_c, zbl_d):
    """Ziegler-Biersack-Littmark (ZBL) universal repulsive potential."""
    if r < 1.0e-6:
        return 1.0e8
    a = zbl_a0 / (Zi ** 0.23 + Zj ** 0.23)
    x = r / a
    phi = (zbl_c[0] * exp(-zbl_d[0] * x) +
           zbl_c[1] * exp(-zbl_d[1] * x) +
           zbl_c[2] * exp(-zbl_d[2] * x) +
           zbl_c[3] * exp(-zbl_d[3] * x))
    return KE_COULOMB * Zi * Zj * phi / r


@njit(fastmath=True, cache=True)
def wolf_coulomb(qi, qj, r, alpha, cutoff):
    """Wolf summation for damped, truncated Coulomb interactions."""
    if r >= cutoff or r < 1.0e-12:
        return 0.0
    ar = alpha * r
    ac = alpha * cutoff
    er = erfc(ar)
    ec = erfc(ac)
    t1 = er / r
    t2 = ec / cutoff
    t3 = ((ec / (cutoff * cutoff)) +
          (2.0 * alpha / sqrt(pi)) * exp(-ac * ac) / cutoff) * (r - cutoff)
    return KE_COULOMB * qi * qj * (t1 - t2 + t3)


@njit(fastmath=True, cache=True)
def bonded_energy_pair(r, bond_type):
    """Harmonic bonded potential: E = 0.5 * k * (r - r0)^2"""
    if bond_type == 0:  # Si-O
        k = BOND_HARMONIC_K_SI_O
        r0 = BOND_HARMONIC_R0_SI_O
    elif bond_type == 1:  # P-O
        k = BOND_HARMONIC_K_P_O
        r0 = BOND_HARMONIC_R0_P_O
    else:
        return 0.0
    dr = r - r0
    return 0.5 * k * dr * dr


@njit(fastmath=True, cache=True)
def pair_energy_nonbonded(r, qi, qj, ti, tj, Zi, Zj,
                          A_mat, F_mat, C_mat, R_HARD_MAT,
                          cutoff, alpha, zbl_a0, zbl_c, zbl_d):
    """Full non-bonded pair energy: Buckingham + Coulomb + ZBL switching."""
    if r >= cutoff or r < 1.0e-12:
        return 0.0
    
    r_in = R_HARD_MAT[ti, tj]
    r_out = r_in + SWITCH_DR
    
    if r < r_out:
        e_zbl = zbl_repulsion(r, Zi, Zj, zbl_a0, zbl_c, zbl_d)
        if r < r_in:
            return e_zbl
            
        A = A_mat[ti, tj]
        F = F_mat[ti, tj]
        C = C_mat[ti, tj]
        
        e_buck = 0.0
        if A > 0.0 and F > 1.0e-12:
            e_buck += A * exp(-r / F)
        if C > 0.0:
            e_buck -= C / (r ** 6)
            
        e_coul = wolf_coulomb(qi, qj, r, alpha, cutoff)
        e_full = e_buck + e_coul
        
        # Smooth switching function (prevents energy discontinuities)
        x_switch = (r - r_in) / SWITCH_DR
        s = x_switch ** 3 * (10.0 - 15.0 * x_switch + 6.0 * x_switch ** 2)
        return s * e_full + (1.0 - s) * e_zbl
    else:
        A = A_mat[ti, tj]
        F = F_mat[ti, tj]
        C = C_mat[ti, tj]
        
        e_buck = 0.0
        if A > 0.0 and F > 1.0e-12:
            e_buck += A * exp(-r / F)
        if C > 0.0:
            e_buck -= C / (r ** 6)
            
        return e_buck + wolf_coulomb(qi, qj, r, alpha, cutoff)


# ============================================================================
# 2. ENERGY CALCULATIONS
# ============================================================================

@njit(fastmath=True, cache=True)
def local_energy_reactive(idx, coords, charges, types, type_Z,
                          A_mat, F_mat, C_mat, R_HARD_MAT,
                          box, cutoff, alpha,
                          neighbors, starts, bond_flags,
                          zbl_a0, zbl_c, zbl_d):
    """Local energy of atom idx including bonded and non-bonded terms."""
    e = 0.0
    qi = charges[idx]
    ti = types[idx]
    Zi = type_Z[ti]
    xi, yi, zi = coords[idx]

    for p in range(starts[idx], starts[idx + 1]):
        j = neighbors[p]
        dx = xi - coords[j, 0]
        dy = yi - coords[j, 1]
        dz = zi - coords[j, 2]
        
        # Minimum image convention
        dx -= box * round(dx / box)
        dy -= box * round(dy / box)
        dz -= box * round(dz / box)
        
        r2 = dx * dx + dy * dy + dz * dz
        if r2 >= cutoff * cutoff:
            continue
        r = sqrt(r2)
        bf = bond_flags[p]

        if bf >= 0:
            # Bonded pair: harmonic + Coulomb only (no Buckingham/ZBL)
            e += bonded_energy_pair(r, bf)
            e += wolf_coulomb(qi, charges[j], r, alpha, cutoff)
        else:
            # Non-bonded pair: full interaction
            tj = types[j]
            Zj = type_Z[tj]
            e += pair_energy_nonbonded(
                r, qi, charges[j], ti, tj, Zi, Zj,
                A_mat, F_mat, C_mat, R_HARD_MAT,
                cutoff, alpha, zbl_a0, zbl_c, zbl_d)

    # Wolf self-energy correction
    e += -KE_COULOMB * (alpha / sqrt(pi)) * qi * qi
    return e


@njit(fastmath=True, cache=True)
def total_energy_reactive(coords, charges, types, type_Z,
                          A_mat, F_mat, C_mat, R_HARD_MAT,
                          neighbors, starts, bond_flags,
                          box, cutoff, alpha,
                          zbl_a0, zbl_c, zbl_d):
    """Total energy of the system (used for full recalculations)."""
    e = 0.0
    n = coords.shape[0]
    
    for i in range(n):
        xi, yi, zi = coords[i]
        qi = charges[i]
        ti = types[i]
        Zi = type_Z[ti]
        
        for p in range(starts[i], starts[i + 1]):
            j = neighbors[p]
            if j <= i:
                continue  # Avoid double counting
                
            dx = xi - coords[j, 0]
            dy = yi - coords[j, 1]
            dz = zi - coords[j, 2]
            dx -= box * round(dx / box)
            dy -= box * round(dy / box)
            dz -= box * round(dz / box)
            
            r2 = dx * dx + dy * dy + dz * dz
            if r2 >= cutoff * cutoff:
                continue
            r = sqrt(r2)
            bf = bond_flags[p]
            
            if bf >= 0:
                e += bonded_energy_pair(r, bf)
                e += wolf_coulomb(qi, charges[j], r, alpha, cutoff)
            else:
                tj = types[j]
                Zj = type_Z[tj]
                e += pair_energy_nonbonded(
                    r, qi, charges[j], ti, tj, Zi, Zj,
                    A_mat, F_mat, C_mat, R_HARD_MAT,
                    cutoff, alpha, zbl_a0, zbl_c, zbl_d)
                    
    return e


# ============================================================================
# 3. NETWORK CONNECTIVITY (NC)
# ============================================================================

@njit(fastmath=True, cache=True)
def compute_nc_numba(types, bond_i_arr, bond_j_arr, n_atoms):
    """
    Compute Network Connectivity: NC = 2 * N_BO / N_NF
    where BO = Oxygen bonded to exactly 2 Network Formers (Si or P).
    """
    n_nf = 0
    for i in range(n_atoms):
        if types[i] == TYPE_SI or types[i] == TYPE_P:
            n_nf += 1

    if n_nf == 0:
        return 0.0

    # Count NF bonds per oxygen
    o_nf_count = np.zeros(n_atoms, dtype=np.int32)
    n_bonds = len(bond_i_arr)
    
    for b in range(n_bonds):
        i = bond_i_arr[b]
        j = bond_j_arr[b]
        # If i is O and j is NF
        if types[i] == TYPE_O and (types[j] == TYPE_SI or types[j] == TYPE_P):
            o_nf_count[i] += 1
        # If j is O and i is NF
        if types[j] == TYPE_O and (types[i] == TYPE_SI or types[i] == TYPE_P):
            o_nf_count[j] += 1

    # Count BOs (O with exactly 2 NF bonds)
    n_bo = 0
    for i in range(n_atoms):
        if types[i] == TYPE_O and o_nf_count[i] == 2:
            n_bo += 1

    return (2.0 * n_bo) / n_nf


# ============================================================================
# 4. LOCAL BOND FLAG UPDATE
# ============================================================================

@njit(fastmath=True, cache=True)
def update_bond_flag_pair(i, j, bond_type, neighbors, starts, bond_flags):
    """Update bond_flags array locally for a single pair (i, j). O(N_neigh)."""
    # Update i -> j
    for p in range(starts[i], starts[i + 1]):
        if neighbors[p] == j:
            bond_flags[p] = bond_type
            break
            
    # Update j -> i
    for p in range(starts[j], starts[j + 1]):
        if neighbors[p] == i:
            bond_flags[p] = bond_type
            break


# ============================================================================
# 5. CSR-BASED REACTIVE SEARCH (Blazing fast O(N_neigh + N_coord))
# ============================================================================

@njit(fastmath=True, cache=True)
def find_o_candidates_for_nf_fast(nf, coords, types, box,
                                  neighbors, starts,
                                  bond_starts, bond_partners,
                                  bond_cutoff):
    """Find O candidates near a Network Former using CSR structure."""
    max_cand = 50
    cand_dist = np.empty(max_cand, dtype=np.float64)
    cand_idx = np.empty(max_cand, dtype=np.int32)
    n_cand = 0

    for p in range(starts[nf], starts[nf + 1]):
        j = neighbors[p]
        if j == nf or types[j] != TYPE_O:
            continue

        # Check if already bonded and count O coordination using CSR
        already_bonded = False
        o_coord = 0
        for bp in range(bond_starts[j], bond_starts[j + 1]):
            o_coord += 1
            if bond_partners[bp] == nf:
                already_bonded = True
                break
                
        if already_bonded or o_coord >= 2:
            continue

        # Distance check with minimum image
        dx = coords[j, 0] - coords[nf, 0]
        dy = coords[j, 1] - coords[nf, 1]
        dz = coords[j, 2] - coords[nf, 2]
        dx -= box * round(dx / box)
        dy -= box * round(dy / box)
        dz -= box * round(dz / box)
        r = sqrt(dx*dx + dy*dy + dz*dz)
        
        if r < bond_cutoff:
            if n_cand < max_cand:
                cand_dist[n_cand] = r
                cand_idx[n_cand] = j
                n_cand += 1

    return cand_dist[:n_cand], cand_idx[:n_cand]


@njit(fastmath=True, cache=True)
def find_nf_candidates_for_o_fast(o_atom, coords, types, box,
                                  neighbors, starts,
                                  bond_starts, bond_partners,
                                  bond_cutoff, exclude1, exclude2):
    """Find NF candidates near an Oxygen using CSR structure."""
    max_cand = 50
    cand_idx = np.empty(max_cand, dtype=np.int32)
    n_cand = 0

    for p in range(starts[o_atom], starts[o_atom + 1]):
        j = neighbors[p]
        if j == o_atom or j == exclude1 or j == exclude2:
            continue
        if types[j] != TYPE_SI and types[j] != TYPE_P:
            continue

        # Check if already bonded and count NF coordination using CSR
        already_bonded = False
        nf_coord = 0
        for bp in range(bond_starts[j], bond_starts[j + 1]):
            nf_coord += 1
            if bond_partners[bp] == o_atom:
                already_bonded = True
                break
                
        if already_bonded or nf_coord >= 4:
            continue

        # Distance check with minimum image
        dx = coords[j, 0] - coords[o_atom, 0]
        dy = coords[j, 1] - coords[o_atom, 1]
        dz = coords[j, 2] - coords[o_atom, 2]
        dx -= box * round(dx / box)
        dy -= box * round(dy / box)
        dz -= box * round(dz / box)
        r = sqrt(dx*dx + dy*dy + dz*dz)
        
        if r < bond_cutoff:
            if n_cand < max_cand:
                cand_idx[n_cand] = j
                n_cand += 1

    return cand_idx[:n_cand]


@njit(fastmath=True, cache=True)
def find_bonded_nfs_for_o_fast(o_atom, types, bond_starts, bond_partners):
    """Find up to 2 Network Formers bonded to an Oxygen using CSR."""
    nf1, nf2 = -1, -1
    n_found = 0
    
    for bp in range(bond_starts[o_atom], bond_starts[o_atom + 1]):
        partner = bond_partners[bp]
        if types[partner] == TYPE_SI or types[partner] == TYPE_P:
            if n_found == 0:
                nf1 = partner
            elif n_found == 1:
                nf2 = partner
            n_found += 1
            if n_found >= 2:
                break
                
    return nf1, nf2