#!/usr/bin/env python3
"""
generator/solgel_generator.py — Sol-Gel 45S5 Structure Generator v5.0 (Final)
==============================================================================
Generates LOW-DENSITY, PARTIALLY-CONNECTED initial structures for
Reactive Monte Carlo simulation of sol-gel 45S5 bioglass.

Key features:
- Exact 45S5 stoichiometry with scale support
- Target NC control with oxygen budget management
- BO bending (~145° Si-O-Si angle)
- NBO prioritization by coordination deficit
- Safe Relaxation to eliminate O-O overlaps
- Bonds stored as (nf_idx, o_idx) pairs for Reactive MC
"""

import sys
import os
import numpy as np
import argparse
import json
from pathlib import Path
from scipy.spatial import cKDTree
from math import sqrt, sin, cos, pi

# ============================================================================
# CONSTANTS
# ============================================================================
NA = 6.02214076e23

# Base composition (scale=1 -> 2835 atoms)
# 45SiO2 - 24.5Na2O - 24.5CaO - 6P2O5 (wt%)
BASE_N_SI = 461
BASE_N_P  = 52
BASE_N_NA = 488
BASE_N_CA = 269
BASE_N_O  = 1565

MASSES = {"Si": 28.0855, "P": 30.97376, "Na": 22.98977,
          "Ca": 40.078, "O": 15.999}

SI_O_BOND = 1.61
P_O_BOND  = 1.49
NA_O_BOND = 2.37
CA_O_BOND = 2.40

MIN_SI_SI = 3.05
MIN_SI_P  = 2.95
MIN_P_P   = 3.40

SI_SI_MIN, SI_SI_MAX = 3.00, 4.40
SI_P_MIN,  SI_P_MAX  = 2.90, 4.10
P_P_MIN,   P_P_MAX   = 2.85, 3.80

NBO_SAFE_DIST = 2.50
NBO_O_OVERLAP = 1.95

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def minimum_image(dr, box):
    return dr - box * np.round(dr / box)

def random_rotation(rng):
    u1, u2, u3 = rng.random(3)
    q0 = sqrt(1.0 - u1) * sin(2.0 * pi * u2)
    q1 = sqrt(1.0 - u1) * cos(2.0 * pi * u2)
    q2 = sqrt(u1) * sin(2.0 * pi * u3)
    q3 = sqrt(u1) * cos(2.0 * pi * u3)
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1.0 - 2.0 * (q2*q2 + q3*q3)
    R[0, 1] = 2.0 * (q1*q2 - q0*q3)
    R[0, 2] = 2.0 * (q1*q3 + q0*q2)
    R[1, 0] = 2.0 * (q1*q2 + q0*q3)
    R[1, 1] = 1.0 - 2.0 * (q1*q1 + q3*q3)
    R[1, 2] = 2.0 * (q2*q3 - q0*q1)
    R[2, 0] = 2.0 * (q1*q3 - q0*q2)
    R[2, 1] = 2.0 * (q2*q3 + q0*q1)
    R[2, 2] = 1.0 - 2.0 * (q1*q1 + q2*q2)
    return R

# ============================================================================
# STEP 1: NETWORK FORMERS
# ============================================================================
def place_network_formers(n_si, n_p, box, rng):
    n_total = n_si + n_p
    coords = np.zeros((n_total, 3), dtype=np.float64)
    types = np.zeros(n_total, dtype=np.int32)  # 0=Si, 3=P
    placed = 0
    for i in range(n_total):
        is_p = (i >= n_si)
        success = False
        for _ in range(200000):
            pos = rng.uniform(0.0, box, 3)
            if placed == 0:
                coords[placed] = pos
                types[placed] = 3 if is_p else 0
                placed += 1
                success = True
                break
            d = coords[:placed] - pos
            d = minimum_image(d, box)
            r = np.sqrt(np.sum(d * d, axis=1))
            valid = True
            for k in range(placed):
                if types[k] == 3 and is_p:
                    if r[k] < MIN_P_P: valid = False; break
                elif (types[k] == 3 and not is_p) or (types[k] == 0 and is_p):
                    if r[k] < MIN_SI_P: valid = False; break
                else:
                    if r[k] < MIN_SI_SI: valid = False; break
            if valid:
                coords[placed] = pos
                types[placed] = 3 if is_p else 0
                placed += 1
                success = True
                break
        if not success:
            best_pos, best_min = None, -1.0
            for _ in range(5000):
                p = rng.uniform(0.0, box, 3)
                d = coords[:placed] - p
                d = minimum_image(d, box)
                r = np.sqrt(np.sum(d * d, axis=1)).min()
                if r > best_min:
                    best_min = r
                    best_pos = p
            coords[placed] = best_pos
            types[placed] = 3 if is_p else 0
            placed += 1
    return coords, types

# ============================================================================
# STEP 2: BUILD TOPOLOGY
# ============================================================================
def build_topology_with_nc(nf_coords, nf_types, box, n_bo_target, n_siop_max, rng):
    n_nf = len(nf_coords)
    tree = cKDTree(nf_coords, boxsize=box)
    pairs = tree.query_pairs(5.0, output_type="ndarray")

    si_si, si_p, p_p = [], [], []
    for i, j in pairs:
        ti, tj = nf_types[i], nf_types[j]
        d = minimum_image(nf_coords[j] - nf_coords[i], box)
        r = np.linalg.norm(d)
        if ti == 0 and tj == 0:
            if SI_SI_MIN <= r <= SI_SI_MAX:
                si_si.append((abs(r - 3.22) + rng.uniform(0.0, 0.05), i, j))
        elif ti != tj:
            if SI_P_MIN <= r <= SI_P_MAX:
                si_p.append((abs(r - 3.11) + rng.uniform(0.0, 0.05), i, j))
        else:
            if P_P_MIN <= r <= P_P_MAX:
                p_p.append((abs(r - 3.00) + rng.uniform(0.0, 0.05), i, j))

    si_si.sort(); si_p.sort(); p_p.sort()
    deg = np.zeros(n_nf, dtype=np.int32)
    edges = []
    selected = set()

    n_siop = 0
    for s, i, j in si_p:
        if n_siop >= n_siop_max: break
        if deg[i] < 4 and deg[j] < 4:
            key = (min(i, j), max(i, j))
            if key not in selected:
                selected.add(key); deg[i] += 1; deg[j] += 1
                edges.append((i, j)); n_siop += 1

    n_siosi = 0
    n_siosi_target = n_bo_target - n_siop
    for s, i, j in si_si:
        if n_siosi >= n_siosi_target: break
        if deg[i] < 4 and deg[j] < 4:
            key = (min(i, j), max(i, j))
            if key not in selected:
                selected.add(key); deg[i] += 1; deg[j] += 1
                edges.append((i, j)); n_siosi += 1

    print(f"    Topology built: {len(edges)} BOs (Target: {n_bo_target})")
    return edges, deg

# ============================================================================
# STEP 3: BRIDGING OXYGENS
# ============================================================================
def place_bridging_oxygens(nf_coords, nf_types, edges, box, rng):
    bo_coords = []
    for i, j in edges:
        ti, tj = nf_types[i], nf_types[j]
        r_i = SI_O_BOND if ti == 0 else P_O_BOND
        r_j = SI_O_BOND if tj == 0 else P_O_BOND
        d = minimum_image(nf_coords[j] - nf_coords[i], box)
        dist = np.linalg.norm(d)
        if dist < 1.0e-6: continue
        frac = r_i / (r_i + r_j)
        o_pos = nf_coords[i] + d * frac

        # Bending for realistic ~145° Si-O-Si angle
        rand_vec = rng.normal(size=3)
        proj = np.dot(rand_vec, d) / (dist ** 2) * d
        perp = rand_vec - proj
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 1e-6:
            perp = perp / perp_norm
            disp_mag = rng.uniform(0.35, 0.55)
            o_pos = o_pos + perp * disp_mag

        bo_coords.append(o_pos % box)

    if bo_coords:
        return np.array(bo_coords, dtype=np.float64)
    return np.empty((0, 3), dtype=np.float64)

# ============================================================================
# STEP 4: TERMINAL OXYGENS (NBOs)
# ============================================================================
def place_terminal_oxygens(nf_coords, nf_types, deg, box, bo_coords, rng, n_o_target):
    term_coords = []
    term_owners = []
    n_nf = len(nf_coords)
    nf_tree = cKDTree(nf_coords, boxsize=box)
    all_o = list(bo_coords) if len(bo_coords) > 0 else []
    o_tree = cKDTree(np.array(all_o), boxsize=box) if all_o else None

    TETRA = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64)
    TETRA /= np.linalg.norm(TETRA, axis=1)[:, None]

    n_bo = len(bo_coords)
    n_nbo_budget = n_o_target - n_bo

    nbo_tasks = []
    for nf in range(n_nf):
        n_needed = max(0, 4 - int(deg[nf]))
        for _ in range(n_needed):
            nbo_tasks.append(nf)

    rng.shuffle(nbo_tasks)
    deficit = {nf: max(0, 4 - int(deg[nf])) for nf in range(n_nf)}
    nbo_tasks.sort(key=lambda x: deficit[x], reverse=True)
    nbo_tasks = nbo_tasks[:max(0, n_nbo_budget)]

    nbo_by_nf = {}
    for nf in nbo_tasks:
        nbo_by_nf[nf] = nbo_by_nf.get(nf, 0) + 1

    for nf, n_nbo_needed in nbo_by_nf.items():
        pos = nf_coords[nf]
        bond = SI_O_BOND if nf_types[nf] == 0 else P_O_BOND
        placed_count = 0

        for _ in range(200):
            if placed_count >= n_nbo_needed: break
            R = random_rotation(rng)
            dirs = [R @ TETRA[k] for k in range(4)]
            rng.shuffle(dirs)
            for direction in dirs:
                if placed_count >= n_nbo_needed: break
                d = np.array(direction) + 0.08 * (rng.random(3) - 0.5)
                d = d / np.linalg.norm(d)
                o_pos = (pos + bond * d) % box

                nearby = nf_tree.query_ball_point(o_pos, NBO_SAFE_DIST)
                if any(idx != nf for idx in nearby): continue
                if o_tree is not None and len(o_tree.query_ball_point(o_pos, NBO_O_OVERLAP)) > 0: continue

                term_coords.append(o_pos)
                term_owners.append(nf)
                all_o.append(o_pos)
                placed_count += 1
                if len(all_o) % 100 == 0:
                    o_tree = cKDTree(np.array(all_o), boxsize=box)

    return np.array(term_coords, dtype=np.float64), np.array(term_owners, dtype=np.int32)

# ============================================================================
# STEP 5: MODIFIERS
# ============================================================================
def place_modifiers(n_na, n_ca, sites, box, rng):
    if len(sites) == 0:
        return rng.uniform(0.0, box, size=(n_na + n_ca, 3))

    static_tree = cKDTree(sites, boxsize=box)
    mod_coords = []

    for elem, n_mod, target_dist in [("Na", n_na, NA_O_BOND), ("Ca", n_ca, CA_O_BOND)]:
        min_mod_dist = 2.2
        for _ in range(n_mod):
            pos = None
            for _ in range(5000):
                site = sites[rng.integers(len(sites))]
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                dist = target_dist + rng.uniform(-0.15, 0.25)
                cand = (site + direction * dist) % box

                dists = static_tree.query(cand, k=10, distance_upper_bound=3.5)[0]
                dists = dists[dists < np.inf]
                if len(dists) == 0 or np.min(dists) > 3.2: continue
                if np.min(dists) < 1.8: continue

                ok = True
                if mod_coords:
                    for m in mod_coords:
                        if np.linalg.norm(minimum_image(cand - m, box)) < min_mod_dist:
                            ok = False; break
                if ok:
                    pos = cand; break
            if pos is None:
                pos = rng.uniform(0.0, box, 3)
            mod_coords.append(pos)

    return np.array(mod_coords, dtype=np.float64)

# ============================================================================
# STEP 6: SAFE RELAXATION (CRITICAL FIX FOR O-O OVERLAPS)
# ============================================================================
# ============================================================================
# STEP 6: ADVANCED SAFE RELAXATION (Multi-Phase with Shake & Emergency)
# ============================================================================
def relax_o_overlaps_safe(coords, types, box, fixed_mask, 
                          r_hard_oo=2.00,  # Realistic O-O hard core
                          max_overlap_pairs=5):  # Target: fewer than 5 overlaps
    """
    Advanced multi-phase relaxation to eliminate O-O overlaps.
    
    Phase 1: Fast push with large step (escape local minima)
    Phase 2: Shake stuck atoms with random displacement
    Phase 3: Fine relaxation with small step
    Phase 4: Emergency relocation for remaining overlaps
    """
    new_coords = coords.copy()
    n_atoms = len(coords)
    
    # Hard-core radii matrix [Å]
    r_hard = np.full((6, 6), 1.2, dtype=np.float64)
    r_hard[4, 4] = r_hard_oo   # O-O: realistic hard core
    r_hard[0, 4] = 1.50; r_hard[4, 0] = 1.50  # Si-O
    r_hard[3, 4] = 1.40; r_hard[4, 3] = 1.40  # P-O
    r_hard[1, 4] = 1.60; r_hard[4, 1] = 1.60  # Ca-O
    r_hard[2, 4] = 1.60; r_hard[4, 2] = 1.60  # Na-O
    r_hard[0, 0] = 2.20  # Si-Si
    r_hard[3, 3] = 2.20  # P-P
    r_hard[0, 3] = 2.20; r_hard[3, 0] = 2.20  # Si-P
    r_hard[1, 1] = 2.00  # Ca-Ca
    r_hard[2, 2] = 2.00  # Na-Na
    r_hard[1, 2] = 2.00; r_hard[2, 1] = 2.00  # Ca-Na
    
    def count_overlaps():
        """Count pairs violating hard-core distances."""
        tree = cKDTree(new_coords, boxsize=box)
        pairs = tree.query_pairs(2.5, output_type="ndarray")
        overlaps = []
        for i, j in pairs:
            if fixed_mask[i] and fixed_mask[j]:
                continue
            d = minimum_image(new_coords[i] - new_coords[j], box)
            r = np.linalg.norm(d)
            limit = r_hard[types[i], types[j]]
            if r < limit and r > 1.0e-6:
                overlaps.append((i, j, r, limit))
        return overlaps
    
    def push_phase(iterations, step_size):
        """Phase 1 & 3: Deterministic push based on overlap."""
        for it in range(iterations):
            overlaps = count_overlaps()
            if len(overlaps) == 0:
                print(f"      Converged after {it + 1} iterations (step={step_size:.3f})")
                return True
            
            moved_atoms = set()
            for i, j, r, limit in overlaps:
                d = minimum_image(new_coords[i] - new_coords[j], box)
                f = d * (step_size * (limit - r) / r)
                
                if not fixed_mask[i] and i not in moved_atoms:
                    new_coords[i] = (new_coords[i] + f) % box
                    moved_atoms.add(i)
                if not fixed_mask[j] and j not in moved_atoms:
                    new_coords[j] = (new_coords[j] - f) % box
                    moved_atoms.add(j)
        return False
    
    def shake_phase(atoms_to_shake, shake_magnitude=0.5):
        """Phase 2: Random displacement to escape local minima."""
        if len(atoms_to_shake) == 0:
            return
        print(f"      Shaking {len(atoms_to_shake)} stuck atoms "
              f"(magnitude={shake_magnitude:.2f} A)...")
        for i in atoms_to_shake:
            if fixed_mask[i]:
                continue
            # Random displacement
            rnd_disp = np.random.default_rng().normal(0, shake_magnitude, size=3)
            new_coords[i] = (new_coords[i] + rnd_disp) % box
    
    def emergency_relocate(atoms_to_relocate, static_coords, static_types):
        """Phase 4: Completely relocate problematic atoms to safe positions."""
        if len(atoms_to_relocate) == 0:
            return
        print(f"      Emergency relocating {len(atoms_to_relocate)} atoms...")
        static_tree = cKDTree(static_coords, boxsize=box)
        
        for i in atoms_to_relocate:
            if fixed_mask[i]:
                continue
            # Try 1000 random positions
            best_pos = None
            best_min_dist = 0.0
            
            for _ in range(1000):
                cand = np.random.default_rng().uniform(0.0, box, 3)
                # Check distance to all static atoms
                dists = static_tree.query(cand, k=1)[0]
                if dists < 1.5:
                    continue
                
                # Check distance to all other dynamic atoms
                min_dist = dists
                for j in range(n_atoms):
                    if j == i or fixed_mask[j]:
                        continue
                    d = minimum_image(cand - new_coords[j], box)
                    r = np.linalg.norm(d)
                    limit = r_hard[types[i], types[j]]
                    if r < limit:
                        min_dist = -1
                        break
                    min_dist = min(min_dist, r - limit)
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_pos = cand
                    if best_min_dist > 0.5:  # Good enough
                        break
            
            if best_pos is not None:
                new_coords[i] = best_pos
    
    # ========================================================================
    # EXECUTE 4-PHASE RELAXATION
    # ========================================================================
    print("    Phase 1: Fast push (escape local minima)...")
    push_phase(iterations=300, step_size=0.15)
    overlaps = count_overlaps()
    print(f"      Remaining overlaps: {len(overlaps)}")
    
    if len(overlaps) > max_overlap_pairs:
        print("    Phase 2: Shake stuck atoms...")
        stuck_atoms = set()
        for i, j, _, _ in overlaps:
            stuck_atoms.add(i)
            stuck_atoms.add(j)
        shake_phase(stuck_atoms, shake_magnitude=0.6)
        
        print("    Phase 3: Fine relaxation...")
        push_phase(iterations=300, step_size=0.05)
        overlaps = count_overlaps()
        print(f"      Remaining overlaps: {len(overlaps)}")
    
    if len(overlaps) > max_overlap_pairs:
        print("    Phase 4: Emergency relocation...")
        problem_atoms = set()
        for i, j, _, _ in overlaps:
            problem_atoms.add(i)
            problem_atoms.add(j)
        
        # Build static reference (fixed atoms + non-problem dynamic atoms)
        static_mask = fixed_mask.copy()
        for i in problem_atoms:
            static_mask[i] = True
        
        static_coords_list = []
        static_types_list = []
        for i in range(n_atoms):
            if static_mask[i] and i not in problem_atoms:
                static_coords_list.append(new_coords[i])
                static_types_list.append(types[i])
        
        if static_coords_list:
            static_coords = np.array(static_coords_list)
            static_types = np.array(static_types_list, dtype=np.int32)
            emergency_relocate(problem_atoms, static_coords, static_types)
        else:
            # Fallback: just shake harder
            shake_phase(problem_atoms, shake_magnitude=1.0)
            push_phase(iterations=200, step_size=0.10)
        
        overlaps = count_overlaps()
        print(f"      Final overlaps: {len(overlaps)}")
    
    # Final check for O-O specifically
    oo_overlaps = [(i, j, r) for i, j, r, lim in count_overlaps() 
                   if types[i] == 4 and types[j] == 4]
    if len(oo_overlaps) > 0:
        print(f"    WARNING: {len(oo_overlaps)} O-O overlaps remain "
              f"(min distance: {min(r for _, _, r in oo_overlaps):.3f} A)")
        print("    These will be handled by MC relaxation in Stage 1.")
    else:
        print("    SUCCESS: Zero O-O overlaps in final structure!")
    
    return new_coords

# ============================================================================
# VERIFICATION
# ============================================================================
def verify_structure(symbols, coords, box, target_nc, expected_atoms):
    print("\n" + "=" * 60)
    print("  SOL-GEL STRUCTURE VERIFICATION (v5.0 Final)")
    print("=" * 60)
    tree = cKDTree(coords, boxsize=box)
    si_idx = [i for i, s in enumerate(symbols) if s == "Si"]
    p_idx = [i for i, s in enumerate(symbols) if s == "P"]
    o_idx = [i for i, s in enumerate(symbols) if s == "O"]

    print(f"  Total atoms: {len(symbols)} (Target: {expected_atoms})")
    print(f"  Si: {len(si_idx)}, P: {len(p_idx)}, O: {len(o_idx)}")

    o_nf_count = []
    for o in o_idx:
        nf_count = sum(
            1 for j in tree.query_ball_point(coords[o], 2.25)
            if j != o and symbols[j] in ["Si", "P"])
        o_nf_count.append(nf_count)

    fo = sum(1 for c in o_nf_count if c == 0)
    nbo = sum(1 for c in o_nf_count if c == 1)
    bo = sum(1 for c in o_nf_count if c == 2)
    n_o = max(1, len(o_idx))

    print(f"\n--- Oxygen Speciation ---")
    print(f"  FO (Free):  {fo:5d} ({fo / n_o * 100:5.2f}%)")
    print(f"  NBO:        {nbo:5d} ({nbo / n_o * 100:5.2f}%)")
    print(f"  BO:         {bo:5d} ({bo / n_o * 100:5.2f}%)")

    nc = (bo * 2) / max(1, len(si_idx) + len(p_idx))
    print(f"\n  Network Connectivity (NC): {nc:.3f} (Target: {target_nc:.3f})")

    total_mass = sum(MASSES[s] for s in symbols)
    density = total_mass / NA / (box ** 3) * 1e24
    print(f"  Density: {density:.4f} g/cm³")
    print("=" * 60)

    assert len(symbols) == expected_atoms, \
        f"Stoichiometry Error! Got {len(symbols)}, expected {expected_atoms}"
    return nc, density

# ============================================================================
# MAIN GENERATOR
# ============================================================================
def generate_solgel_final(output_xyz, output_json, seed=42, scale=1,
                          density=1.0, target_nc=0.8):
    rng = np.random.default_rng(seed)

    N_SI = BASE_N_SI * scale
    N_P = BASE_N_P * scale
    N_NA = BASE_N_NA * scale
    N_CA = BASE_N_CA * scale
    N_O = BASE_N_O * scale
    N_ATOMS = N_SI + N_P + N_NA + N_CA + N_O

    print(f"\n{'=' * 60}")
    print(f"  FINAL SOL-GEL 45S5 GENERATOR (v5.0)")
    print(f"{'=' * 60}")
    print(f"  Scale: {scale}x -> {N_ATOMS} atoms")
    print(f"  Target density: {density:.3f} g/cm³")
    print(f"  Target initial NC: {target_nc:.3f}")

    total_mass = (N_SI * MASSES["Si"] + N_P * MASSES["P"] +
                  N_NA * MASSES["Na"] + N_CA * MASSES["Ca"] +
                  N_O * MASSES["O"])
    box = ((total_mass / NA) / density * 1.0e24) ** (1.0 / 3.0)
    print(f"  Box size: {box:.4f} Å")

    n_nf = N_SI + N_P
    n_bo_target = int(round(target_nc * n_nf / 2.0))
    n_siop_max = min(int(0.05 * N_O), n_bo_target, 2 * N_P)

    print(f"\n[1/6] Placing Network Formers...")
    nf_coords, nf_types = place_network_formers(N_SI, N_P, box, rng)

    print(f"[2/6] Building Topology (Target BOs: {n_bo_target})...")
    edges, deg = build_topology_with_nc(
        nf_coords, nf_types, box, n_bo_target, n_siop_max, rng)

    print(f"[3/6] Placing Bridging Oxygens (BO) with Bending...")
    bo_coords = place_bridging_oxygens(nf_coords, nf_types, edges, box, rng)

    print(f"[4/6] Placing Terminal Oxygens (NBO) with Prioritization...")
    term_coords, term_owners = place_terminal_oxygens(
        nf_coords, nf_types, deg, box, bo_coords, rng, N_O)

    o_coords = (np.vstack([term_coords, bo_coords])
                if len(bo_coords) > 0 else term_coords)

    print(f"[5/6] Placing Modifiers (Na, Ca)...")
    mod_coords = place_modifiers(N_NA, N_CA, o_coords, box, rng)

    # Combine all coordinates and types for relaxation
    all_coords = np.vstack([nf_coords, o_coords, mod_coords])
    all_types = np.concatenate([
        nf_types,
        np.full(len(o_coords), 4, dtype=np.int32),  # O
        np.full(N_NA, 2, dtype=np.int32),           # Na
        np.full(N_CA, 1, dtype=np.int32),           # Ca
    ])

    print(f"[6/6] Safe Relaxation (fixing O-O overlaps)...")
    fixed_mask = np.zeros(len(all_coords), dtype=bool)
    # Network formers (Si, P) are fixed
    fixed_mask[:len(nf_coords)] = True
    # Bridging oxygens are fixed (they define topology)
    n_term = len(term_coords)
    n_bo = len(bo_coords)
    fixed_mask[len(nf_coords) + n_term : len(nf_coords) + n_term + n_bo] = True
    # Terminal oxygens, and modifiers are free to move
    
    # === NEW: Advanced multi-phase relaxation ===
    relaxed_coords = relax_o_overlaps_safe(
        all_coords, all_types, box, fixed_mask,
        r_hard_oo=2.00,        # Realistic O-O hard core (was 2.20)
        max_overlap_pairs=5     # Allow up to 5 minor overlaps (handled by MC)
    )

    # Assemble symbols and apply permutation
    symbols = (["Si"] * N_SI + ["P"] * N_P + ["O"] * len(o_coords) +
               ["Na"] * N_NA + ["Ca"] * N_CA)

    perm = rng.permutation(len(symbols))
    symbols = [symbols[i] for i in perm]
    final_coords = relaxed_coords[perm]
    inv_perm = np.argsort(perm)

    # Build bond list mapping to the FINAL permuted indices
    initial_bonds = []
    for k, (nf_i, nf_j) in enumerate(edges):
        # The k-th BO corresponds to the k-th edge
        bo_pre_idx = len(nf_coords) + n_term + k
        bo_final_idx = int(inv_perm[bo_pre_idx])
        nf_i_final = int(inv_perm[nf_i])
        nf_j_final = int(inv_perm[nf_j])
        initial_bonds.append([nf_i_final, bo_final_idx])
        initial_bonds.append([nf_j_final, bo_final_idx])

    print(f"\n  Generated {len(initial_bonds)} (nf, o) bond pairs")

    nc, actual_density = verify_structure(
        symbols, final_coords, box, target_nc, N_ATOMS)

    # Save XYZ
    with open(output_xyz, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"Sol-gel 45S5 v5.0, N={len(symbols)}, "
                f"rho={actual_density:.4f}, box={box:.4f}, "
                f"NC={nc:.3f}, seed={seed}\n")
        for i in range(len(symbols)):
            f.write(f"{symbols[i]:2s} {final_coords[i, 0]:12.6f} "
                    f"{final_coords[i, 1]:12.6f} "
                    f"{final_coords[i, 2]:12.6f}\n")
    print(f"\n[OK] XYZ saved to: {output_xyz}")

    # Save JSON
    topology = {
        "n_atoms": len(symbols),
        "box_size": float(box),
        "density": float(actual_density),
        "network_connectivity": float(nc),
        "seed": seed,
        "target_nc": target_nc,
        "n_si": N_SI, "n_p": N_P, "n_na": N_NA,
        "n_ca": N_CA, "n_o": N_O,
        "initial_bonds": initial_bonds,
        "n_bonds": len(initial_bonds),
    }
    with open(output_json, "w") as f:
        json.dump(topology, f, indent=2)
    print(f"[OK] Topology saved to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final Sol-Gel 45S5 Generator v5.0")
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--target-nc", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=1,
                        choices=[1, 2, 4])
    args = parser.parse_args()

    # Ensure data directory exists
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    out_xyz = data_dir / f"solgel_v5_rho{args.density:.1f}_nc{args.target_nc:.1f}_seed{args.seed}.xyz"
    out_json = data_dir / f"solgel_v5_rho{args.density:.1f}_nc{args.target_nc:.1f}_seed{args.seed}.json"

    generate_solgel_final(out_xyz, out_json, args.seed, args.scale,
                          args.density, args.target_nc)