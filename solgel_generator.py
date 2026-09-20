#!/usr/bin/env python3
"""
solgel_generator_v2.py — Corrected Sol-Gel 45S5 Generator (Strategy A)
=======================================================================
Generates a low-density structure with NC ≈ 1.9 (matching melt-quench)
but with large voids and incomplete inter-cluster bonding.
Suitable for Reactive MC where bonds will form/break dynamically.
"""

import numpy as np
import argparse
import json
from pathlib import Path
from scipy.spatial import cKDTree
from math import sqrt, sin, cos, pi

NA = 6.02214076e23

# Exact 45S5 composition for scale=1
BASE_N_SI = 461
BASE_N_P  = 52
BASE_N_NA = 488
BASE_N_CA = 269
BASE_N_O  = 1565

MASSES = {"Si": 28.0855, "P": 30.97376, "Na": 22.98977, "Ca": 40.078, "O": 15.999}

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

def minimum_image(dr, box):
    return dr - box * np.round(dr / box)

def random_rotation(rng):
    u1, u2, u3 = rng.random(3)
    q0 = sqrt(1.0 - u1) * sin(2.0 * pi * u2)
    q1 = sqrt(1.0 - u1) * cos(2.0 * pi * u2)
    q2 = sqrt(u1) * sin(2.0 * pi * u3)
    q3 = sqrt(u1) * cos(2.0 * pi * u3)
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    R[0, 1] = 2.0 * (q1 * q2 - q0 * q3)
    R[0, 2] = 2.0 * (q1 * q3 + q0 * q2)
    R[1, 0] = 2.0 * (q1 * q2 + q0 * q3)
    R[1, 1] = 1.0 - 2.0 * (q1 * q1 + q3 * q3)
    R[1, 2] = 2.0 * (q2 * q3 - q0 * q1)
    R[2, 0] = 2.0 * (q1 * q3 - q0 * q2)
    R[2, 1] = 2.0 * (q2 * q3 + q0 * q1)
    R[2, 2] = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    return R

def place_network_formers(n_si, n_p, box, rng):
    n_total = n_si + n_p
    coords = np.zeros((n_total, 3), dtype=np.float64)
    types = np.zeros(n_total, dtype=np.int32)
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
                elif types[k] == 3 and not is_p:
                    if r[k] < MIN_SI_P: valid = False; break
                elif types[k] == 0 and is_p:
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

def build_topology_with_nc(nf_coords, nf_types, box, n_bo_target, n_siop_max, rng):
    """Build topology targeting specific NC (Network Connectivity)."""
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
                
    si_si.sort()
    si_p.sort()
    p_p.sort()
    
    deg = np.zeros(n_nf, dtype=np.int32)
    edges = []
    selected = set()
    
    # 1. Si-O-P (limited to avoid P clustering)
    n_siop = 0
    for s, i, j in si_p:
        if n_siop >= n_siop_max: break
        if deg[i] < 4 and deg[j] < 4:
            key = (min(i, j), max(i, j))
            if key not in selected:
                selected.add(key)
                deg[i] += 1; deg[j] += 1
                edges.append((i, j))
                n_siop += 1
                
    # 2. Si-O-Si
    n_siosi = 0
    n_siosi_target = n_bo_target - n_siop
    for s, i, j in si_si:
        if n_siosi >= n_siosi_target: break
        if deg[i] < 4 and deg[j] < 4:
            key = (min(i, j), max(i, j))
            if key not in selected:
                selected.add(key)
                deg[i] += 1; deg[j] += 1
                edges.append((i, j))
                n_siosi += 1
                
    print(f"    Topology built: {len(edges)} BOs (Target was {n_bo_target})")
    return edges, deg

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
        bo_coords.append(o_pos % box)
    return np.array(bo_coords, dtype=np.float64) if bo_coords else np.empty((0, 3), dtype=np.float64)

def place_terminal_oxygens(nf_coords, nf_types, deg, box, bo_coords, rng, n_o_target):
    """Place NBOs, but respect total oxygen budget."""
    term_coords, term_owners = [], []
    n_nf = len(nf_coords)
    nf_tree = cKDTree(nf_coords, boxsize=box)
    all_o = list(bo_coords) if len(bo_coords) > 0 else []
    o_tree = cKDTree(np.array(all_o), boxsize=box) if all_o else None
    
    TETRA = np.array([[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]], dtype=np.float64)
    TETRA /= np.linalg.norm(TETRA, axis=1)[:, None]
    
    # Calculate how many NBOs we can afford
    n_bo = len(bo_coords)
    n_nbo_budget = n_o_target - n_bo
    
    # Calculate how many NBOs we need
    n_nbo_needed_total = sum(max(0, 4 - int(deg[nf])) for nf in range(n_nf))
    
    # If we can't afford all NBOs, prioritize randomly
    print(f"    NBO budget: {n_nbo_budget}, NBO needed: {n_nbo_needed_total}")
    
    # Build list of (nf, nbo_needed)
    nbo_tasks = []
    for nf in range(n_nf):
        n_needed = max(0, 4 - int(deg[nf]))
        for _ in range(n_needed):
            nbo_tasks.append(nf)
    
    # Shuffle to randomize which NBOs get placed
    rng.shuffle(nbo_tasks)
    
    # Only place up to budget
    nbo_tasks = nbo_tasks[:n_nbo_budget]
    
    # Group by NF for efficient placement
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
                if len(all_o) % 100 == 0 and len(all_o) > 0:
                    o_tree = cKDTree(np.array(all_o), boxsize=box)
                    
    return np.array(term_coords, dtype=np.float64), np.array(term_owners, dtype=np.int32)

def place_modifiers(n_na, n_ca, sites, box, rng):
    if len(sites) == 0:
        return rng.uniform(0.0, box, size=(n_na + n_ca, 3))
        
    static_tree = cKDTree(sites, boxsize=box)
    mod_coords = []
    
    for elem, n_mod, target_dist in [("Na", n_na, NA_O_BOND), ("Ca", n_ca, CA_O_BOND)]:
        min_mod_dist = 2.0
        for _ in range(n_mod):
            pos = None
            for _ in range(3000):
                site = sites[rng.integers(len(sites))]
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                dist = target_dist + rng.uniform(-0.15, 0.25)
                cand = (site + direction * dist) % box
                if len(static_tree.query_ball_point(cand, 1.4)) > 0: continue
                
                ok = True
                if mod_coords:
                    for m in mod_coords:
                        if np.linalg.norm(minimum_image(cand - m, box)) < min_mod_dist:
                            ok = False; break
                if ok:
                    pos = cand; break
            if pos is None: pos = rng.uniform(0.0, box, 3)
            mod_coords.append(pos)
            
    return np.array(mod_coords, dtype=np.float64)

def verify_structure(symbols, coords, box, target_nc):
    print("\n" + "=" * 60)
    print("  SOL-GEL STRUCTURE VERIFICATION (v2 - Budget-Aware)")
    print("=" * 60)
    tree = cKDTree(coords, boxsize=box)
    si_idx = [i for i, s in enumerate(symbols) if s == "Si"]
    p_idx  = [i for i, s in enumerate(symbols) if s == "P"]
    o_idx  = [i for i, s in enumerate(symbols) if s == "O"]
    
    print(f"  Total atoms: {len(symbols)} (Target: 2835)")
    print(f"  Si: {len(si_idx)}, P: {len(p_idx)}, O: {len(o_idx)}")
    
    o_nf_count = []
    for o in o_idx:
        nf_count = sum(1 for j in tree.query_ball_point(coords[o], 2.25)
                       if j != o and symbols[j] in ["Si", "P"])
        o_nf_count.append(nf_count)
        
    nbo = sum(1 for c in o_nf_count if c == 1)
    bo  = sum(1 for c in o_nf_count if c == 2)
    n_o = max(1, len(o_idx))
    
    print(f"\n--- Oxygen Speciation ---")
    print(f"  NBO: {nbo:5d} ({nbo/n_o*100:5.2f}%)")
    print(f"  BO:  {bo:5d} ({bo/n_o*100:5.2f}%)")
    
    nc = (bo * 2) / max(1, len(si_idx) + len(p_idx))
    print(f"\n  Network Connectivity (NC): {nc:.3f} (Target: {target_nc:.3f})")
    
    total_mass = sum(MASSES[s] for s in symbols)
    density = total_mass / NA / (box**3) * 1e24
    print(f"  Density: {density:.4f} g/cm³")
    print("=" * 60)
    return nc, density

def generate_solgel_v2(output_xyz, output_json, seed=42, scale=1, density=1.0, target_nc=1.9):
    rng = np.random.default_rng(seed)
    
    N_SI = BASE_N_SI * scale
    N_P  = BASE_N_P  * scale
    N_NA = BASE_N_NA * scale
    N_CA = BASE_N_CA * scale
    N_O  = BASE_N_O  * scale
    N_ATOMS = N_SI + N_P + N_NA + N_CA + N_O
    
    print(f"\n{'='*60}")
    print(f"  SOL-GEL 45S5 GENERATOR v2 (Budget-Aware)")
    print(f"{'='*60}")
    print(f"  Scale: {scale}x → {N_ATOMS} atoms (Exact 45S5)")
    print(f"  Target density: {density:.3f} g/cm³")
    print(f"  Target NC: {target_nc:.3f}")
    
    total_mass = (N_SI*MASSES["Si"] + N_P*MASSES["P"] + N_NA*MASSES["Na"] + 
                  N_CA*MASSES["Ca"] + N_O*MASSES["O"])
    box = ((total_mass / NA) / density * 1.0e24) ** (1.0 / 3.0)
    print(f"  Box size: {box:.4f} Å")
    
    n_nf = N_SI + N_P
    n_bo_target = int(round(target_nc * n_nf / 2.0))
    n_siop_max = min(int(0.02 * N_O), n_bo_target, 2 * N_P)
    
    print(f"\n[1/5] Placing Network Formers...")
    nf_coords, nf_types = place_network_formers(N_SI, N_P, box, rng)
    
    print(f"[2/5] Building Topology (Target BOs: {n_bo_target})...")
    edges, deg = build_topology_with_nc(nf_coords, nf_types, box, n_bo_target, n_siop_max, rng)
    
    print(f"[3/5] Placing Bridging Oxygens (BO)...")
    bo_coords = place_bridging_oxygens(nf_coords, nf_types, edges, box, rng)
    
    print(f"[4/5] Placing Terminal Oxygens (NBO)...")
    term_coords, term_owners = place_terminal_oxygens(
        nf_coords, nf_types, deg, box, bo_coords, rng, N_O
    )
    
    o_coords = np.vstack([term_coords, bo_coords]) if len(bo_coords) > 0 else term_coords
    
    print(f"[5/5] Placing Modifiers (Na, Ca)...")
    mod_coords = place_modifiers(N_NA, N_CA, o_coords, box, rng)
    
    # Assemble
    all_coords = np.vstack([nf_coords, o_coords, mod_coords])
    symbols = (["Si"]*N_SI + ["P"]*N_P + ["O"]*len(o_coords) + 
               ["Na"]*N_NA + ["Ca"]*N_CA)
               
    perm = rng.permutation(len(symbols))
    symbols = [symbols[i] for i in perm]
    final_coords = all_coords[perm]
    
    # Remap bonds
    inv_perm = np.argsort(perm)
    final_edges = [(int(inv_perm[i]), int(inv_perm[j])) for i, j in edges]
    
    nc, actual_density = verify_structure(symbols, final_coords, box, target_nc)
    
    # Save XYZ
    with open(output_xyz, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"Sol-gel 45S5 (v2), N={len(symbols)}, rho={actual_density:.4f}, "
                f"box={box:.4f}, NC={nc:.3f}, seed={seed}\n")
        for i in range(len(symbols)):
            f.write(f"{symbols[i]:2s} {final_coords[i,0]:12.6f} "
                    f"{final_coords[i,1]:12.6f} {final_coords[i,2]:12.6f}\n")
    print(f"\n[OK] XYZ saved to: {output_xyz}")
    
    # Save JSON
    topology = {
        "n_atoms": len(symbols), "box_size": float(box), "density": float(actual_density),
        "network_connectivity": float(nc), "seed": seed, "target_nc": target_nc,
        "n_si": N_SI, "n_p": N_P, "n_na": N_NA, "n_ca": N_CA, "n_o": N_O,
        "initial_bonds": final_edges,
        "n_bonds": len(final_edges),
    }
    with open(output_json, "w") as f:
        json.dump(topology, f, indent=2)
    print(f"[OK] Topology saved to: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sol-Gel 45S5 Generator v2")
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--target-nc", type=float, default=1.9,
                       help="Target NC (max 1.90 without H atoms)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=1, choices=[1, 2, 4])
    args = parser.parse_args()
    
    out_xyz = f"solgel_v2_rho{args.density:.1f}_nc{args.target_nc:.1f}_seed{args.seed}.xyz"
    out_json = f"solgel_v2_rho{args.density:.1f}_nc{args.target_nc:.1f}_seed{args.seed}.json"
    
    generate_solgel_v2(out_xyz, out_json, args.seed, args.scale, args.density, args.target_nc)