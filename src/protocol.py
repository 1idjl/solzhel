"""
src/protocol.py — 5-stage Sol-Gel Protocol (v5.1)
===================================================
[FIX v5.1]: Corrected debug_freq to be offset from recalc_freq
so that drift is measured BETWEEN recalculations, not at the
same time (which always gave 0).
"""

import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_solgel_protocol(sim, output_dir: Path, gelation_sweeps=50,
                         aging_sweeps=30, densification_sweeps=20,
                         cooling_sweeps=10, production_sweeps=100):
    """Execute the 5-stage sol-gel reactive MC protocol."""
    N = sim.system.N_ATOMS
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("SOL-GEL REACTIVE MC PROTOCOL (v5.1 Final)")
    logger.info("=" * 70)
    
    bf_freq = max(1, N // 50)
    bb_freq = max(1, N // 100)
    bs_freq = max(1, N // 20)
    vol_freq = max(1, N // 50)
    
    logger.info(f"Move frequencies: form={bf_freq}, break={bb_freq}, "
                f"switch={bs_freq}, vol={vol_freq}")

    def run_stage(T, steps, desc, bf_f=0, bb_f=0, bs_f=0, v_f=0,
                  adaptive=True):
        """Run a single stage of the protocol."""
        logger.info(f"{desc}: T={T:.0f} K, steps={steps}")
        stage_acc = stage_att = window_acc = window_att = 0
        ema = 0.5
        recalc_freq = max(2000, N)
        
        # [FIX v5.1] Offset debug_freq so drift is measured between recalcs
        debug_freq = recalc_freq + 100
        
        pbar = tqdm(range(steps), desc=f"T={T:.0f}K")
        for step_in_stage in pbar:
            sim.step += 1
            sim.attempts += 1
            stage_att += 1
            window_att += 1
            
            if bf_f > 0 and step_in_stage % bf_f == 0:
                sim.bond_formation_move(T)
            if bb_f > 0 and step_in_stage % bb_f == 0:
                sim.bond_breaking_move(T)
            if bs_f > 0 and step_in_stage % bs_f == 0:
                sim.bond_switch_move(T)
            if v_f > 0 and step_in_stage % v_f == 0:
                sim.volume_move(T)
            
            if sim.mc_move(T):
                sim.accepted += 1
                stage_acc += 1
                window_acc += 1
            
            if step_in_stage % recalc_freq == 0:
                sim.recalculate_energy()
            
            if sim.debug_energy and step_in_stage % debug_freq == 0:
                drift = sim.check_energy_drift()
                if abs(drift) > 0.005:
                    logger.info(f"[DEBUG] Step {sim.step}: "
                                f"Energy drift = {drift:+.6f} eV/atom")
            
            sim._maybe_log()
            
            # Adaptive displacement tuning
            if adaptive and stage_att % max(500, N // 10) == 0:
                ca = window_acc / max(1, window_att)
                ema = 0.8 * ema + 0.2 * ca
                if ema < 0.35:
                    for e in sim.max_disp:
                        sim.max_disp[e] = max(sim.max_disp[e] * 0.92, 0.02)
                elif ema > 0.45:
                    for e in sim.max_disp:
                        sim.max_disp[e] = min(sim.max_disp[e] * 1.08, 0.25)
                window_acc = window_att = 0
            
            nc = sim.nc_log[-1] if sim.nc_log else 0.0
            acc_pct = sim.accepted / max(1, sim.attempts) * 100
            rho = (sim.system.total_mass * 1.0e24 /
                   (6.022e23 * sim.system.box ** 3))
            pbar.set_postfix({
                'acc': f'{acc_pct:.1f}%',
                'NC': f'{nc:.2f}',
                'bonds': sim.bond_network.n_active_bonds,
                'E': f'{sim.current_energy/N:.3f}',
                'rho': f'{rho:.3f}'
            })
        pbar.close()

    # ========================================================================
    # STAGE 1: GELATION
    # ========================================================================
    logger.info("\n--- STAGE 1: GELATION ---")
    run_stage(2000.0, gelation_sweeps * N, "Gelation",
              bf_freq, bb_freq, bs_freq)
    sim.recalculate_energy()
    sim.save_structure(output_dir / "stage1_gelation.xyz",
                       "Stage 1: Gelation")

    # ========================================================================
    # STAGE 2: AGING
    # ========================================================================
    logger.info("\n--- STAGE 2: AGING ---")
    run_stage(1000.0, aging_sweeps * N, "Aging",
              bf_freq, bb_freq * 2, bf_freq)
    sim.recalculate_energy()
    sim.save_structure(output_dir / "stage2_aging.xyz",
                       "Stage 2: Aging")

    # ========================================================================
    # STAGE 3: DENSIFICATION (NPT)
    # ========================================================================
    if sim.npt_enabled:
        logger.info("\n--- STAGE 3: DENSIFICATION (NPT) ---")
        run_stage(600.0, densification_sweeps * N, "Densification",
                  bf_freq * 2, bb_freq * 3, bs_freq, vol_freq)
        sim.recalculate_energy()
        sim.save_structure(output_dir / "stage3_densification.xyz",
                           "Stage 3: Densification")

    # ========================================================================
    # STAGE 4: COOLING
    # ========================================================================
    logger.info("\n--- STAGE 4: COOLING ---")
    for T in [800.0, 600.0, 400.0, 300.0]:
        run_stage(T, max(1, cooling_sweeps // 4) * N,
                  f"Cooling to {T:.0f}K",
                  bf_freq * 2, bb_freq * 2, bs_freq)
    sim.recalculate_energy()

    # ========================================================================
    # STAGE 5: PRODUCTION
    # ========================================================================
    logger.info("\n--- STAGE 5: PRODUCTION ---")
    run_stage(300.0, production_sweeps * N, "Production",
              bf_freq * 3, bb_freq * 3, bs_freq * 2)
    sim.recalculate_energy()

    sim.save_structure(output_dir / "final_structure.xyz",
                       "Final Sol-Gel 45S5")
    
    # Save logs
    energy_path = output_dir / "energy_log.csv"
    with open(energy_path, 'w') as f:
        f.write("Step,Energy_eV_per_atom,NC\n")
        for i, (e, nc) in enumerate(zip(sim.energy_log, sim.nc_log)):
            step = (i + 1) * sim.log_freq
            f.write(f"{step},{e:.6f},{nc:.4f}\n")
    
    if sim.debug_energy and sim.drift_log:
        drift_path = output_dir / "drift_log.csv"
        with open(drift_path, 'w') as f:
            f.write("Drift_eV_per_atom\n")
            for d in sim.drift_log:
                f.write(f"{d:+.6f}\n")
        logger.info(f"[DEBUG] Drift stats: "
                    f"mean={np.mean(sim.drift_log):+.6f}, "
                    f"max_abs={np.max(np.abs(sim.drift_log)):.6f} eV/atom")

    # Print summary
    rho = (sim.system.total_mass * 1.0e24 /
           (6.022e23 * sim.system.box ** 3))
    logger.info("\n" + "=" * 70)
    logger.info("SIMULATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Final step: {sim.step} | "
                f"Final energy: {sim.current_energy/N:.6f} eV/atom")
    logger.info(f"Final density: {rho:.4f} g/cm3 | "
                f"Final box: {sim.system.box:.4f} A")
    logger.info(f"Final NC: {sim.compute_nc():.3f} | "
                f"Final bonds: {sim.bond_network.n_active_bonds}")
    logger.info(f"MC acceptance: "
                f"{sim.accepted/max(1,sim.attempts)*100:.1f}%")
    logger.info(f"Bond form: {sim.bond_form_accepted}/"
                f"{sim.bond_form_attempts} | "
                f"break: {sim.bond_break_accepted}/"
                f"{sim.bond_break_attempts} | "
                f"switch: {sim.bond_switch_accepted}/"
                f"{sim.bond_switch_attempts}")
    if sim.npt_enabled:
        logger.info(f"Volume moves: "
                    f"{sim.vol_accepted}/{sim.vol_attempts}")
    logger.info("=" * 70)