#!/usr/bin/env python3
"""
scripts/run_simulation.py — Main entry point for Sol-Gel Reactive MC
"""
import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.glass_system import GlassSystem
from src.simulator import SolGelSimulator
from src.protocol import run_solgel_protocol

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Reactive MC for Sol-Gel 45S5 Bioactive Glass v5.0")
    parser.add_argument('input_xyz', type=Path, help="Input XYZ file")
    parser.add_argument('--topology', type=Path, default=None, help="Topology JSON file")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--density', type=float, default=None)
    parser.add_argument('--npt', action='store_true')
    parser.add_argument('--pressure-atm', type=float, default=1.0)
    parser.add_argument('--disp-volume', type=float, default=0.02)
    parser.add_argument('--debug-energy', action='store_true')
    parser.add_argument('--gelation-sweeps', type=int, default=50)
    parser.add_argument('--aging-sweeps', type=int, default=30)
    parser.add_argument('--densification-sweeps', type=int, default=20)
    parser.add_argument('--cooling-sweeps', type=int, default=10)
    parser.add_argument('--production-sweeps', type=int, default=100)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--continue', dest='continue_mode', action='store_true')
    args = parser.parse_args()

    if args.output_dir is None:
        ens = "NPT" if args.npt else "NVT"
        args.output_dir = PROJECT_ROOT / "results" / f"solgel_mc_{ens}_seed{args.seed}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim = None
    try:
        system = GlassSystem(args.input_xyz, args.topology, density=args.density)
        sim = SolGelSimulator(system, seed=args.seed, npt_enabled=args.npt, pressure_atm=args.pressure_atm,
                              disp_volume=args.disp_volume, debug_energy=args.debug_energy)
        if args.continue_mode: sim.load_checkpoint(args.output_dir / "checkpoint.json")
        
        run_solgel_protocol(sim, args.output_dir, args.gelation_sweeps, args.aging_sweeps,
                            args.densification_sweeps, args.cooling_sweeps, args.production_sweeps)
        sim.save_checkpoint(args.output_dir / "checkpoint.json")
        logger.info(f"\n[OK] All results saved to {args.output_dir}")
    except KeyboardInterrupt:
        logger.warning("\n[!] Interrupted. Saving emergency checkpoint...")
        if sim: sim.save_checkpoint(args.output_dir / "checkpoint.json")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        if sim: sim.save_checkpoint(args.output_dir / "checkpoint.json")
        sys.exit(1)

if __name__ == "__main__":
    main()