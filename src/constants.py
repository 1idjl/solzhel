"""
src/constants.py — Physical constants, atom types, and 45S5 composition
========================================================================
Central location for all constants used across the project.
All values are in eV, Angstrom, and atomic units unless noted.

References:
- Tilocca, J. Mater. Chem. 20, 6848 (2010) — bioglass review
- Mead & Mountjoy, Chem. Mater. 18, 3956 (2006) — sol-gel calcium silicates
"""

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
KB_EV = 8.617333262145e-5        # Boltzmann constant [eV/K]
NA_AVOGADRO = 6.02214076e23      # Avogadro's number [1/mol]
KE_COULOMB = 14.3996454784255    # Coulomb constant [eV·Å/e²]
ATM_TO_EV_A3 = 6.324e-7         # Pressure conversion: 1 atm → eV/Å³
AMU_TO_GRAM = 1.66054e-24       # Atomic mass unit → grams

# ============================================================================
# ATOM TYPE INDICES
# Must remain consistent with generator, simulator, and analysis codes.
# ============================================================================
TYPE_SI = 0
TYPE_CA = 1
TYPE_NA = 2
TYPE_P  = 3
TYPE_O  = 4
TYPE_SR = 5  # Reserved for future Sr-doped studies

N_TYPES = 6  # Total number of atom types

TYPE_TO_ELEM = {
    TYPE_SI: 'Si',
    TYPE_CA: 'Ca',
    TYPE_NA: 'Na',
    TYPE_P:  'P',
    TYPE_O:  'O',
    TYPE_SR: 'Sr',
}

ELEM_TO_TYPE = {v: k for k, v in TYPE_TO_ELEM.items()}

# ============================================================================
# MAXIMUM COORDINATION NUMBERS
# Used by BondNetwork to prevent over-coordination in reactive MC.
# ============================================================================
MAX_COORD = {
    TYPE_SI: 4,   # Si: tetrahedral (SiO4)
    TYPE_P:  4,   # P:  tetrahedral (PO4)
    TYPE_O:  2,   # O:  bridging (BO) or non-bridging (NBO)
    TYPE_CA: 8,   # Ca: modifier (ionic coordination)
    TYPE_NA: 8,   # Na: modifier (ionic coordination)
    TYPE_SR: 8,   # Sr: modifier (ionic coordination)
}

# ============================================================================
# ATOMIC MASSES [g/mol]
# ============================================================================
MASSES = {
    'Si': 28.0855,
    'Ca': 40.078,
    'Na': 22.98977,
    'P':  30.97376,
    'O':  15.999,
    'Sr': 87.62,
}

# ============================================================================
# 45S5 BIOACTIVE GLASS COMPOSITION (base unit, scale=1 → 2835 atoms)
# 45SiO2 - 24.5Na2O - 24.5CaO - 6P2O5 (wt%)
# ============================================================================
BASE_N_SI = 461
BASE_N_P  = 52
BASE_N_NA = 488
BASE_N_CA = 269
BASE_N_O  = 1565
BASE_N_ATOMS = BASE_N_SI + BASE_N_P + BASE_N_NA + BASE_N_CA + BASE_N_O  # 2835

# ============================================================================
# EQUILIBRIUM BOND LENGTHS [Å]
# From Tilocca (2010) and Pedone et al. (2008) potentials.
# ============================================================================
SI_O_BOND = 1.61   # Si-O tetrahedral
P_O_BOND  = 1.49   # P-O tetrahedral
NA_O_BOND = 2.37   # Na-O ionic
CA_O_BOND = 2.40   # Ca-O ionic
SR_O_BOND = 2.60   # Sr-O ionic (reserved)

# ============================================================================
# NETWORK FORMER PLACEMENT DISTANCES [Å]
# Minimum distances for initial placement of Si and P atoms.
# ============================================================================
MIN_SI_SI = 3.05
MIN_SI_P  = 2.95
MIN_P_P   = 3.40

# ============================================================================
# TOPOLOGY SEARCH RANGES [Å]
# Distance windows for finding bridging oxygen candidates.
# ============================================================================
SI_SI_MIN, SI_SI_MAX = 3.00, 4.40
SI_P_MIN,  SI_P_MAX  = 2.90, 4.10
P_P_MIN,   P_P_MAX   = 2.85, 3.80

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
CUTOFF_DEFAULT = 8.0       # Non-bonded interaction cutoff [Å]
SKIN_DEFAULT = 1.5         # Neighbor list skin [Å]
SWITCH_DR = 0.3            # ZBL switching function width [Å]
WOLF_ALPHA_DEFAULT = 0.20  # Wolf damping parameter [1/Å]

# ============================================================================
# REACTIVE MC PARAMETERS
# ============================================================================
BOND_FORM_CUTOFF = 3.0     # Max distance for bond formation [Å]
BOND_HARMONIC_K_SI_O = 25.0  # Harmonic spring constant Si-O [eV/Å²]
BOND_HARMONIC_K_P_O  = 30.0  # Harmonic spring constant P-O  [eV/Å²]
BOND_HARMONIC_R0_SI_O = SI_O_BOND  # Equilibrium distance Si-O [Å]
BOND_HARMONIC_R0_P_O  = P_O_BOND   # Equilibrium distance P-O  [Å]

# ============================================================================
# SOL-GEL PROTOCOL DEFAULTS
# ============================================================================
SOLGEL_TARGET_DENSITY = 1.0     # Initial sol density [g/cm³]
SOLGEL_TARGET_NC = 0.8          # Initial network connectivity
GLASS_FINAL_DENSITY = 2.674     # Target final density for 45S5 [g/cm³]
MAX_DENSITY_LIMIT = 2.80        # Hard upper limit for density [g/cm³]

# ============================================================================
# SAFETY DISTANCES [Å]
# ============================================================================
ANALYSIS_CUTOFF = 2.25     # Cutoff for coordination analysis
NBO_SAFE_DIST = 2.50       # Min distance between NBO and other NFs
NBO_O_OVERLAP = 1.95       # Min O-O distance to prevent overlap