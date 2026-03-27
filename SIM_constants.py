# constants.py   DON'T USE PAPER MODE, USE NEURONLIKE
import math
from scipy import constants as cst
#  fundamental constants 
c0   = 299_792_458.0                 # m/s, speed of causality
eps0 = cst.epsilon_0           # F/m
hbar = 1.054_571_817e-34             # J·s reduced h

#  Molecular line (vacuum) 
lam0  = 280e-9                       # central wavelength in vacuum of tryptophan, as per papaer [m]
mu_D  = 6.0                          # transition dipole magnitude [Debye] (paper uses ~6 D)
mu    = mu_D * 3.33564e-30           # C·m
omega0 = 2.0 * math.pi * c0 / lam0   # angular frequency in vacuum [s^-1]

#  Environment (neuron-ish if paper is silent).  DO NOT USE PAPER MODE
n_med = 1.36                         # neuronal cytosol/axon ~1.35–1.38
use_local_field = True              # toggle Lorentz–Lorenz only in neuron mode

# --- Mode switch ---
paper_mode = False                   # True => reproduce paper’s convention

# distance r cutoff

r_cut = 300e-9
def gamma0_from_paper():
    """
    Paper reports a single-site linewidth ~2.73e-3 cm^-1 (gives τ ≈ 1.9 ns).
    Convert wavenumber linewidth [cm^-1] -> angular rate [s^-1] via 2π c.
    """
    gamma0_cm1 = 2.73e-3             
    return gamma0_cm1 * 2.0 * math.pi * (c0 * 100.0)

# k and gamma0 according to mode 
if paper_mode:
    k = 2.0 * math.pi / lam0                     # vacuum wave number k0 [m^-1]
    gamma0 = gamma0_from_paper()                 # single-site rate [s^-1]
else:
    k = (n_med * omega0) / c0                    # medium wave number [m^-1]
    L = ((n_med**2 + 2.0) / 3.0) if use_local_field else 1.0
    gamma0 = (n_med * omega0**3 * (mu**2) * (L**2)) / (3.0 * math.pi * eps0 * hbar * c0**3)

# converters (so the terminal produces intellegible output)
J_to_eV = 1.0 / 1.602_176_634e-19
s_to_ps = 1e-12

# --- Quantum yield and gamma ---
#QY_single = 0.12  # tryptophan in water-like environment
#gamma_nr = C.gamma0 * (1.0 - QY_single) / QY_single  # ≈ 7.33 * gamma0
#THESE ONES WE PASS IN ARGS WHEN CLI