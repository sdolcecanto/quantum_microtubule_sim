# physics_hook_fast.py
from __future__ import annotations
import numpy as np
import math
from typing import Tuple, Dict, Any
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigs

#  optional numba kernel, set NUMBA=1 to use. on my machine i didn't manage to make it work tbh
NUMBA = 0
if NUMBA:
    from numba import njit

def _omega_gamma_pairs(mu_i, mu_j, rvec, k, gamma0):
    """Vectorized Lehmberg for arrays of pairs.
       mu_i, mu_j: (M,3); rvec: (M,3); returns (Omega, Gamma) shape (M,)"""
    r = np.linalg.norm(rvec, axis=1)
    x = k * r
    rhat = rvec / np.clip(r[:, None], 1e-30, None)

    ui_uj = np.einsum('ij,ij->i', mu_i, mu_j)
    ui_r  = np.einsum('ij,ij->i', mu_i, rhat)
    uj_r  = np.einsum('ij,ij->i', mu_j, rhat)

    A = ui_uj - ui_r * uj_r
    B = ui_uj - 3.0 * ui_r * uj_r

    # safe sin/cos
    small = (x < 1e-6)
    sinx  = np.empty_like(x)
    cosx  = np.empty_like(x)
    sinx[small] = x[small] - (x[small]**3)/6.0
    cosx[small] = 1.0 - 0.5*(x[small]**2)
    sinx[~small] = np.sin(x[~small])
    cosx[~small] = np.cos(x[~small])

    invx  = np.where(x!=0, 1.0/x, 0.0)
    invx2 = invx*invx
    invx3 = invx2*invx

    # Γ_ij
    Gamma = (1.5*gamma0) * ( A*(sinx*invx) + B*(cosx*invx2 - sinx*invx3) )
    # Ω_ij
    Omega = -(0.75*gamma0) * ( A*(cosx*invx) - B*(sinx*invx2 + cosx*invx3) )
    return Omega, Gamma

if NUMBA:
    _omega_gamma_pairs = njit(_omega_gamma_pairs)  # type: ignore

def build_sparse_mats(R: np.ndarray, MU: np.ndarray, k: float, gamma0: float,
                      r_cut: float) -> Tuple[csr_matrix, csr_matrix]:
    """
    R: (N,3), MU: (N,3) unit dipoles
    returns Ω, Υ as CSR (with Υ_ii = γ0)
    """
    N = R.shape[0]
    tree = cKDTree(R)
    pairs = tree.query_pairs(r_cut, output_type='ndarray')   # (M,2)
    if pairs.size == 0:
        Omega = diags(np.zeros(N), format='csr')
        Upsilon = diags(np.full(N, gamma0), format='csr')
        return Omega, Upsilon

    i = pairs[:,0]; j = pairs[:,1]
    rvec = R[j] - R[i]
    Om, Gm = _omega_gamma_pairs(MU[i], MU[j], rvec, k, gamma0)

    # assemble symmetric sparse (only off-diagonals here)
    data_O = np.concatenate([Om, Om])
    rows_O = np.concatenate([i,  j])
    cols_O = np.concatenate([j,  i])

    data_G = np.concatenate([Gm, Gm])
    rows_G = rows_O
    cols_G = cols_O

    Omega   = csr_matrix((data_O, (rows_O, cols_O)), shape=(N, N))
    Upsilon = csr_matrix((data_G, (rows_G, cols_G)), shape=(N, N))
    # add self-decay γ0 on diagonal
    Upsilon = Upsilon + diags(np.full(N, gamma0), format='csr')
    return Omega, Upsilon

def heff_from_ou(Omega: csr_matrix, Upsilon: csr_matrix) -> csr_matrix:
    return Omega.astype(np.complex128) - 0.5j * Upsilon.astype(np.complex128)

def brightest_modes(Heff: csr_matrix, k_eigs: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = eigs(Heff, k=k_eigs, which="LM")
    return vals, vecs

def summarize_brightness(evals: np.ndarray, gamma0: float, top_n: int = 3):
    gammas = (-2.0 * np.imag(evals)).real
    order = np.argsort(gammas)[::-1]
    top = order[:top_n]
    return {
        "Gamma_over_gamma0_top": (gammas[top] / gamma0).tolist(),
        "Gamma_top": gammas[top].tolist(),
        "evals_top": evals[top].tolist(),
        "Gamma_max_over_gamma0": float(np.max(gammas) / gamma0),
    }
