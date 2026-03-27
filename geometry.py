# geometry.py

import numpy as np

def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-30, None)

def _random_unit_vectors(n, rng):
    a = rng.normal(size=(n, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a

# straight protofilament 1d line along z
def protofilament(n_sites: int, spacing: float = 8.0e-9, mu_dir=(1,0,0)):
    """

    """
    z = np.arange(n_sites, dtype=float) * spacing
    R = np.c_[np.zeros_like(z), np.zeros_like(z), z]       # (N,3)
    MU = np.tile(_unit(mu_dir), (n_sites,1))               # (N,3)
    return R, MU

# ring on xy-plane 
def ring(n_sites: int, radius: float, orientation: str = "tangential"):

    phi = 2*np.pi*np.arange(n_sites)/n_sites
    x, y = radius*np.cos(phi), radius*np.sin(phi)
    R = np.c_[x, y, np.zeros_like(x)]

    if orientation == "tangential":
        MU = np.c_[-np.sin(phi), np.cos(phi), np.zeros_like(phi)]
    elif orientation == "radial":
        MU = np.c_[np.cos(phi), np.sin(phi), np.zeros_like(phi)]
    elif orientation == "z":
        MU = np.c_[np.zeros_like(phi), np.zeros_like(phi), np.ones_like(phi)]
    else:
        raise ValueError("orientation must be tangential|radial|z")

    MU = _unit(MU)
    return R, MU

#  13-proto microtubule (simple helical scaffold) 
def microtubule_13proto(n_axial: int,
                        a_axial: float = 8.0e-9,
                        radius: float = 12.5e-9,
                        helix_pitch: float = 3*8.0e-9,
                        orientation: str = "tangential",
                        phi0: float = 0.0,
                        dz: float = 0.0):
    """
    
    """
    P = 13
    k_idx = np.arange(n_axial, dtype=float)
    p_idx = np.arange(P, dtype=float)

    # Mesh over (protofilament p, axial index k)
    k_grid, p_grid = np.meshgrid(k_idx, p_idx, indexing="ij")
    dphi = 2*np.pi / P if helix_pitch is None else 2*np.pi*(a_axial/helix_pitch)


    phi = 2*np.pi*(p_grid / P) + k_grid * dphi + phi0  # helical advance
    
    z = k_grid * a_axial + dz
    x = radius*np.cos(phi)
    y = radius*np.sin(phi)

    R = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    if orientation == "tangential":
        MU = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)
    elif orientation == "radial":
        MU = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1)
    elif orientation == "axial":
        MU = np.stack([np.zeros_like(phi), np.zeros_like(phi), np.ones_like(phi)], axis=-1)
    else:
        raise ValueError("orientation must be tangential|radial|axial")

    MU = _unit(MU.reshape(-1,3))
    return R, MU


def multi_microtubules(n_mt: int,
                       n_axial: int,
                       a_axial: float = 8.0e-9,
                       radius: float = 12.5e-9,
                       orientation: str = "tangential",
                       layout: str = "line",
                       center_spacing: float = 35e-9,
                       rng: np.random.Generator = None):
    """
    
    """
    assert n_mt >= 1
    if rng is None:
        rng = np.random.default_rng()

    # Place MT centers in xy-plane
    if layout == "line":
        xs = (np.arange(n_mt) - (n_mt-1)/2.0) * center_spacing
        centers = np.c_[xs, np.zeros_like(xs), np.zeros_like(xs)]
    elif layout == "triangle":
        assert n_mt == 3, "triangle layout requires n_mt=3"
        s = center_spacing
        centers = np.array([
            [0.0,        +s/np.sqrt(3), 0.0],
            [-s/2.0,     -s/(2*np.sqrt(3)), 0.0],
            [+s/2.0,     -s/(2*np.sqrt(3)), 0.0],
        ])
    elif layout == "square":
        assert n_mt == 4, "square layout requires n_mt=4"
        s = center_spacing / np.sqrt(2)  # so nearest-neighbor distance is approximately center_spacing
        centers = np.array([
            [-s, -s, 0.0],
            [-s, +s, 0.0],
            [+s, -s, 0.0],
            [+s, +s, 0.0],
        ])
    else:
        raise ValueError("layout must be: line | triangle | square")

    R_all = []
    MU_all = []
    labels = []

    for m in range(n_mt):
        phi0 = float(rng.uniform(0.0, 2.0*np.pi))
        dz   = float(rng.uniform(0.0, a_axial))
        R, MU = microtubule_13proto(n_axial=n_axial, a_axial=a_axial,
                                    radius=radius, helix_pitch=None,
                                    orientation=orientation, phi0=phi0, dz=dz)
        # translate by center
        R = R + centers[m][None, :]
        R_all.append(R)
        MU_all.append(MU)
        labels.append(np.full(R.shape[0], m, dtype=np.int32))

    R_out  = np.vstack(R_all)
    MU_out = np.vstack(MU_all)
    labels = np.concatenate(labels)
    return R_out, MU_out, labels