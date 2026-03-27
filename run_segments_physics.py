# run_segments_physics.py
# this too an amalgamation of many different sources, cited of course in the thesis
import importlib, os, json, numpy as np
from multiprocessing import Pool, cpu_count
from scipy.sparse import diags

G  = importlib.import_module("geometry")
PH = importlib.import_module("physics_hook_fast")   # vectorized/numba-optional (numba is much faster but good luck)
C  = importlib.import_module("SIM_constants")       # must define k, gamma0, r_cut
try:
    print(f"[constants] n={getattr(C,'n_med',None)}  L={getattr(C,'L',1.0):.3f}  "
          f"k={C.k:.3e} 1/m  gamma0={C.gamma0:.3e} s^-1")
except Exception:
    pass

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--segments", type=int, default=30, help="number of segments")
    p.add_argument("--slice_um", type=float, default=3.0, help="slice length in microns")
    p.add_argument("--r_cut_nm", type=float, default=300.0, help="neighbor cutoff in nm")
    p.add_argument("--k_eigs", type=int, default=8, help="# of brightest modes to compute")
    p.add_argument("--orientation", type=str, default="tangential",
                   choices=["tangential","radial","axial"])
    p.add_argument("--sigma", type=float, default=0.0,
                   help="static diagonal disorder std (s^-1); 0 disables")
    # inside parse_args() in run_segments_physics.py
    p.add_argument("--orient_sigma_deg", type=float, default=0.0,
               help="per-site dipole tilt std dev (degrees)")
    p.add_argument("--pos_sigma_nm", type=float, default=0.0,
               help="per-site positional jitter std dev (nm)")
    p.add_argument("--vacancy", type=float, default=0.0,
               help="fraction of sites randomly removed [0..1]")

    p.add_argument("--qy_single", type=float, default=0.12,
                   help="single-Trp quantum yield")
    p.add_argument("--outfile", type=str, default="segments_results.jsonl",
                   help="results path (JSONL append)")
    p.add_argument("--seed", type=int, default=12345)
    
    p.add_argument("--n_mt", type=int, default=1,
               help="number of parallel microtubules in a segment (1,2,3,4)")
    p.add_argument("--mt_layout", type=str, default="line",
               choices=["line","triangle","square"],
               help="layout of MT centers for n_mt>1")
    p.add_argument("--center_spacing_nm", type=float, default=35.0,
               help="center-to-center spacing between MT axes (nm)")
    p.add_argument("--workers", type=int, default=max(1, cpu_count()-1),
               help="number of parallel worker processes")


    
    return p.parse_args()

ARGS = parse_args()

#knobs (from CLI args) 
L_SLICE_M    = ARGS.slice_um * 1e-6
N_SEGMENTS   = ARGS.segments
ORIENTATION  = ARGS.orientation
A_AXIAL      = 8.0e-9   # dimer spacing stays fixed
K_EIGS       = ARGS.k_eigs
RESULTS_PATH = ARGS.outfile
RNG_SEED     = ARGS.seed

# QY and non-radiative rate
QY_SINGLE = ARGS.qy_single
GAMMA_NR  = C.gamma0 * (1.0 - QY_SINGLE) / QY_SINGLE  # ≈ 7.33*gamma0 if QY_SINGLE=0.12

# Override cutoff in constants (so physics uses the CLI cutoff) DO NOT CHANGE OTHERWISE CLI DOESN'T LET YOU SET PARAMETERS
C.r_cut = ARGS.r_cut_nm * 1e-9


def n_axial_from_length(Lm, a_ax=A_AXIAL):
    n = int(max(1, np.floor(Lm / a_ax) + 1))
    return n

def participation_ratio(v):
    v = v / np.linalg.norm(v)
    return float(1.0 / np.sum(np.abs(v)**4))

def _random_unit_vectors(n, rng):
    a = rng.normal(size=(n, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a

def _rodrigues_batch(v, axes, angles):
    """
    Rotate each row of v by angle[i] around axes[i] (all (N,3)).
    v_rot = v*cosθ + (k×v)*sinθ + k*(k·v)*(1−cosθ)
    Entirely stolen from a stack exchange post i'm unable to recover
    """
    c = np.cos(angles)[:, None]
    s = np.sin(angles)[:, None]
    kv = np.sum(axes * v, axis=1, keepdims=True)        # (N,1)
    k_cross_v = np.cross(axes, v)                        # (N,3)
    return v * c + k_cross_v * s + axes * kv * (1.0 - c)



def build_one_segment(rng: np.random.Generator):
    n_ax = n_axial_from_length(L_SLICE_M, A_AXIAL)

    if ARGS.n_mt == 1:
        # single MT (your old code)
        phi0 = float(rng.uniform(0.0, 2.0*np.pi))
        dz   = float(rng.uniform(0.0, A_AXIAL))
        R, MU = G.microtubule_13proto(
            n_axial=n_ax, a_axial=A_AXIAL,
            radius=12.5e-9, helix_pitch=None,
            orientation=ORIENTATION, phi0=phi0, dz=dz
        )
        labels = np.zeros(R.shape[0], dtype=np.int32)
    else:
        # multi-MT slice
        R, MU, labels = G.multi_microtubules(
            n_mt=ARGS.n_mt, n_axial=n_ax, a_axial=A_AXIAL,
            radius=12.5e-9, orientation=ORIENTATION,
            layout=ARGS.mt_layout,
            center_spacing=ARGS.center_spacing_nm * 1e-9,
            rng=rng
        )
        phi0, dz = None, None  # not meaningful for multi-MT

    # APPLY STRUCTURAL IMPERFECTIONS, DISORDER ETC ETC
    if ARGS.pos_sigma_nm > 0.0:
        R = R + rng.normal(0.0, ARGS.pos_sigma_nm * 1e-9, size=R.shape)

    if ARGS.orient_sigma_deg > 0.0:
        N = MU.shape[0]
        axes = _random_unit_vectors(N, rng)
        angles = rng.normal(0.0, np.deg2rad(ARGS.orient_sigma_deg), size=N)
        MU = _rodrigues_batch(MU, axes, angles)
        MU /= np.linalg.norm(MU, axis=1, keepdims=True)

    if ARGS.vacancy > 0.0:
        keep = rng.random(R.shape[0]) > ARGS.vacancy
        if not np.any(keep):
            keep[rng.integers(0, R.shape[0])] = True
        R, MU, labels = R[keep], MU[keep], labels[keep]
    # END DISORDER
    return dict(R=R, MU=MU, labels=labels, phi0=phi0, dz=dz, n_axial=n_ax)

def run_one(idx_seg_and_seg):
    idx, seg = idx_seg_and_seg
    # per-worker RNG (reproducible and independent across processes)
    rng = np.random.default_rng((1234567, idx))

    # disorder strength (standard deviation) in s^-1 (from CLI)
    sigma = ARGS.sigma

    R, MU = seg["R"], seg["MU"]
    N = R.shape[0]

    # Build Ω, Υ
    Omega, Upsilon = PH.build_sparse_mats(R, MU, C.k, C.gamma0, C.r_cut)
    Omega = Omega.astype(np.float32)
    Upsilon = Upsilon.astype(np.float32)

    # Add static diagonal disorder (site energies) to Ω for realism:
    if sigma > 0.0:
        delta = rng.normal(loc=0.0, scale=sigma, size=N)
        Omega = Omega + diags(delta, offsets=0, format="csr")

    # H_eff = Ω - i Υ/2
    Heff = PH.heff_from_ou(Omega, Upsilon)

    # Brightest modes
    vals, vecs = PH.brightest_modes(Heff, k_eigs=K_EIGS)

    
    
    
    # Radiative rates Γ_j from eigenvalues (compute ONCE)
    gammas = (-2.0*np.imag(vals)).real
    bidx = int(np.argmax(gammas))
    Gamma_bright = float(gammas[bidx])

    # Quantum yield for brightest mode and for all returned modes
    QY_bright = Gamma_bright / (Gamma_bright + GAMMA_NR)
    order = np.argsort(gammas)[::-1]
    QY_all = (gammas[order] / (gammas[order] + GAMMA_NR)).astype(float).tolist()

    # Summaries (uses vals to get Γ/γ0 ratios for top modes)
    summ = PH.summarize_brightness(vals, C.gamma0, top_n=min(3, K_EIGS))

    # Participation ratio of the brightest state
    labels = seg["labels"]
    v = vecs[:, bidx]
    v      = v / np.linalg.norm(v)
    PR = float(1.0 / np.sum(np.abs(v)**4))

# weight on each MT (sum of |v|^2 over sites belonging to that MT)
    n_mt_present = int(labels.max()) + 1
    w_per_mt = []
    for m in range(n_mt_present):
        mask = (labels == m)
        w_per_mt.append(float(np.sum(np.abs(v[mask])**2)))
  

    row = {
        "idx": idx,
        "n_axial": int(seg["n_axial"]),
        "N_sites": int(N),
        "phi0": seg["phi0"],
        "dz_m": seg["dz"],
        "Gamma_max_over_gamma0": float(summ["Gamma_max_over_gamma0"]),
        "Gamma_over_gamma0_top": [float(x) for x in summ["Gamma_over_gamma0_top"]],
        "PR_brightest": float(PR),
        "sigma_disorder_s^-1": float(sigma),
        # QY & rates
        "Gamma_bright": Gamma_bright,
        "QY_bright": float(QY_bright),
        "gamma_nr": float(GAMMA_NR),
        "QY_single_assumed": float(QY_SINGLE),
        "QY_top": QY_all[:min(3, len(QY_all))],
        # multi microtubule info REMEMBER DONT TRY 5 MTS OR MACHINE CRASHES
        "n_mt": int(ARGS.n_mt),
        "mt_layout": ARGS.mt_layout,
        "center_spacing_nm": float(ARGS.center_spacing_nm),
        "bright_weights_per_mt": w_per_mt
    }

    # Save run configuration 
    row.update({
    "config": {
        "slice_um": ARGS.slice_um,
        "r_cut_nm": ARGS.r_cut_nm,
        "k_eigs": ARGS.k_eigs,
        "orientation": ARGS.orientation,
        "sigma": ARGS.sigma,
        "qy_single": ARGS.qy_single,
        "orient_sigma_deg": ARGS.orient_sigma_deg,
        "pos_sigma_nm": ARGS.pos_sigma_nm,
        "vacancy": ARGS.vacancy,
        "n_mt": int(ARGS.n_mt),
        "mt_layout": ARGS.mt_layout,
        "center_spacing_nm": float(ARGS.center_spacing_nm),
        }
    })


    return row

def save_row(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")

def main():
    rng = np.random.default_rng(RNG_SEED)
    segments = [build_one_segment(rng) for _ in range(N_SEGMENTS)]

    # resume-safe
    done = set()
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(int(obj["idx"]))
                except Exception:
                    pass

    work = [(i, seg) for i, seg in enumerate(segments) if i not in done]
    print(f"Slice length: {L_SLICE_M*1e6:.2f} µm | n_axial={segments[0]['n_axial']} | sites/MT={segments[0]['R'].shape[0]}")
    print(f"Segments to run: {len(work)} (skipping {len(segments)-len(work)} already done)")
    print(f"Cutoff: {ARGS.r_cut_nm:.0f} nm | k_eigs={K_EIGS} | orientation={ORIENTATION} | "
          f"sigma={ARGS.sigma:.2e} s^-1 | QY_single={QY_SINGLE:.2f}"
          f"tiltσ={ARGS.orient_sigma_deg:.1f}° | posσ={ARGS.pos_sigma_nm:.2f} nm | vac={ARGS.vacancy:.2f}")
    print(f"MTs: n_mt={ARGS.n_mt}, layout={ARGS.mt_layout}, center_spacing={ARGS.center_spacing_nm:.1f} nm")


    
    # parallel + progress
    n_workers = ARGS.workers
    try:
        from tqdm import tqdm
        iterator = tqdm
    except Exception:
        iterator = lambda x, **k: x

    with Pool(processes=n_workers) as pool:
        for row in iterator(pool.imap_unordered(run_one, work, chunksize=1),
                            total=len(work), desc="Segments"):
            save_row(RESULTS_PATH, row)

    # aggregate
    data = []
    with open(RESULTS_PATH, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception:
                pass

    if not data:
        print("\n== SUMMARY ==\n(no results found — did the run finish?)")
        return

    g  = np.array([d["Gamma_max_over_gamma0"] for d in data], float)
    pr = np.array([d["PR_brightest"]        for d in data], float)
    Ns = np.array([d["N_sites"]             for d in data], int)

    def q(a, p): return float(np.quantile(a, p))

    print("\n== SUMMARY ==")
    print("segments:", len(data))
    print("sites/segment median [p10, p90]:", int(np.median(Ns)),
          f"[{int(q(Ns,0.10))}, {int(q(Ns,0.90))}]")
    print("Γ_max/γ0  median [p10, p90]  mean:",
          f"{np.median(g):.2f}",
          f"[{q(g,0.10):.2f}, {q(g,0.90):.2f}]",
          f"{g.mean():.2f}")
    print("PR_bright median [p10, p90]  mean:",
          f"{np.median(pr):.1f}",
          f"[{q(pr,0.10):.1f}, {q(pr,0.90):.1f}]",
          f"{pr.mean():.1f}")

    # QY summary (only if present in rows)
    if "QY_bright" in data[0]:
        qy = np.array([d["QY_bright"] for d in data], float)
        print("QY_bright median [p10, p90]  mean:",
              f"{np.median(qy):.3f}",
              f"[{q(qy,0.10):.3f}, {q(qy,0.90):.3f}]",
              f"{qy.mean():.3f}")
    else:
        print("QY_bright: (not available — did you add QY fields to each row in run_one?)")

if __name__ == "__main__":
    main()
