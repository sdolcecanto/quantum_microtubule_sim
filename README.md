# Quantum Microtubule Simulation

This project is a simulation of interacting dipole systems arranged in microtubule-like structures. The goal is to study how local electromagnetic interactions can lead to collective behavior in biological systems.

The work comes from my Master's thesis and focuses on building a computational model rather than just theoretical discussion.

---

## What it does

- Builds microtubule-inspired geometries (single and multiple structures)
- Models dipole–dipole interactions using physically grounded equations
- Constructs sparse interaction matrices for large systems
- Computes eigenmodes of a non-Hermitian Hamiltonian
- Extracts collective effects like enhanced radiative behavior
- Supports disorder (noise, vacancies, orientation changes)
- Runs simulations in parallel across multiple segments

---

## Why this is interesting

Instead of looking at individual components in isolation, this model explores how structure + interaction can produce emergent behavior at a larger scale.

From a technical point of view, it's mainly about:
- turning a theoretical model into working code
- handling large interacting systems efficiently
- extracting meaningful signals from complex simulations

This work moves beyond idealized models by explicitly incorporating disorder, one of the main criticisms of quantum biological effects, and evaluates its impact on collective behavior.

---

## Project structure

- `geometry.py` — builds the spatial structure of the system  
- `physics_hook_fast.py` — interaction model and matrix construction  
- `SIM_constants.py` — physical constants and configuration  
- `run_segments_physics.py` — main simulation pipeline  

---

## Running the simulation

Example:
```bash
python run_segments_physics.py \
  --segments 30 \
  --slice_um 3.0 \
  --r_cut_nm 300 \
  --k_eigs 8 \
  --orientation tangential \
  --sigma 5e8 \
  --orient_sigma_deg 3 \
  --pos_sigma_nm 0.1 \
  --vacancy 0.01 \
  --qy_single 0.12 \
  --n_mt 4 \
  --mt_layout square \
  --center_spacing_nm 35 \
  --outfile results_4mt_square_35nm.jsonl



