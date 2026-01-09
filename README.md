# Battery Fast-Charge Pack MBRL

This repository provides a runnable, paper-style project scaffold inspired by
"Adaptive Model-Based Reinforcement Learning for Fast-Charging Optimization of
Lithium-Ion Batteries" and extended to a 3P6S pack with aging, cell heterogeneity,
and an external SOH prior (STAPINN-like) for fast online calibration.

## Highlights
- Pack-level environment with max-cell voltage/temperature constraints.
- Static, differential, and combined surrogate models (NN-backed for now).
- DDPG actor-critic with replaceable trainer logic for Cycle0/Adaptive cycles.
- SOH prior pipeline with a fast calibration stage that regularizes to prior.
- Fully runnable scripts with toy backend when liionpack/pybamm are unavailable.

## Quick Start
```bash
pip install -r requirements.txt
python scripts/train_cycle0_build_static_gp.py --config configs/pack_3p6s_spme.yaml
python scripts/train_adaptive_cycles.py --config configs/pack_3p6s_spme_with_soh_prior.yaml
python scripts/evaluate_policy.py --config configs/pack_3p6s_spme.yaml --ckpt runs/cycle0/policy.pt
```

## Project Layout
See the config files in `configs/` for key hyperparameters, sampling step,
reward weights, and environment settings.
