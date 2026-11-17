# Code for: Mathematical Modeling and Optimal Control of Malaria Transmission in Post-Conflict Tigray

Requires Python >= 3.11 with the pinned versions in requirements.txt. Install with `python -m pip install -r requirements.txt`. Run from this folder in the following order. Verification and sensitivity take the most time.

| Step | Script | Produces |
|---|---|---|
| 1 | `python verify_numerics.py` | time-step refinement, initialisation/relaxation tests, adjoint gradient test, feasible comparators, robustness grid -> `verify_results.json`, `verify_hist.npz`, `verify_profiles.npz` |
| 2 | `python analyses_model.py` | R0, periodic Rp, endemic equilibrium, uniqueness/bifurcation checks -> `analysis_A_B.json`, `bifurcation_curve.npy` |
| 3 | `python analyses_scen.py` | resource-feasible fixed scenarios and timing windows (from initial seed), plausibility check -> `analysis_scen.json` |
| 4 | `python analyses_burnin.py` | timing windows from the quasi-periodic (40-year burn-in) state -> `analysis_burnin.json` |
| 5 | `python rp_curve.py` | R0 vs Rp curve -> `rp_curve.json` |
| 6 | `python sens.py` | 5000-sample LHS/PRCC (seed 42) -> `sens.json` |
| 7 | `python revision_checks.py` | threshold coefficients, monthly onset timing and feasible objective comparisons |
| 8 | `python make_figures.py` | all ten figures -> `../figs/figure1_...` to `../figs/figure10_...` (file numbers match the printed figure numbers) |

`malaria_model.py` holds the model, RK4 integrators, adjoint system, KKT control update, forward-backward sweep, R0 and Rp.
Figures 1, 2 and 4 are drawn only from values published in Hailemariam et al. (2025), listed in the table inside `make_figures.py`.

Notes
* Resource feasibility c1*u1 + c2*u2 <= R(t) is asserted for every timing scenario and checked for every sweep solution. Timing policies use exactly 90 days at high intensity and exactly 58.75 resource-unit days, including the uniform comparator.
* Sampling ranges for a, tau_h, mu_v, alpha_h, alpha_v in `sens.py` are plausibility ranges chosen for this analysis; these reproduce the ranges used in the manuscript.
* The first and last grid nodes of the computed control influence the adjacent integration intervals; the control panels plot interior nodes.
