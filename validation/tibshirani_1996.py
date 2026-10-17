"""
Replication of Tibshirani (1996), Table 1 (least squares), with fitreg.

Study: Prostate cancer data of Stamey et al. (1989), 97 men. Log PSA is
regressed on eight clinical measures, all standardized. The data in
data/prostate.csv are the original release used by Tibshirani (as in the R
package lasso2): The log prostate weight of observation 32 is 6.1076, which
later releases corrected to 3.8044.

Application of fitreg from PhaseDist: fitreg models PSA as
Y | x ~ PH(pi, exp(x'beta) T), so log E(Y | x) = log(mu0) - x'beta, with mu0
the mean at x = 0. Thus -beta is the change in the log mean PSA per unit of a
covariate, comparable to the published coefficients of log(PSA).

Run from the repository root:  python validation/tibshirani_1996.py

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.
JRSS B, 58(1), 267-288.
"""

import os
import sys

import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "src"))
from phasedist.fitreg import fitreg

# Data (97 rows)
raw = pd.read_csv(os.path.join(here, "data", "prostate.csv"))

predictors = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
data = pd.DataFrame({"psa": np.exp(raw["lpsa"])})
for name in predictors:
    data[name] = (raw[name] - raw[name].mean()) / raw[name].std()

# Published results, Table 1: Coefficient, standard error, Z score
published = {
    "lcavol": (0.69, 0.10, 6.68),
    "lweight": (0.23, 0.08, 2.67),
    "age": (-0.15, 0.08, -1.76),
    "lbph": (0.16, 0.08, 1.83),
    "svi": (0.32, 0.10, 3.14),
    "lcp": (-0.15, 0.13, -1.16),
    "gleason": (0.03, 0.11, 0.29),
    "pgg45": (0.13, 0.12, 1.02),
}

# Fit
model = fitreg(
    formula="psa ~ lcavol + lweight + age + lbph + svi + lcp + gleason + pgg45",
    data=data,
    nphases=3,
    dtype="coxian",
    restarts=10,
    seed=1,
)
coef = model.getcoef()
stderr = model.getstderr()

# Compare
print("Tibshirani (1996), Table 1: Published least squares vs fitreg (3-phase Coxian)")
print("fitreg reports -beta, which is comparable to the published coefficients.")
print()
print(f"{'':10s} {'published':>24s}   {'fitreg':>24s}")
print(f"{'':10s} {'coef':>8s} {'se':>7s} {'Z':>7s}   {'coef':>8s} {'se':>7s} {'z':>7s}")
for name in predictors:
    pub_coef, pub_se, pub_z = published[name]
    fit_coef = -coef[name]
    fit_se = stderr[name]
    print(f"{name:10s} {pub_coef:8.2f} {pub_se:7.2f} {pub_z:7.2f}   {fit_coef:8.2f} {fit_se:7.2f} {fit_coef / fit_se:7.2f}")
print()
print(f"fitreg log-likelihood of PSA: {model.getloglik():.2f}")