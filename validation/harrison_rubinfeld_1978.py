"""
Replication of Harrison and Rubinfeld (1978), Table VII, equation 1, with fitreg.

Study: Hedonic house prices in 506 Boston census tracts. The log median home
value is regressed on 13 tract characteristics, among them air pollution
(NOX squared). The variables are built below as in the paper.

Why fitreg: The home value is positive and the model is multiplicative,
log(value) = x'b + error. fitreg fits log(value) = -x'beta + log(U) with a
phase-type U, so -beta is comparable to the published coefficients.

Run from the repository root:  python validation/harrison_rubinfeld_1978.py
(runtime is about 15 minutes)

Harrison, D., & Rubinfeld, D. L. (1978). Hedonic housing prices and the
demand for clean air. Journal of Environmental Economics and Management,
5(1), 81-102.
"""

import os
import sys

import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "src"))
from phasedist.fitreg import fitreg

# Data (506 rows from the R package MASS), medv is the median home value in $1000
boston = pd.read_csv(os.path.join(here, "data", "boston.csv"))
data = pd.DataFrame({
    "value": boston["medv"],
    "rm2": boston["rm"] ** 2,
    "age": boston["age"],
    "logdis": np.log(boston["dis"]),
    "lograd": np.log(boston["rad"]),
    "tax": boston["tax"],
    "ptratio": boston["ptratio"],
    "b": boston["black"] / 1000,  # (B - 0.63)^2, B the proportion of black residents
    "loglstat": np.log(boston["lstat"] / 100),
    "crim": boston["crim"],
    "zn": boston["zn"],
    "indus": boston["indus"],
    "chas": boston["chas"],
    "nox2": (10 * boston["nox"]) ** 2,
})

# Published results, Table VII equation 1: Coefficient, t value
# (the t value of age reads 1.7 in the scanned table, its coefficient and
# standard error give 0.17)
published = {
    "rm2": (0.0063, 4.83),
    "age": (0.0000898, 0.17),
    "logdis": (-0.19, -5.73),
    "lograd": (0.096, 5.00),
    "tax": (-0.00042, -3.43),
    "ptratio": (-0.031, -6.21),
    "b": (0.36, 3.53),
    "loglstat": (-0.37, -14.84),
    "crim": (-0.012, -9.53),
    "zn": (0.0000803, 0.16),
    "indus": (0.000241, 0.10),
    "chas": (0.088, 2.75),
    "nox2": (-0.0064, -5.64),
}

# Fit
model = fitreg(
    formula="value ~ rm2 + age + logdis + lograd + tax + ptratio + b + loglstat + crim + zn + indus + chas + nox2",
    data=data,
    nphases=30,
    dtype="generlang",
    restarts=3,
    seed=1,
)
coef = model.getcoef()
stderr = model.getstderr()

# Compare
print("Harrison and Rubinfeld (1978), Table VII eq. 1: published OLS vs fitreg (30-phase generalized Erlang)")
print("fitreg reports -beta, which is comparable to the published coefficients.")
print()
print(f"{'':10s} {'published':>20s}   {'fitreg':>20s}")
print(f"{'':10s} {'coef':>12s} {'t':>7s}   {'coef':>12s} {'z':>7s}")
for name in published:
    pub_coef, pub_t = published[name]
    fit_coef = -coef[name]
    fit_z = fit_coef / stderr[name]
    print(f"{name:10s} {pub_coef:12.4g} {pub_t:7.2f}   {fit_coef:12.4g} {fit_z:7.2f}")
print()
print(f"fitreg log-likelihood of the value: {model.getloglik():.2f}")