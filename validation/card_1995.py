"""
Replication of Card (1995), Table 2, column (1), with fitreg.

Study: Returns to schooling, 3010 men from the US National Longitudinal
Survey of Young Men (1976). The log hourly wage is regressed on education,
experience, experience squared and indicators for black, living in a city
(SMSA) and living in the South. Card's chapter is a scanned book chapter, so
the published numbers are those printed in the vignette of the R package
ivreg, which documents this regression as Card's Table 2(1). The experience
terms are printed there only in another form and are not compared.

Application of fitreg from PhaseDist: fitreg models the wage as
Y | x ~ PH(pi, exp(x'beta) T), so log E(Y | x) = log(mu0) - x'beta, with mu0
the mean at x = 0. Thus -beta is the change in the log mean wage per unit of
a covariate, comparable to the published coefficients of log(wage).

Run from the repository root:  python validation/card_1995.py

Card, D. (1995). Using geographic variation in college proximity to estimate
the return to schooling. In Aspects of Labour Market Behaviour (pp. 201-222).
University of Toronto Press.
"""

import os
import sys

import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "src"))
from phasedist.fitreg import fitreg

# Data (3010 rows from the R package wooldridge), wage is the hourly wage in cents
data = pd.read_csv(os.path.join(here, "data", "card.csv"))

# Published results: Coefficient, standard error
published = {
    "educ": (0.074009, 0.003505),
    "black": (-0.189632, 0.017627),
    "smsa": (0.161423, 0.015573),
    "south": (-0.124862, 0.015118),
}

# Fit
model = fitreg(
    formula="wage ~ educ + exper + expersq + black + smsa + south",
    data=data,
    nphases=8,
    dtype="coxian",
    restarts=5,
    seed=1,
)
coef = model.getcoef()
stderr = model.getstderr()

# Compare
print("Card (1995), Table 2(1): Published OLS vs fitreg (8-phase Coxian)")
print("fitreg reports -beta, which is comparable to the published coefficients.")
print()
print(f"{'':10s} {'published':>19s}   {'fitreg':>19s}")
print(f"{'':10s} {'coef':>9s} {'se':>9s}   {'coef':>9s} {'se':>9s}")
for name in ["educ", "exper", "expersq", "black", "smsa", "south"]:
    fit_coef = -coef[name]
    fit_se = stderr[name]
    if name in published:
        pub_coef, pub_se = published[name]
        print(f"{name:10s} {pub_coef:9.4f} {pub_se:9.4f}   {fit_coef:9.4f} {fit_se:9.4f}")
    else:
        print(f"{name:10s} {'(not printed)':>19s}   {fit_coef:9.4f} {fit_se:9.4f}")
print()
print(f"Return to one more year of schooling: published {published['educ'][0]:.4f}, fitreg {-coef['educ']:.4f}")
print(f"fitreg log-likelihood of the wage: {model.getloglik():.2f}")