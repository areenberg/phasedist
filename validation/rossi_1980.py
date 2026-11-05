"""
Comparison with a published Cox regression: Recidivism of released prisoners
(Rossi, Berk and Lenihan, 1980), as analysed by Fox and Weisberg (2018).

Study: 432 men released from Maryland state prisons were followed for one
year (52 weeks). The response is the week of the first arrest; 318 men were
not arrested and are right-censored at week 52. Fox and Weisberg fit a Cox
model with financial aid (fin), age, race, work experience (wexp), marital
status (mar), parole (paro) and the number of prior convictions (prio).

Application of fitreg from PhaseDist: fitreg models the right-censored
time to arrest as Y | x ~ PH(pi, exp(x'beta) T), so
log E(Y | x) = log(mu0) - x'beta, with mu0 the mean at x = 0: A positive beta
shortens the mean time, as a positive Cox coefficient raises the hazard. The
Cox coefficient acts on the hazard; its counterpart is kappa * beta, with
kappa = d log h(y | x) / d eta averaged over the observed arrest times (1 for
an exponential baseline, the shape for a Weibull). The signs and z values
compare directly.

Run from the repository root:  python validation/rossi_1980.py

Rossi, P. H., Berk, R. A., & Lenihan, K. J. (1980). Money, Work and Crime:
Some Experimental Results. Academic Press.
Fox, J., & Weisberg, S. (2018). Cox proportional-hazards regression for
survival data in R. Appendix to An R Companion to Applied Regression, 3rd ed.
"""

import os
import sys

import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "src"))
from phasedist.fitreg import fitreg

# Data (432 rows; from the R package carData)
raw = pd.read_csv(os.path.join(here, "data", "rossi.csv"))
arrested = raw["arrest"] == 1
data = pd.DataFrame({
    "week": np.where(arrested, raw["week"], np.nan),       # arrest week, missing if censored
    "fin": (raw["fin"] == "yes") * 1.0,
    "age": raw["age"] * 1.0,
    "race": (raw["race"] == "other") * 1.0,
    "wexp": (raw["wexp"] == "yes") * 1.0,
    "mar": (raw["mar"] == "not married") * 1.0,
    "paro": (raw["paro"] == "yes") * 1.0,
    "prio": raw["prio"] * 1.0,
    "lower": np.where(arrested, np.nan, raw["week"]),       # right-censored: week > 52
    "upper": np.nan,
})

# Published Cox regression (Fox and Weisberg 2018): Coefficient, standard error
published = {
    "fin": (-0.3794, 0.1914),
    "age": (-0.0574, 0.0220),
    "race": (-0.3139, 0.3080),
    "wexp": (-0.1498, 0.2122),
    "mar": (0.4337, 0.3819),
    "paro": (-0.0849, 0.1958),
    "prio": (0.0915, 0.0286),
}

# Fit (two phases are chosen by AIC among one to three)
model = fitreg(
    formula="week ~ fin + age + race + wexp + mar + paro + prio",
    data=data,
    censoring=("lower", "upper"),
    nphases=2,
    dtype="coxian",
    restarts=5,
    seed=1,
)
coef = model.getcoef()
stderr = model.getstderr()

# kappa: The slope of log(u * baseline hazard(u)) in log(u), averaged over the
# transformed arrest times u = exp(x'beta) * week
baseline = model.getdist()


def log_u_hazard(u):
    return np.log(u * baseline.getdensity(u) / (1.0 - baseline.getcumprob(u)))


names = list(published)
eta = data[names].to_numpy() @ np.array([coef[name] for name in names])
times = np.exp(eta[arrested]) * data["week"][arrested].to_numpy()
step = 1.001
kappa = np.mean([(log_u_hazard(u * step) - log_u_hazard(u / step)) / (2 * np.log(step)) for u in times])

# Compare
print("Rossi et al. (1980) recidivism: Published Cox regression vs fitreg (2-phase Coxian)")
print("Both coefficients are positive for a higher hazard; kappa * beta is fitreg on the hazard scale.")
print()
print(f"{'':6s} {'published Cox':>27s}   {'fitreg':>27s}   {'fitreg':>10s}")
print(f"{'':6s} {'coef':>9s} {'se':>8s} {'z':>8s}   {'beta':>9s} {'se':>8s} {'z':>8s}   {'kappa*beta':>10s}")
for name in names:
    pub_coef, pub_se = published[name]
    print(f"{name:6s} {pub_coef:9.4f} {pub_se:8.4f} {pub_coef / pub_se:8.2f}   "
          f"{coef[name]:9.4f} {stderr[name]:8.4f} {coef[name] / stderr[name]:8.2f}   {kappa * coef[name]:10.4f}")
print()
print(f"kappa = {kappa:.3f} (1 for an exponential baseline)")
print(f"fitreg log-likelihood: {model.getloglik():.2f}")
