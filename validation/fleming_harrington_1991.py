"""
Comparison with a published Cox regression: Survival in primary biliary
cirrhosis (Mayo Clinic), Fleming and Harrington (1991), Table 4.6.3.

Study: Patients with primary biliary cirrhosis, a liver disease, seen at the
Mayo Clinic in 1974-1984 and followed until death or censoring (liver
transplantation or the end of follow-up). Fleming and Harrington's final Cox
model has log bilirubin, log prothrombin time, log albumin, age and edema
(0, 0.5 or 1). Their data version has 418 patients and 161 deaths; the R
version used here is nearly identical and has 416 patients with all five
covariates, 160 deaths.

Application of fitreg from PhaseDist: fitreg models the right-censored
survival time as Y | x ~ PH(pi, exp(x'beta) T), so
log E(Y | x) = log(mu0) - x'beta, with mu0 the mean at x = 0: A positive beta
shortens the mean time, as a positive Cox coefficient raises the hazard. The
Cox coefficient acts on the hazard; its counterpart is kappa * beta, with
kappa = d log h(y | x) / d eta averaged over the observed death times (1 for
an exponential baseline, the shape for a Weibull). The signs and z values
compare directly.

Run from the repository root:  python validation/fleming_harrington_1991.py

Fleming, T. R., & Harrington, D. P. (1991). Counting Processes and Survival
Analysis. Wiley. (The standard errors below are those SAS prints for the same
model and data in the PROC PHREG documentation.)
"""

import os
import sys

import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "src"))
from phasedist.fitreg import fitreg

# Data (418 rows; from the R package survival); status 2 is death
raw = pd.read_csv(os.path.join(here, "data", "pbc.csv"))
raw = raw[raw["protime"].notna()]
died = raw["status"] == 2
data = pd.DataFrame({
    "time": np.where(died, raw["time"], np.nan),           # days to death, missing if censored
    "logbili": np.log(raw["bili"]),
    "logprotime": np.log(raw["protime"]),
    "logalb": np.log(raw["albumin"]),
    "age": raw["age"],
    "edema": raw["edema"],
    "lower": np.where(died, np.nan, raw["time"]),          # right-censored: time > lower
    "upper": np.nan,
})

# Published Cox regression (Fleming and Harrington 1991): Coefficient, standard error
published = {
    "logbili": (0.8707, 0.0826),
    "logprotime": (2.3800, 0.7667),
    "logalb": (-2.5330, 0.6482),
    "age": (0.0394, 0.0077),
    "edema": (0.8592, 0.2711),
}

# Fit (two phases are chosen by AIC among one to three)
model = fitreg(
    formula="time ~ logbili + logprotime + logalb + age + edema",
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
# transformed death times u = exp(x'beta) * time
baseline = model.getdist()


def log_u_hazard(u):
    return np.log(u * baseline.getdensity(u) / (1.0 - baseline.getcumprob(u)))


names = list(published)
eta = data[names].to_numpy() @ np.array([coef[name] for name in names])
times = np.exp(eta[died]) * data["time"][died].to_numpy()
step = 1.001
kappa = np.mean([(log_u_hazard(u * step) - log_u_hazard(u / step)) / (2 * np.log(step)) for u in times])

# Compare
print("Fleming and Harrington (1991), PBC: Published Cox regression vs fitreg (2-phase Coxian)")
print("Both coefficients are positive for a higher hazard; kappa * beta is fitreg on the hazard scale.")
print()
print(f"{'':10s} {'published Cox':>27s}   {'fitreg':>27s}   {'fitreg':>10s}")
print(f"{'':10s} {'coef':>9s} {'se':>8s} {'z':>8s}   {'beta':>9s} {'se':>8s} {'z':>8s}   {'kappa*beta':>10s}")
for name in names:
    pub_coef, pub_se = published[name]
    print(f"{name:10s} {pub_coef:9.4f} {pub_se:8.4f} {pub_coef / pub_se:8.2f}   "
          f"{coef[name]:9.4f} {stderr[name]:8.4f} {coef[name] / stderr[name]:8.2f}   {kappa * coef[name]:10.4f}")
print()
print(f"kappa = {kappa:.3f} (1 for an exponential baseline)")
print(f"fitreg log-likelihood: {model.getloglik():.2f}")
