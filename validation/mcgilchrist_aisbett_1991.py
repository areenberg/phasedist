"""
Comparison with a published random-effect survival model: Recurrence of
infections at the catheter insertion point of kidney patients on portable
dialysis, McGilchrist and Aisbett (1991), as analysed in the Stata manual
entry for mestreg (Examples 1 and 2).

Study: 38 patients, each with the times to two recurrences of infection
(76 times, 18 of them right-censored when the catheter was removed for
other reasons). Covariates: Age, and female (sex 2 in the R data). The
published model is a parametric survival model with a normal random
intercept per patient: Weibull on the accelerated failure time scale
(mestreg ..., distribution(weibull) time), and the same with a gamma
distribution. The data are survival::kidney from R (the same data as
Stata's catheter.dta).

Application of fitreg from PhaseDist: fitreg models
Y | x, b ~ PH(pi, exp(x'beta + b) T), so
log E(Y | x, b) = log(mu0) - x'beta - b, with mu0 the mean at x = 0 and
b = 0. The AFT coefficients of mestreg describe log time in the same way and
compare with -beta, and var(_cons[patient]) with the squared random effect
std. dev. With two times per patient that variance trades off against the
spread of the base distribution (Weibull or gamma against phase-type), so it
depends most on that choice. The number of phases is chosen by AIC.

Run from the repository root:  python validation/mcgilchrist_aisbett_1991.py

McGilchrist, C. A., & Aisbett, C. W. (1991). Regression with frailty in survival
analysis. Biometrics, 47(2), 461-466. StataCorp. Stata Multilevel Mixed-Effects
Reference Manual, mestreg.
"""

import os
import sys

import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "src"))
from phasedist.fitreg import fitreg

# Data (76 rows; from the R package survival); status 1 is an infection
raw = pd.read_csv(os.path.join(here, "data", "kidney.csv"))
infected = raw["status"] == 1
data = pd.DataFrame({
    "time": np.where(infected, raw["time"], np.nan),          # days to infection, missing if censored
    "age": raw["age"],
    "female": (raw["sex"] == 2).astype(float),
    "patient": pd.Categorical(raw["id"]),
    "lower": np.where(infected, np.nan, raw["time"]),         # right-censored: time > lower
    "upper": np.nan,
})

# Published (Stata, mestreg ..., time): AFT coefficient, standard error;
# var(_cons[patient]) with its standard error; log-likelihood; LR test of the
# random effect, chibar2(01) and its p-value
published = {
    "Weibull": dict(age=(-0.0058496, 0.010872), female=(1.325034, 0.3719102), var=(0.5304902, 0.2343675),
                    loglik=-329.87832, lr=(9.40, 0.0011)),
    "gamma": dict(age=(-0.0060276, 0.0108267), female=(1.324745, 0.3685132), var=(0.5071823, 0.2241959),
                  loglik=-329.52634, lr=(11.13, 0.0004)),
}

# Fit with one to three Coxian phases
fits = {}
for nphases in (1, 2, 3):
    fits[nphases] = fitreg(
        formula="time ~ age + female + (1|patient)",
        data=data,
        censoring=("lower", "upper"),
        nphases=nphases,
        dtype="coxian",
        restarts=10,
        seed=1,
    )
best = min(fits, key=lambda k: fits[k].getaic())

# Compare
print()
print("McGilchrist and Aisbett (1991), kidney catheter infections: Stata mestreg against fitreg")
print("AFT coefficients (fitreg: -beta) with standard errors; variance of the random intercept")
print()
print(f"{'':28s} {'age':>18s} {'female':>18s} {'variance':>18s} {'loglik':>9s} {'test of the random effect':>34s}")
for name, row in published.items():
    print(f"{'Stata, ' + name:28s} {row['age'][0]:9.4f} ({row['age'][1]:.4f}) {row['female'][0]:9.4f} ({row['female'][1]:.4f})"
          f" {row['var'][0]:9.4f} ({row['var'][1]:.4f}) {row['loglik']:9.2f}   LR {row['lr'][0]:5.2f}, p = {row['lr'][1]:.4f}")
for nphases, model in fits.items():
    coef, stderr = model.getcoef(), model.getstderr()
    sigma = model.getsigma()
    variance = sigma ** 2
    varse = 2.0 * sigma * model.sigmastderr if model.sigmastderr is not None else np.nan
    score, lrt = model.getsigmatest(), model.getsigmatest("lrt")
    label = "fitreg, %d phase%s%s" % (nphases, "" if nphases == 1 else "s", " (AIC)" if nphases == best else "")
    print(f"{label:28s} {-coef['age']:9.4f} ({stderr['age']:.4f}) {-coef['female']:9.4f} ({stderr['female']:.4f})"
          f" {variance:9.4f} ({varse:.4f}) {model.getloglik():9.2f}   LR {lrt['statistic']:5.2f}, p = {lrt['pvalue']:.4f};"
          f" score p = {score['pvalue']:.4f}")
print()
print("AIC: " + ", ".join("%d phase%s %.2f" % (k, "" if k == 1 else "s", m.getaic()) for k, m in fits.items()))
