"""
Simulation validation of fitreg's tests and confidence intervals: Do they
hold their stated level?

A test at level 5% must reject a true null hypothesis in 5% of data sets, so
its p-values must be uniform under the null; a 95% confidence interval must
contain the true value in 95% of data sets. Each experiment simulates data
from the phase-type regression of Bladt (2022), y = U exp(-eta) with
eta = 0.8 x + beta_z z, x ~ N(0, 1), z a balanced binary covariate, and U from
the baseline distribution, and counts how often the defaults of fitreg
reject or cover:

Experiment 1  Likelihood-ratio tests and profile-likelihood intervals, the
              defaults of coeftest and confint. Under the null beta_z = 0 the
              test of z must reject at its level; the interval of beta_x
              (true value 0.8) must cover at 95%. Five settings: exponential
              baseline at n = 50 and 200, Coxian baseline at n = 100 (fitreg's
              defaults), 30% right-censored responses, and a random intercept.
              The n = 200 setting also has a three-level factor w without
              effect, whose likelihood-ratio test (getlrtest, 2 degrees of
              freedom) must hold its level too. Wald intervals are shown for
              comparison.
Experiment 2  Power of the likelihood-ratio test of beta_z, exponential
              baseline, n = 200, against the asymptotic power from the
              noncentral chi-square distribution with noncentrality
              n Var(z) beta_z^2 (the Fisher information of an exponential
              baseline is 1 per observation on the scale of eta).
Experiment 3  The score test of sigma = 0, the default of getsigmatest:
              Its level with no random effect, and its power as sigma grows
              (20 groups of 10, exponential baseline). The likelihood-ratio
              alternative, conservative with few groups, is shown for
              comparison.
Experiment 4  The confidence interval of sigma, confint(sigma=True), on the
              fits of experiment 3: Its coverage, and how often it warns
              that sigma is poorly determined (f < 4) or that the data do not
              support the random effect.

Rates are shown with 95% Clopper-Pearson intervals, which should contain the
nominal level. A one-phase fit has a concave log-likelihood, so it is run
with one start (restarts=1) to save time; the multi-phase fits use the
default 10. The data are simulated with numpy, independently of PhaseDist.

Run from the repository root:  python validation/fitreg_tests.py
Runtime: about an hour on 2 cores, less with more (WORKERS below). The tables
are printed and saved, with the figures and the raw results (CSV), in
validation/output/. To rebuild the tables and figures from the saved results
without simulating again:  python validation/fitreg_tests.py --report

Bladt, M. (2022). Phase-type distributions for claim severity regression
modeling. ASTIN Bulletin, 52(2), 417-448.
"""

import contextlib
import io
import os
import sys
import textwrap
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "src"))
from phasedist import fitreg

OUTPUT = os.path.join(here, "output")
NAME = "fitreg_tests"
SEED = 2027
WORKERS = os.cpu_count()
BETA_X = 0.8
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2", "#c98500"]
REPORT = []


def report(line=""):
    """Prints a line of the report and keeps it for the saved copy."""
    print(line)
    REPORT.append(line)


def simulate(rng, n, beta_z=0.0, baseline="exponential", groups=None, sigma=0.0, censoring_mean=None, factor=False):
    """
    One data set: y = U exp(-eta), eta = 0.8 x + beta_z z (+ b_g). With
    factor=True a three-level factor w without effect is added. With
    censoring_mean, y is right-censored at independent exponential times of
    that mean, coded as fitreg reads it: response missing, the time in 'lower'.
    """
    x = rng.normal(size=n)
    z = rng.permutation(np.resize([0.0, 1.0], n))           # balanced binary covariate
    eta = BETA_X * x + beta_z * z
    data = {"x": x, "z": z}
    if factor:
        data["w"] = pd.Categorical(rng.permutation(np.resize(["a", "b", "c"], n)))
    if groups:
        group = np.repeat(np.arange(groups), n // groups)
        eta = eta + rng.normal(0.0, sigma, groups)[group]
        data["g"] = pd.Categorical(group)
    if baseline == "exponential":
        u = rng.exponential(1.0, n)
    else:   # Coxian: rate 2, then on with probability 0.5 to rate 0.5
        u = rng.exponential(1.0 / 2.0, n) + np.where(rng.random(n) < 0.5, rng.exponential(1.0 / 0.5, n), 0.0)
    y = u * np.exp(-eta)
    if censoring_mean is None:
        data["y"] = y
    else:
        c = rng.exponential(censoring_mean, n)
        data["y"] = np.where(y <= c, y, np.nan)
        data["lower"] = np.where(y <= c, np.nan, c)
        data["upper"] = np.nan
    return pd.DataFrame(data)


def quiet(function, *arguments, **keywords):
    """Calls a function without printing; returns its result and the warnings raised, as text."""
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()) as printed:
        warnings.simplefilter("always")
        result = function(*arguments, **keywords)
    return result, [str(w.message) for w in caught], printed.getvalue()


def results(experiment, function, jobs):
    """
    Runs function on every job in parallel and returns the results as a
    DataFrame, saved as CSV. With --report the saved results are read instead.
    """
    path = os.path.join(OUTPUT, f"{NAME}_{experiment}.csv")
    if "--report" in sys.argv:
        return pd.read_csv(path)
    began = time.time()
    with Pool(WORKERS) as pool:
        rows = pool.map(function, jobs, chunksize=1)
    print(f"  ({len(jobs)} data sets in {time.time() - began:.0f} s)")
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def warning_summary(frame):
    """Reports the warnings raised in an experiment, by message."""
    messages = [m for text in frame["warnings"].fillna("") for m in text.split(" | ") if m]
    if not messages:
        report("Warnings: none")
        return
    report(f"Warnings ({len(messages)}):")
    for message in sorted(set(messages)):
        report(textwrap.fill(message, 100, initial_indent=f"  {messages.count(message):5d} x  ",
                             subsequent_indent=" " * 11))


def rng_for(*keys):
    """A random number generator of its own for every experiment, setting and replicate."""
    return np.random.default_rng([SEED, *keys])


def rate(hits):
    """A proportion with its 95% Clopper-Pearson interval, as text."""
    hits = np.asarray(hits, dtype=bool)
    k, n = int(hits.sum()), hits.size
    low = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    high = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return f"{k / n:6.3f} [{low:.3f},{high:.3f}]"


def save(figure, filename):
    """Saves a figure in the output folder."""
    figure.tight_layout()
    figure.savefig(os.path.join(OUTPUT, f"{NAME}_{filename}"), dpi=130)
    plt.close(figure)


# ----------------------------------------------------------------------
# Experiment 1: Level of the likelihood-ratio tests, coverage of the intervals
# ----------------------------------------------------------------------

# name: (data settings, fit settings, data sets)
SETTINGS = {
    "exponential, n = 50": (dict(n=50), dict(nphases=1, restarts=1), 400),
    "exponential, n = 200": (dict(n=200, factor=True), dict(nphases=1, restarts=1), 400),
    "Coxian, n = 100": (dict(n=100, baseline="coxian"), dict(), 100),
    "30% censored, n = 200": (dict(n=200, censoring_mean=3.0), dict(nphases=1, restarts=1, censoring=("lower", "upper")), 300),
    "random intercept": (dict(n=200, groups=20, sigma=0.5), dict(nphases=1, restarts=1), 300),
}


def level_job(job):
    label, replicate = job
    simulation, fitting, _ = SETTINGS[label]
    data = simulate(rng_for(1, list(SETTINGS).index(label), replicate), **simulation)
    formula = "y ~ x + z" + (" + w" if "factor" in simulation else "") + (" + (1|g)" if "groups" in simulation else "")
    model, warned, _ = quiet(fitreg, formula=formula, data=data, seed=replicate, **fitting)
    test, more, _ = quiet(model.coeftest, coefficients="z", show=False)
    profile, more2, _ = quiet(model.confint, coefficients="x", show=False)
    wald = model.confint(method="wald", coefficients="x", show=False)["x"]
    row = dict(setting=label, p_z=test["z"]["pvalue"],
               profile_lower=profile["x"]["lower"], profile_upper=profile["x"]["upper"],
               wald_lower=wald["lower"], wald_upper=wald["upper"], warnings=" | ".join(warned + more + more2))
    if "factor" in simulation:
        effect, more3, _ = quiet(model.getlrtest, ["w"])
        row.update(p_w=effect["w"]["pvalue"], df_w=effect["w"]["df"])
        row["warnings"] = " | ".join(warned + more + more2 + more3)
    return row


def experiment1():
    jobs = [(label, r) for label, (_, _, count) in SETTINGS.items() for r in range(count)]
    report("EXPERIMENT 1: Likelihood-ratio tests (coeftest, getlrtest) and profile-likelihood intervals (confint)")
    frame = results("experiment1", level_job, jobs)
    report()
    report("Rejection rate of the likelihood-ratio test of beta_z = 0 (true): should match the level")
    report(f"{'setting':24s} {'n data':>6s} {'level 1%':>21s} {'level 5%':>21s} {'level 10%':>21s} {'uniform p':>10s}")
    pvalues = {}
    for label in SETTINGS:
        p = frame.loc[frame["setting"] == label, "p_z"].to_numpy()
        pvalues[label] = p
        report(f"{label:24s} {p.size:6d} {rate(p < 0.01):>21s} {rate(p < 0.05):>21s} {rate(p < 0.10):>21s} "
               f"{stats.kstest(p, 'uniform').pvalue:10.2f}")
    p = frame["p_w"].dropna().to_numpy()
    df = int(frame["df_w"].dropna().iloc[0])
    report(f"{'factor w (' + str(df) + ' df), n = 200':24s} {p.size:6d} {rate(p < 0.01):>21s} {rate(p < 0.05):>21s} "
           f"{rate(p < 0.10):>21s} {stats.kstest(p, 'uniform').pvalue:10.2f}")
    pvalues[f"factor w ({df} df), n = 200"] = p
    report("('uniform p': Kolmogorov-Smirnov test of the p-values against the uniform distribution)")
    report()
    report("Coverage of the 95% intervals of beta_x (true value 0.8): should be 0.95")
    report(f"{'setting':24s} {'profile likelihood (default)':>30s} {'Wald':>24s}")
    coverage = {}
    for label in SETTINGS:
        part = frame[frame["setting"] == label]
        profile = (part["profile_lower"] <= BETA_X) & (BETA_X <= part["profile_upper"])
        wald = (part["wald_lower"] <= BETA_X) & (BETA_X <= part["wald_upper"])
        coverage[label] = (profile, wald)
        report(f"{label:24s} {rate(profile):>30s} {rate(wald):>24s}")
    warning_summary(frame)
    report()

    figure, axes = plt.subplots(2, 3, figsize=(9.0, 5.0))
    for axis, (label, p) in zip(axes.ravel(), pvalues.items()):
        axis.hist(p, bins=20, range=(0, 1), density=True, color="#d9d9d6", edgecolor="white")
        axis.axhline(1.0, color=COLORS[0], lw=1.6)
        axis.set_title(label, fontsize=9)
        axis.set_xlabel("p-value", fontsize=8)
        axis.set_yticks([])
    figure.suptitle("Experiment 1: p-values of likelihood-ratio tests of true null hypotheses (uniform: the line)", fontsize=10)
    save(figure, "experiment1_pvalues.png")

    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    for row, (label, (profile, wald)) in enumerate(coverage.items()):
        for offset, hits, color, name in ((-0.12, profile, COLORS[0], "profile likelihood"), (0.12, wald, COLORS[1], "Wald")):
            k, n = int(hits.sum()), hits.size
            low = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
            high = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
            axis.errorbar(k / n, row + offset, xerr=[[k / n - low], [high - k / n]], fmt="o", color=color, ms=5,
                          capsize=3, label=name if row == 0 else None)
    axis.axvline(0.95, color="#6b6b6b", lw=1.0, ls="--")
    axis.set_yticks(range(len(coverage)))
    axis.set_yticklabels(list(coverage), fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Coverage of the 95% interval of beta_x (with 95% Clopper-Pearson interval)")
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    axis.set_title("Experiment 1: Coverage of the confidence intervals", fontsize=10)
    save(figure, "experiment1_coverage.png")


# ----------------------------------------------------------------------
# Experiment 2: Power of the likelihood-ratio test
# ----------------------------------------------------------------------

EFFECTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
POWER_N, POWER_REPLICATES = 200, 200


def power_job(job):
    beta_z, replicate = job
    data = simulate(rng_for(2, EFFECTS.index(beta_z), replicate), POWER_N, beta_z=beta_z)
    model, warned, _ = quiet(fitreg, formula="y ~ x + z", data=data, nphases=1, restarts=1, seed=replicate)
    test, more, _ = quiet(model.coeftest, coefficients="z", show=False)
    return dict(beta_z=beta_z, p_z=test["z"]["pvalue"], warnings=" | ".join(warned + more))


def experiment2():
    report(f"EXPERIMENT 2: Power of the likelihood-ratio test of beta_z, exponential baseline, n = {POWER_N}, "
           f"{POWER_REPLICATES} data sets per effect")
    frame = results("experiment2", power_job, [(b, r) for b in EFFECTS for r in range(POWER_REPLICATES)])
    critical = stats.chi2.ppf(0.95, 1)
    report(f"{'beta_z':>7s} {'simulated power':>22s} {'asymptotic':>11s}")
    simulated, asymptotic = [], []
    for beta_z in EFFECTS:
        p = frame.loc[frame["beta_z"] == beta_z, "p_z"].to_numpy()
        noncentrality = POWER_N * 0.25 * beta_z ** 2           # n Var(z) beta_z^2, Var(z) = 1/4
        theory = stats.ncx2.sf(critical, 1, noncentrality) if beta_z > 0 else 0.05
        simulated.append(np.mean(p < 0.05))
        asymptotic.append(theory)
        report(f"{beta_z:7.2f} {rate(p < 0.05):>22s} {theory:11.3f}")
    warning_summary(frame)
    report()

    figure, axis = plt.subplots(figsize=(5.6, 4.0))
    grid = np.linspace(0.0, 0.5, 100)
    axis.plot(grid, [stats.ncx2.sf(critical, 1, POWER_N * 0.25 * b ** 2) if b > 0 else 0.05 for b in grid],
              color="#6b6b6b", lw=1.4, label="asymptotic power")
    for beta_z, value in zip(EFFECTS, simulated):
        k = int(round(value * POWER_REPLICATES))
        low = 0.0 if k == 0 else stats.beta.ppf(0.025, k, POWER_REPLICATES - k + 1)
        high = 1.0 if k == POWER_REPLICATES else stats.beta.ppf(0.975, k + 1, POWER_REPLICATES - k)
        axis.errorbar(beta_z, value, yerr=[[value - low], [high - value]], fmt="o", color=COLORS[0], ms=5, capsize=3,
                      label="simulated (95% CI)" if beta_z == 0 else None)
    axis.set_xlabel("beta_z")
    axis.set_ylabel("Rejection rate at the 5% level")
    axis.set_title("Experiment 2: Power of the likelihood-ratio test, n = 200", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    save(figure, "experiment2_power.png")


# ----------------------------------------------------------------------
# Experiments 3 and 4: The test of sigma = 0, and the interval of sigma
# ----------------------------------------------------------------------

SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.5]
SIGMA_REPLICATES = {0.0: 400, 0.1: 200, 0.2: 200, 0.3: 200, 0.5: 200}


def sigma_job(job):
    sigma, replicate = job
    data = simulate(rng_for(3, SIGMAS.index(sigma), replicate), 200, groups=20, sigma=sigma)
    model, warned, _ = quiet(fitreg, formula="y ~ x + z + (1|g)", data=data, nphases=1, restarts=1, seed=replicate)
    interval, _, printed = quiet(model.confint, coefficients=[], sigma=True, show=False)
    row = interval["sd(g)"]
    return dict(sigma=sigma, p_score=model.getsigmatest()["pvalue"], p_lrt=model.getsigmatest("lrt")["pvalue"],
                estimate=row["estimate"], lower=row["lower"], upper=row["upper"], f=row["f"],
                poorly_determined="poorly determined" in printed, no_support="do not support" in printed,
                warnings=" | ".join(warned))


def experiment3and4():
    jobs = [(s, r) for s in SIGMAS for r in range(SIGMA_REPLICATES[s])]
    report("EXPERIMENT 3: Test of sigma = 0 (getsigmatest), 20 groups of 10, exponential baseline")
    frame = results("experiment3_4", sigma_job, jobs)
    report("Rejection rate at the 5% level: the level where sigma = 0, the power elsewhere")
    report(f"{'sigma':>6s} {'n data':>6s} {'score test (default)':>22s} {'likelihood ratio':>22s}")
    for sigma in SIGMAS:
        part = frame[frame["sigma"] == sigma]
        report(f"{sigma:6.1f} {len(part):6d} {rate(part['p_score'] < 0.05):>22s} {rate(part['p_lrt'] < 0.05):>22s}")
    p = frame.loc[frame["sigma"] == 0.0, "p_score"].to_numpy()
    report(f"Score test p-values under sigma = 0 against the uniform: Kolmogorov-Smirnov p = {stats.kstest(p, 'uniform').pvalue:.2f}")
    warning_summary(frame)
    report()

    report("EXPERIMENT 4: The 95% interval of sigma (confint(sigma=True)) on the fits of experiment 3")
    report("Coverage of all fits counts a fit without an interval (no support) as a miss. 'Warned': the")
    report("share of fits with f < 4, where confint warns that the interval might be misleading.")
    report(f"{'sigma':>6s} {'median f':>9s} {'coverage, all fits':>21s} {'coverage, intervals':>21s} "
           f"{'coverage, f >= 4':>21s} {'warned':>7s} {'no support':>11s}")
    for sigma in SIGMAS[1:]:
        part = frame[frame["sigma"] == sigma]
        covered = (part["lower"] <= sigma) & (sigma <= part["upper"])
        exists = part["lower"].notna()
        unwarned = exists & ~part["poorly_determined"].astype(bool)
        report(f"{sigma:6.1f} {np.nanmedian(part['f']):9.1f} {rate(covered):>21s} {rate(covered[exists]):>21s} "
               f"{rate(covered[unwarned]) if unwarned.any() else '':>21s} "
               f"{part['poorly_determined'].mean():7.2f} {part['no_support'].mean():11.2f}")
    part = frame[frame["sigma"] == 0.0]
    report(f"With sigma = 0 the data support no random effect in {part['no_support'].mean():.0%} of the fits, "
           f"and f < 4 in a further {part['poorly_determined'].mean():.0%}.")
    flagged = {s: (frame.loc[frame["sigma"] == s, "poorly_determined"] | frame.loc[frame["sigma"] == s, "no_support"]).mean()
               for s in SIGMAS[1:]}
    report("Fits that warn (f < 4) or find no support: "
           + ", ".join(f"{share:.0%} at sigma = {s}" for s, share in flagged.items()) + ".")
    report("Where sigma is small, the few fits without a warning are those that overestimated sigma (f grows")
    report("with the estimate), so their intervals tend to lie above the truth: The warning screens out")
    report("poorly determined fits, but is no guarantee for the rest.")
    report()

    figure, axis = plt.subplots(figsize=(5.6, 4.0))
    for color, (column, name) in zip(COLORS, (("p_score", "score test (default)"), ("p_lrt", "likelihood ratio"))):
        rates = [np.mean(frame.loc[frame["sigma"] == s, column] < 0.05) for s in SIGMAS]
        axis.plot(SIGMAS, rates, "o-", color=color, lw=1.8, ms=6, label=name)
    axis.axhline(0.05, color="#6b6b6b", lw=1.0, ls="--", label="5% level")
    axis.set_xlabel("sigma (true)")
    axis.set_ylabel("Rejection rate of sigma = 0")
    axis.set_title("Experiment 3: Test of the random effect, 20 groups of 10", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    save(figure, "experiment3_sigmatest.png")


if __name__ == "__main__":
    os.makedirs(OUTPUT, exist_ok=True)
    experiment1()
    experiment2()
    experiment3and4()
    with open(os.path.join(OUTPUT, f"{NAME}.txt"), "w") as stream:
        stream.write("\n".join(REPORT) + "\n")
