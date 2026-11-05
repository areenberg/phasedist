"""
Simulation validation of fitreg: Are the estimated regression coefficients
and their standard errors right?

fitreg fits the phase-type regression of Bladt (2022),
Y | x ~ PH(pi, exp(eta) T) with eta = x'beta, optionally with a Gaussian
random intercept, eta = x'beta + b. Equivalently y = U exp(-eta), with U from
the baseline phase-type distribution. Every experiment simulates data from
this model with two covariates, x ~ N(0, 1) and a balanced binary z, and

    beta_x = 0.8,   beta_z = -0.5,

and checks three things for each coefficient:

  * bias: the mean estimate equals the truth, within its 95% interval;
  * standard errors: the standard deviation of the estimates across the
    data sets (SD) equals the average reported standard error (SE);
  * z-statistics: (estimate - truth) / SE is standard normal (mean 0, SD 1,
    and a Pearson test of its normal probabilities in 10 equal bins).

Experiment 1  Exponential baseline at n = 50, 200 and 1000 (one phase fitted).
              The root-mean-square error must fall like 1/sqrt(n).
Experiment 2  Coxian baseline (rate 2, then on with probability 0.5 to rate
              0.5) at n = 100 and 300, fitted with fitreg's defaults (2-phase
              Coxian, 10 random starts).
Experiment 3  Right-censored responses (about 30%, independent censoring
              times), exponential baseline, n = 200.
Experiment 4  A random intercept: 20 groups of 10, sigma = 0.5, exponential
              baseline. Also the estimate of sigma, which maximum likelihood
              biases downwards with few groups (order 1/groups).

A one-phase fit has a concave log-likelihood, so it is run with one start
(restarts=1) to save time; the multi-phase fits of experiment 2 use the
default 10. The data are simulated with numpy, independently of PhaseDist.

Run from the repository root:  python validation/fitreg_estimation.py
Runtime: about 30 minutes on 2 cores, less with more (WORKERS below). The
tables are printed and saved, with the figures and the raw results (CSV), in
validation/output/. To rebuild the tables and figures from the saved results
without simulating again:  python validation/fitreg_estimation.py --report

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
NAME = "fitreg_estimation"
SEED = 2026
WORKERS = os.cpu_count()
BETA = {"x": 0.8, "z": -0.5}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
REPORT = []


def report(line=""):
    """Prints a line of the report and keeps it for the saved copy."""
    print(line)
    REPORT.append(line)


def simulate(rng, n, baseline="exponential", groups=None, sigma=0.0, censoring_mean=None):
    """
    One data set from the model: y = U exp(-eta), eta = 0.8 x - 0.5 z (+ b_g).
    With censoring_mean, y is right-censored at independent exponential times
    of that mean, coded as fitreg reads it: response missing, the censoring
    time in 'lower'.
    """
    x = rng.normal(size=n)
    z = rng.permutation(np.resize([0.0, 1.0], n))           # balanced binary covariate
    eta = BETA["x"] * x + BETA["z"] * z
    data = {"x": x, "z": z}
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


def quietfit(**arguments):
    """Runs fitreg without printing; returns the model and the warnings it raised, as one text."""
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        model = fitreg(**arguments)
    return model, " | ".join(str(w.message) for w in caught)


def estimates(model, warned, **extra):
    """The coefficients and standard errors of a fit as one row."""
    coef, stderr = model.getcoef(), model.getstderr()
    row = dict(extra, warnings=warned)
    for name in BETA:
        row[name] = coef[name]
        row[name + "_se"] = stderr[name]
    return row


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
    print(f"  ({len(jobs)} fits in {time.time() - began:.0f} s)")
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def warning_summary(frames):
    """Reports the warnings raised in the fits, by message."""
    messages = [m for frame in frames for text in frame["warnings"].fillna("") for m in text.split(" | ") if m]
    if not messages:
        report("Warnings in all fits: none")
        return
    report(f"Warnings in all fits ({len(messages)}):")
    for message in sorted(set(messages)):
        report(textwrap.fill(message, 100, initial_indent=f"  {messages.count(message):5d} x  ",
                             subsequent_indent=" " * 11))


def rng_for(*keys):
    """A random number generator of its own for every experiment, setting and replicate."""
    return np.random.default_rng([SEED, *keys])


def summary_rows(frame, label):
    """Prints the bias, SD, SE and z-statistic summary of each coefficient for one setting."""
    for name, truth in BETA.items():
        b, se = frame[name].to_numpy(), frame[name + "_se"].to_numpy()
        half = stats.t.ppf(0.975, b.size - 1) * b.std(ddof=1) / np.sqrt(b.size)
        z = (b - truth) / se
        counts = np.bincount(np.minimum((stats.norm.cdf(z) * 10).astype(int), 9), minlength=10)
        pearson = np.sum((counts - z.size / 10) ** 2 / (z.size / 10))
        report(f"{label:22s} {name:3s} {truth:6.2f} {b.mean() - truth:+8.4f} +/- {half:6.4f} {b.std(ddof=1):8.4f} "
               f"{np.mean(se):8.4f} {b.std(ddof=1) / np.mean(se):7.3f} {z.mean():+7.3f} {z.std(ddof=1):7.3f} "
               f"{stats.chi2.sf(pearson, 9):9.2f}")


def table_header():
    report(f"{'setting':22s} {'':3s} {'truth':>6s} {'bias +/- 95% CI':>19s} {'SD':>8s} {'mean SE':>8s} {'SD/SE':>7s} "
           f"{'z mean':>7s} {'z SD':>7s} {'normal p':>9s}")


def qqplot(axis, frame, title):
    """Normal QQ plot of the z-statistics of both coefficients."""
    for color, (name, truth) in zip(COLORS, BETA.items()):
        z = np.sort((frame[name] - truth) / frame[name + "_se"])
        normal = stats.norm.ppf((np.arange(1, z.size + 1) - 0.5) / z.size)
        axis.plot(normal, z, ".", color=color, ms=3, label=f"beta_{name}")
    axis.plot([-3.5, 3.5], [-3.5, 3.5], color="#6b6b6b", lw=1.0)
    axis.set_xlim(-3.5, 3.5)
    axis.set_ylim(-4.5, 4.5)
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("N(0, 1) quantiles", fontsize=8)


def save(figure, filename):
    """Saves a figure in the output folder."""
    figure.tight_layout()
    figure.savefig(os.path.join(OUTPUT, f"{NAME}_{filename}"), dpi=130)
    plt.close(figure)


# ----------------------------------------------------------------------
# The fits of the four experiments
# ----------------------------------------------------------------------

def exponential_job(job):
    n, replicate = job
    data = simulate(rng_for(1, n, replicate), n)
    model, warned = quietfit(formula="y ~ x + z", data=data, nphases=1, restarts=1, seed=replicate)
    return estimates(model, warned, n=n)


def coxian_job(job):
    n, replicate = job
    data = simulate(rng_for(2, n, replicate), n, baseline="coxian")
    model, warned = quietfit(formula="y ~ x + z", data=data, seed=replicate)     # the defaults
    return estimates(model, warned, n=n)


def censored_job(replicate):
    data = simulate(rng_for(3, replicate), 200, censoring_mean=3.0)
    model, warned = quietfit(formula="y ~ x + z", data=data, nphases=1, restarts=1, seed=replicate,
                             censoring=("lower", "upper"))
    return estimates(model, warned, censored=data["y"].isna().mean())


def random_intercept_job(replicate):
    data = simulate(rng_for(4, replicate), 200, groups=20, sigma=0.5)
    model, warned = quietfit(formula="y ~ x + z + (1|g)", data=data, nphases=1, restarts=1, seed=replicate)
    se = model.sigmastderr
    return estimates(model, warned, sigma=model.getsigma(), sigma_se=se if se is not None and np.isfinite(se) else np.nan)


SIZES = [50, 200, 1000]          # experiment 1, 400 data sets each
COX_SIZES = [100, 300]           # experiment 2, 100 data sets each
R_EXPONENTIAL, R_COXIAN, R_OTHER = 400, 100, 300


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    settings = {}

    report(f"EXPERIMENT 1: Exponential baseline, {R_EXPONENTIAL} data sets per n")
    frame = results("experiment1", exponential_job, [(n, r) for n in SIZES for r in range(R_EXPONENTIAL)])
    for n in SIZES:
        settings[f"exponential, n = {n}"] = frame[frame["n"] == n]
    report(f"EXPERIMENT 2: Coxian baseline, fitreg defaults, {R_COXIAN} data sets per n")
    frame = results("experiment2", coxian_job, [(n, r) for n in COX_SIZES for r in range(R_COXIAN)])
    for n in COX_SIZES:
        settings[f"Coxian, n = {n}"] = frame[frame["n"] == n]
    report(f"EXPERIMENT 3: Right-censored, exponential baseline, n = 200, {R_OTHER} data sets")
    frame = results("experiment3", censored_job, range(R_OTHER))
    settings["30% censored, n = 200"] = frame
    censored_share = frame["censored"].mean()
    report(f"EXPERIMENT 4: Random intercept, 20 groups of 10, sigma = 0.5, {R_OTHER} data sets")
    random = results("experiment4", random_intercept_job, range(R_OTHER))
    settings["random intercept"] = random
    report()

    report("THE COEFFICIENTS")
    report("SD: standard deviation of the estimates. SE: reported standard error; SD/SE should be near 1.")
    report("z = (estimate - truth) / SE should have mean 0 and SD 1; 'normal p' tests its normality.")
    report()
    table_header()
    for label, frame in settings.items():
        summary_rows(frame, label)
    report()
    report(f"Censored share in experiment 3: {censored_share:.2f}")
    warning_summary(settings.values())
    report()

    # sigma of the random intercept
    sigma = random["sigma"].to_numpy()
    half = stats.t.ppf(0.975, sigma.size - 1) * sigma.std(ddof=1) / np.sqrt(sigma.size)
    report("THE RANDOM-INTERCEPT STD. DEV. (experiment 4, truth 0.5)")
    report(f"  mean estimate {sigma.mean():.4f} +/- {half:.4f} (95% CI), relative bias {sigma.mean() / 0.5 - 1:+.1%}")
    report(f"  SD of the estimates {sigma.std(ddof=1):.4f}, mean reported SE {np.nanmean(random['sigma_se']):.4f}")
    report(f"  estimates at 0 (the boundary): {np.mean(sigma == 0.0):.1%}")
    report()

    # root-mean-square error against n, exponential baseline
    report("ROOT-MEAN-SQUARE ERROR, exponential baseline: sqrt(n) RMSE should settle at a constant")
    report(f"{'n':>6s} " + "".join(f"{'beta_' + name:>12s}" for name in BETA))
    rmse = {name: [] for name in BETA}
    for n in SIZES:
        frame = settings[f"exponential, n = {n}"]
        cells = ""
        for name, truth in BETA.items():
            error = np.sqrt(np.mean((frame[name] - truth) ** 2))
            rmse[name].append(error)
            cells += f"{np.sqrt(n) * error:12.3f}"
        report(f"{n:6d} {cells}")
    report()

    figure, axes = plt.subplots(2, 4, figsize=(10.0, 5.6))
    for axis, (label, frame) in zip(axes.ravel(), settings.items()):
        qqplot(axis, frame, label)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 3].hist(sigma, bins=30, color="#d9d9d6", edgecolor="white")
    axes[1, 3].axvline(0.5, color=COLORS[1], lw=1.6, label="truth")
    axes[1, 3].axvline(sigma.mean(), color=COLORS[0], lw=1.6, ls="--", label="mean")
    axes[1, 3].set_title("random intercept, estimated sigma", fontsize=9)
    axes[1, 3].set_yticks([])
    axes[1, 3].legend(frameon=False, fontsize=8)
    figure.suptitle("z = (estimate - truth) / SE against the standard normal (on the line when the SEs are right)", fontsize=10)
    save(figure, "zstatistics.png")

    figure, axis = plt.subplots(figsize=(5.6, 4.0))
    for color, (name, errors) in zip(COLORS, rmse.items()):
        axis.loglog(SIZES, errors, "o-", color=color, lw=1.8, ms=6, label=f"beta_{name}")
        axis.loglog(SIZES, errors[0] * np.sqrt(SIZES[0] / np.array(SIZES)), "--", color=color, lw=0.9)
    axis.set_xticks(SIZES)
    axis.set_xticklabels([str(n) for n in SIZES])
    axis.minorticks_off()
    axis.set_xlabel("Sample size n")
    axis.set_ylabel("Root-mean-square error")
    axis.set_title("Exponential baseline: The error falls like 1/sqrt(n) (dashed)", fontsize=9)
    axis.legend(frameon=False, fontsize=8)
    save(figure, "rmse.png")

    with open(os.path.join(OUTPUT, f"{NAME}.txt"), "w") as stream:
        stream.write("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
