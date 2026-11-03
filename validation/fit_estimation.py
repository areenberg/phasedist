"""
Simulation validation of fit: Are the estimated distribution parameters right?

fit estimates a phase-type (PH) distribution from data by maximum likelihood
with the EM algorithm. Each experiment simulates data from a known
distribution, fits it, and compares the estimates with the truth or with an
exact result:

Experiment 1  Exponential (one phase). The estimate has a closed form,
              rate = 1 / mean, and an exact sampling distribution,
              2 n rate / estimate ~ chi-square(2n), so the fit is checked
              against both.
Experiment 2  Right-censored exponential. The estimate is again closed form,
              rate = events / total time at risk.
Experiment 3  Geometric (one discrete phase). The estimated probability of
              staying is 1 - 1 / mean, and its expectation is known exactly.
Experiment 4  Two-phase distributions, hyperexponential and generalised
              Erlang, at n = 100, 300 and 1000. The estimates must close in on
              the truth as n grows, with a root-mean-square error falling like
              1/sqrt(n); the fitted log-likelihood must not lie below that of
              the true parameters (or the fit stopped at a local maximum, which
              fit's single random start can), and the fitted mean must equal
              the sample mean (a property of the EM fixed point).
Experiment 5  The same for a discrete two-phase distribution (the sum of two
              geometric times).
Experiment 6  Censoring: The hyperexponential of experiment 4 at n = 1000,
              right-censored (independent censoring times), left-censored (a
              detection limit) and interval-censored (inspection every 0.2,
              the mean of the fast branch; much coarser, and that branch falls
              almost entirely in the first interval, leaving its rate barely
              identified).

Parameters of a PH distribution are only identified up to a relabelling of
the phases, so they are compared in a fixed order (by rate). The data are
simulated with numpy from the raw parameters, independently of PhaseDist.

Run from the repository root:  python validation/fit_estimation.py
Runtime: about 25 minutes on 2 cores, less with more (WORKERS below). The
tables are printed and saved, with the figures and the raw results (CSV), in
validation/output/. To rebuild the tables and figures from the saved results
without simulating again:  python validation/fit_estimation.py --report
"""

import contextlib
import io
import os
import sys
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
from phasedist import fit

OUTPUT = os.path.join(here, "output")
NAME = "fit_estimation"
SEED = 2026
WORKERS = os.cpu_count()
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
REPORT = []


def report(line=""):
    """Prints a line of the report and keeps it for the saved copy."""
    print(line)
    REPORT.append(line)


def quietfit(**arguments):
    """Runs fit without printing, and counts the warnings it raises."""
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        model = fit(**arguments)
    return model, len(caught)


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


def rng_for(*keys):
    """A random number generator of its own for every experiment, setting and replicate."""
    return np.random.default_rng([SEED, *keys])


def meaninterval(values):
    """Mean and the half-width of its 95% t interval."""
    values = np.asarray(values, dtype=float)
    return values.mean(), stats.t.ppf(0.975, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size)


def save(figure, filename):
    """Saves a figure in the output folder."""
    figure.tight_layout()
    figure.savefig(os.path.join(OUTPUT, f"{NAME}_{filename}"), dpi=130)
    plt.close(figure)


# ----------------------------------------------------------------------
# Experiment 1: Exponential, closed form and exact sampling distribution
# ----------------------------------------------------------------------

EXP_RATE, EXP_N, EXP_REPLICATES = 0.5, 20, 2000


def exponential_job(replicate):
    rng = rng_for(1, replicate)
    y = rng.exponential(1.0 / EXP_RATE, EXP_N)
    model, warned = quietfit(obs=y, nphases=1, seed=replicate)
    return dict(estimate=-model.getphasegen()[0, 0], closedform=1.0 / y.mean(), warnings=warned)


def experiment1():
    report(f"EXPERIMENT 1: Exponential, rate {EXP_RATE}, n = {EXP_N}, {EXP_REPLICATES} data sets")
    frame = results("experiment1", exponential_job, range(EXP_REPLICATES))
    gap = np.max(np.abs(frame["estimate"] / frame["closedform"] - 1.0))
    # 2 n rate / estimate ~ chi-square(2n): Pearson test in 20 equally likely bins
    pivot = 2.0 * EXP_N * EXP_RATE / frame["estimate"]
    counts = np.bincount(np.minimum((stats.chi2.cdf(pivot, 2 * EXP_N) * 20).astype(int), 19), minlength=20)
    pearson = np.sum((counts - len(pivot) / 20) ** 2 / (len(pivot) / 20))
    mean, half = meaninterval(frame["estimate"])
    expected = EXP_N * EXP_RATE / (EXP_N - 1)
    report(f"  Largest relative difference from the closed form 1/mean:   {gap:.1e}")
    report(f"  Mean estimate {mean:.4f} +/- {half:.4f} (95% CI); exact expectation n rate/(n-1) = {expected:.4f}")
    report(f"  2 n rate / estimate against chi-square({2 * EXP_N}): Pearson statistic {pearson:.1f} on 19 df, "
           f"p = {stats.chi2.sf(pearson, 19):.2f}")
    report(f"  Warnings: {frame['warnings'].sum()}")
    report()

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    grid = np.linspace(pivot.min(), pivot.max(), 300)
    axes[0].hist(pivot, bins=40, density=True, color="#d9d9d6", edgecolor="white")
    axes[0].plot(grid, stats.chi2.pdf(grid, 2 * EXP_N), color=COLORS[0], lw=2.0, label=f"chi-square({2 * EXP_N})")
    axes[0].set_xlabel("2 n rate / estimated rate")
    axes[0].set_yticks([])
    axes[0].legend(frameon=False, fontsize=8)
    theoretical = stats.chi2.ppf((np.arange(1, len(pivot) + 1) - 0.5) / len(pivot), 2 * EXP_N)
    axes[1].plot(theoretical, np.sort(pivot), ".", color=COLORS[0], ms=3)
    axes[1].plot(theoretical[[0, -1]], theoretical[[0, -1]], color="#6b6b6b", lw=1.0)
    axes[1].set_xlabel(f"chi-square({2 * EXP_N}) quantiles")
    axes[1].set_ylabel("Simulated quantiles")
    figure.suptitle("Experiment 1: The estimated exponential rate has its exact sampling distribution", fontsize=10)
    save(figure, "experiment1_exponential.png")


# ----------------------------------------------------------------------
# Experiment 2: Right-censored exponential, closed form
# ----------------------------------------------------------------------

CENSOR_RATE, CENSOR_N, CENSOR_REPLICATES = 0.25, 50, 200


def censored_exponential_job(replicate):
    rng = rng_for(2, replicate)
    y = rng.exponential(1.0 / EXP_RATE, CENSOR_N)
    c = rng.exponential(1.0 / CENSOR_RATE, CENSOR_N)        # independent censoring times
    observed = np.minimum(y, c)
    # fit marks a right-censored observation, Y > c, as [nan, c]
    censoring = np.column_stack([np.full(CENSOR_N, np.nan), np.where(y > c, c, np.nan)])
    model, warned = quietfit(obs=observed, censoring=censoring, nphases=1, seed=replicate)
    return dict(estimate=-model.getphasegen()[0, 0], closedform=np.sum(y <= c) / observed.sum(),
                censored=np.mean(y > c), warnings=warned)


def experiment2():
    report(f"EXPERIMENT 2: Right-censored exponential, rate {EXP_RATE}, n = {CENSOR_N}, {CENSOR_REPLICATES} data sets")
    frame = results("experiment2", censored_exponential_job, range(CENSOR_REPLICATES))
    gap = np.max(np.abs(frame["estimate"] / frame["closedform"] - 1.0))
    report(f"  Censored share {frame['censored'].mean():.2f}")
    report(f"  Largest relative difference from the closed form events/time at risk: {gap:.1e}")
    report(f"  Warnings: {frame['warnings'].sum()}")
    report()


# ----------------------------------------------------------------------
# Experiment 3: Geometric, closed form and exact expectation
# ----------------------------------------------------------------------

STAY, GEO_N, GEO_REPLICATES = 0.7, 20, 2000


def geometric_job(replicate):
    rng = rng_for(3, replicate)
    y = rng.geometric(1.0 - STAY, GEO_N)                  # 1, 2, ...: time to leave
    model, warned = quietfit(obs=y, nphases=1, discrete=True, seed=replicate)
    return dict(estimate=model.getphasegen()[0, 0], closedform=1.0 - 1.0 / y.mean(), warnings=warned)


def experiment3():
    report(f"EXPERIMENT 3: Geometric, probability of staying {STAY}, n = {GEO_N}, {GEO_REPLICATES} data sets")
    frame = results("experiment3", geometric_job, range(GEO_REPLICATES))
    gap = np.max(np.abs(frame["estimate"] - frame["closedform"]))
    # the sum S of n geometric times is n plus a negative binomial number of
    # failures, so E[1 - n/S] is a sum over its distribution
    failures = np.arange(0, 2000)
    expected = np.sum(stats.nbinom.pmf(failures, GEO_N, 1.0 - STAY) * (1.0 - GEO_N / (GEO_N + failures)))
    mean, half = meaninterval(frame["estimate"])
    report(f"  Largest difference from the closed form 1 - 1/mean:   {gap:.1e}")
    report(f"  Mean estimate {mean:.4f} +/- {half:.4f} (95% CI); exact expectation {expected:.4f}")
    report(f"  Warnings: {frame['warnings'].sum()}")
    report()


# ----------------------------------------------------------------------
# Experiments 4 and 5: Two-phase distributions, consistency
# ----------------------------------------------------------------------

SIZES = [100, 300, 1000]
REPLICATES = 200

# Hyperexponential: with probability 0.3 rate 5, otherwise rate 0.5.
HYPER = dict(p_fast=0.3, rate_fast=5.0, rate_slow=0.5)
# Generalised Erlang: rate 1, then rate 3 (the order is not identified).
ERLANG = dict(rate_slow=1.0, rate_fast=3.0)
# Discrete generalised Erlang: probability 0.5 of staying in phase 1, 0.8 in phase 2.
DISCRETE = dict(stay_low=0.5, stay_high=0.8)


def simulate_hyper(rng, n):
    fast = rng.random(n) < HYPER["p_fast"]
    return np.where(fast, rng.exponential(1.0 / HYPER["rate_fast"], n), rng.exponential(1.0 / HYPER["rate_slow"], n))


def loglik_hyper(y, censoring=None, p=HYPER["p_fast"], fast=HYPER["rate_fast"], slow=HYPER["rate_slow"]):
    """Log-likelihood of the hyperexponential, with censored rows read as fit does."""
    survival = lambda x: p * np.exp(-fast * x) + (1 - p) * np.exp(-slow * x)
    density = p * fast * np.exp(-fast * y) + (1 - p) * slow * np.exp(-slow * y)
    if censoring is None:
        return np.sum(np.log(density))
    left, right = censoring[:, 0], censoring[:, 1]
    contribution = np.where(np.isnan(left) & np.isnan(right), density,                    # exact
                   np.where(np.isnan(left), survival(np.nan_to_num(right)),               # Y > right
                   np.where(np.isnan(right), 1 - survival(np.nan_to_num(left)),           # Y <= left
                            survival(np.nan_to_num(left)) - survival(np.nan_to_num(right)))))   # left < Y <= right
    return np.sum(np.log(contribution))


def hyper_estimates(model):
    """(p_fast, rate_fast, rate_slow) of a fitted hyperexponential."""
    rates = -np.diag(model.getphasegen())
    fast, slow = np.argmax(rates), np.argmin(rates)
    return dict(p_fast=model.getinitdist()[fast], rate_fast=rates[fast], rate_slow=rates[slow])


def hyper_job(job):
    n, replicate = job
    rng = rng_for(4, 1, n, replicate)
    y = simulate_hyper(rng, n)
    model, warned = quietfit(obs=y, nphases=2, dtype="hyperexp", seed=replicate)
    return dict(n=n, **hyper_estimates(model), loglik_fit=model.getloglik(), loglik_true=loglik_hyper(y),
                mean_fit=model.getmean(), mean_sample=y.mean(), warnings=warned)


def erlang_job(job):
    n, replicate = job
    rng = rng_for(4, 2, n, replicate)
    a, b = ERLANG["rate_slow"], ERLANG["rate_fast"]
    y = rng.exponential(1.0 / a, n) + rng.exponential(1.0 / b, n)
    model, warned = quietfit(obs=y, nphases=2, dtype="generlang", seed=replicate)
    rates = np.sort(-np.diag(model.getphasegen()))
    true_loglik = np.sum(np.log(a * b / (b - a) * (np.exp(-a * y) - np.exp(-b * y))))
    return dict(n=n, rate_slow=rates[0], rate_fast=rates[1], loglik_fit=model.getloglik(), loglik_true=true_loglik,
                mean_fit=model.getmean(), mean_sample=y.mean(), warnings=warned)


def discrete_job(job):
    n, replicate = job
    rng = rng_for(5, n, replicate)
    low, high = DISCRETE["stay_low"], DISCRETE["stay_high"]
    y = rng.geometric(1.0 - low, n) + rng.geometric(1.0 - high, n)
    model, warned = quietfit(obs=y, nphases=2, dtype="generlang", discrete=True, seed=replicate)
    stays = np.sort(np.diag(model.getphasegen()))
    true_loglik = sum(np.log(two_geometric_pmf(m, low, high)) for m in y)
    return dict(n=n, stay_low=stays[0], stay_high=stays[1], loglik_fit=model.getloglik(),
                loglik_true=true_loglik, mean_fit=model.getmean(), mean_sample=y.mean(), warnings=warned)


def two_geometric_pmf(m, low, high):
    """P(G1 + G2 = m) for geometric times on 1, 2, ... with probabilities low and high of staying."""
    k = np.arange(1, m)                                    # G1 = k, G2 = m - k
    return np.sum((1 - low) * low ** (k - 1) * (1 - high) * high ** (m - k - 1))


def consistency_table(frame, truth, title):
    """Bias, SD and sqrt(n) RMSE of each parameter, by n, and the checks on every fit."""
    report(title)
    report(f"{'parameter':10s} {'truth':>7s} {'n':>6s} {'mean':>8s} {'bias +/- 95% CI':>20s} {'SD':>8s} {'sqrt(n) RMSE':>13s}")
    rmse = {}
    for name, value in truth.items():
        rmse[name] = []
        for n in SIZES:
            estimates = frame.loc[frame["n"] == n, name]
            mean, half = meaninterval(estimates)
            error = np.sqrt(np.mean((estimates - value) ** 2))
            rmse[name].append(error)
            report(f"{name:10s} {value:7.3f} {n:6d} {mean:8.4f} {mean - value:+9.4f} +/- {half:6.4f} "
                   f"{estimates.std(ddof=1):8.4f} {np.sqrt(n) * error:13.3f}")
    below = frame["loglik_fit"] - frame["loglik_true"]
    meangap = np.abs(frame["mean_fit"] / frame["mean_sample"] - 1.0)
    stuck = frame.loc[below < -1e-6, "n"].tolist()
    report(f"Fits whose log-likelihood lies below that of the truth (by more than 1e-6), i.e. that stopped at a "
           f"local maximum: {len(stuck)} of {len(frame)}" + (f", at n = {', '.join(map(str, stuck))}" if stuck else ""))
    report(f"Largest relative gap between the fitted mean and the sample mean: {meangap.max():.1e}")
    report(f"Warnings: {frame['warnings'].sum()}")
    report()
    return rmse


def experiment4and5():
    jobs = [(n, r) for n in SIZES for r in range(REPLICATES)]
    report(f"EXPERIMENTS 4 AND 5: Two-phase distributions, {REPLICATES} data sets per n")
    report("If the estimates are consistent, the bias shrinks and sqrt(n) RMSE settles down as n grows.")
    report()
    panels = {}
    frame = results("experiment4_hyperexponential", hyper_job, jobs)
    panels["hyperexponential"] = consistency_table(frame, HYPER, "Hyperexponential (p_fast: probability of the fast branch)")
    frame = results("experiment4_generalised_erlang", erlang_job, jobs)
    panels["generalised Erlang"] = consistency_table(frame, ERLANG, "Generalised Erlang (continuous)")
    frame = results("experiment5_discrete", discrete_job, jobs)
    panels["discrete generalised Erlang"] = consistency_table(frame, DISCRETE, "Discrete generalised Erlang (stay: probability of staying in a phase)")

    figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.4))
    for axis, (title, rmse) in zip(axes, panels.items()):
        for color, (name, errors) in zip(COLORS, rmse.items()):
            axis.loglog(SIZES, errors, "o-", color=color, lw=1.8, ms=6, label=name)
            axis.loglog(SIZES, errors[0] * np.sqrt(SIZES[0] / np.array(SIZES)), "--", color=color, lw=0.9)
        axis.set_xticks(SIZES)
        axis.set_xticklabels([str(n) for n in SIZES])
        axis.minorticks_off()
        axis.set_xlabel("Sample size n")
        axis.set_title(title, fontsize=10)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Root-mean-square error")
    figure.suptitle("Experiments 4 and 5: The error falls like 1/sqrt(n) (dashed lines)", fontsize=10)
    save(figure, "experiment4_5_consistency.png")


# ----------------------------------------------------------------------
# Experiment 6: Censored hyperexponential data
# ----------------------------------------------------------------------

CENSORED_N, CENSORED_REPLICATES = 1000, 100
SCHEMES = ["uncensored", "right-censored", "left-censored", "interval-censored"]


def censored_hyper_job(job):
    scheme, replicate = job
    rng = rng_for(6, SCHEMES.index(scheme), replicate)
    n = CENSORED_N
    y = simulate_hyper(rng, n)
    nan = np.full(n, np.nan)
    # fit's censoring rows: [nan, c] for Y > c, [c, nan] for Y <= c, [a, b] for a < Y <= b
    if scheme == "uncensored":
        censoring = None
    elif scheme == "right-censored":
        c = rng.exponential(4.0, n)                           # independent censoring times
        censoring = np.column_stack([nan, np.where(y > c, c, np.nan)])
    elif scheme == "left-censored":
        censoring = np.column_stack([np.where(y <= 0.1, 0.1, np.nan), nan])   # detection limit 0.1
    else:
        lower = np.floor(y / 0.2) * 0.2                        # inspected every 0.2
        censoring = np.column_stack([lower, lower + 0.2])
    model, warned = quietfit(obs=y.copy(), censoring=censoring, nphases=2, dtype="hyperexp", seed=replicate)
    share = 0.0 if censoring is None else np.mean(~np.all(np.isnan(censoring), axis=1))
    return dict(scheme=scheme, **hyper_estimates(model), loglik_fit=model.getloglik(),
                loglik_true=loglik_hyper(y, censoring), censored=share, warnings=warned)


def experiment6():
    report(f"EXPERIMENT 6: Censored hyperexponential data, n = {CENSORED_N}, {CENSORED_REPLICATES} data sets per scheme")
    report("Right: independent exponential censoring times (mean 4). Left: values below the detection limit")
    report("0.1. Interval: every value only known to an interval of width 0.2. Censoring loses information,")
    report("so the SD may grow, but the bias should stay small.")
    report()
    frame = results("experiment6_censoring", censored_hyper_job, [(s, r) for s in SCHEMES for r in range(CENSORED_REPLICATES)])
    report(f"{'scheme':18s} {'censored':>8s} " + "".join(f"{name + ' bias +/- CI':>24s}" for name in HYPER)
           + f" {'below truth':>11s} {'warnings':>9s}")
    for scheme in SCHEMES:
        part = frame[frame["scheme"] == scheme]
        cells = ""
        for name, value in HYPER.items():
            mean, half = meaninterval(part[name])
            cells += f"{mean - value:+15.4f} +/- {half:6.4f}"
        below = int(np.sum(part["loglik_fit"] - part["loglik_true"] < -1e-6))
        report(f"{scheme:18s} {part['censored'].mean():8.2f} {cells} {below:11d} {part['warnings'].sum():9d}")
    report()

    figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.4))
    for axis, (name, value) in zip(axes, HYPER.items()):
        data = [frame.loc[frame["scheme"] == s, name] for s in SCHEMES]
        axis.boxplot(data, widths=0.5, showfliers=False, medianprops=dict(color=COLORS[0], lw=1.6))
        axis.axhline(value, color=COLORS[1], lw=1.2, ls="--", label="truth")
        axis.set_xticks(range(1, len(SCHEMES) + 1))
        axis.set_xticklabels([s.replace("-censored", "") for s in SCHEMES], fontsize=8)
        axis.set_title(name, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Experiment 6: Estimates under censoring, n = 1000 (boxes: quartiles, whiskers: 1.5 IQR)", fontsize=10)
    save(figure, "experiment6_censoring.png")


if __name__ == "__main__":
    os.makedirs(OUTPUT, exist_ok=True)
    experiment1()
    experiment2()
    experiment3()
    experiment4and5()
    experiment6()
    with open(os.path.join(OUTPUT, f"{NAME}.txt"), "w") as stream:
        stream.write("\n".join(REPORT) + "\n")
