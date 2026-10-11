'''
STATISTICAL VALIDATION OF THE fitreg CLASS

The unit test (tests/test_fitreg.py) checks that the class computes the maximum
likelihood estimates it claims to compute. This script checks that those
estimates behave as the theory says they should, which takes many replicates
and is too slow for the unit tests: it simulates data from a known phase-type
regression many times, fits each data set, and reports

    - the bias of every estimate, relative to its standard error;
    - the mean reported standard error against the standard deviation of the
      estimates over the replicates (their ratio should be near one);
    - the coverage of the 95% Wald intervals, with a Clopper-Pearson interval
      for the coverage. A parameter is flagged when the Clopper-Pearson
      interval at the Bonferroni-adjusted level (over the parameters of the
      study) excludes 0.95;
    - the accuracy of the fitted conditional CDF: its largest distance to the
      true conditional CDF over a grid, at a few covariate profiles.

The interval of sigma is the Wald interval for log(sigma), transformed back,
since the class computes its standard error on the log scale (design note,
Section 9).

Studies (all three use a three-phase Coxian truth with distinct rates and fit
a three-phase Coxian):
    fixed:   y ~ x*g + w. Continuous and categorical main effects and a
             continuous-by-categorical interaction, 400 observations.
    random:  y ~ x + g + (1|group). A Gaussian random intercept with
             sigma = 1, 40 groups of 10 observations.
    null:    y ~ x + g + (1|group) on data without a random intercept (sigma =
             0), 40 groups of 10 observations. Asymptotically sigma is
             estimated as exactly zero in half the data sets, and the
             likelihood-ratio statistic against the model without random
             intercept is a 50:50 mixture of chi-square(0) and chi-square(1)
             (Self and Liang, 1987); the study reports the share at the
             boundary and the rejection rate of that test at the 5% level,
             which in finite samples is typically somewhat below 5%.

The data are simulated by running the Markov jump process of the truth from
its raw rates, and the true CDF is computed with scipy's expm, so nothing is
shared with the class under test.

References:
    Self, S. G., & Liang, K.-Y. (1987). Asymptotic properties of maximum
    likelihood estimators and likelihood ratio tests under nonstandard
    conditions. Journal of the American Statistical Association, 82(398),
    605-610. https://doi.org/10.1080/01621459.1987.10478472

Usage (from the tests/validation folder):
    python validate_fitreg.py --study fixed --replicates 200 --workers 2
    python validate_fitreg.py --study all --replicates 200 --workers 8

Each study writes its replicates to <output>/fitreg_<study>.csv and its
summary to <output>/fitreg_<study>.txt. A replicate is a deterministic function
of --seed and its index, so a run can be extended with --first.
'''

import os

# one thread per worker: the work is spread over processes instead
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import contextlib
import io
import multiprocessing
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import beta as betadist
from scipy.stats import chi2, norm

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
)

from phasedist.fitreg import fitreg


# ------------------------------------------------------------------
# THE TRUTH
# ------------------------------------------------------------------

TRUERATES = np.array([3.0, 1.0, 0.4])
TRUECONTINUE = np.array([0.6, 0.5])
TRUESIGMA = 1.0
LEVELS = ["A", "B", "C"]

STUDIES = {
    "fixed": dict(
        formula="y ~ x*g + w",
        beta={"x": 0.8, "g[B]": -0.6, "g[C]": 0.7, "w": 0.4, "x:g[B]": 0.5, "x:g[C]": -0.4},
        nobs=400,
        profiles=[{"x": 0.0, "g": "A", "w": 0.0}, {"x": 1.0, "g": "B", "w": -0.5},
                  {"x": -1.0, "g": "C", "w": 0.5}],
    ),
    "random": dict(
        formula="y ~ x + g + (1|group)",
        beta={"x": 0.8, "g[B]": -0.6, "g[C]": 0.7},
        ngroups=40,
        groupsize=10,
        profiles=[{"x": 0.0, "g": "A"}, {"x": 1.0, "g": "C"}],
    ),
    "null": dict(
        formula="y ~ x + g + (1|group)",
        beta={"x": 0.8, "g[B]": -0.6, "g[C]": 0.7},
        ngroups=40,
        groupsize=10,
        profiles=[{"x": 0.0, "g": "A"}, {"x": 1.0, "g": "C"}],
    ),
}

FIT = dict(nphases=3, dtype="coxian")


def truesubgenerator():
    """Returns the sub-generator of the true Coxian, built from the raw rates."""
    p = TRUERATES.size
    T = -np.diag(TRUERATES)
    for k in range(p - 1):
        T[k, k + 1] = TRUECONTINUE[k] * TRUERATES[k]
    return T


TRUET = truesubgenerator()
TRUEPI = np.r_[1.0, np.zeros(TRUERATES.size - 1)]


def simulatecoxian(n, rng):
    """Simulates n absorption times of the true Coxian from its jump process."""
    out = np.empty(n)
    for k in range(n):
        phase, total = 0, 0.0
        while True:
            total += rng.exponential(1.0 / TRUERATES[phase])
            if phase == TRUERATES.size - 1 or rng.random() >= TRUECONTINUE[phase]:
                break
            phase += 1
        out[k] = total
    return out


def linearpredictor(beta, frame):
    """Returns x'beta for the columns of a study, built by hand."""
    x = np.asarray(frame["x"], dtype=float)
    g = np.asarray(frame["g"], dtype=object)
    eta = beta["x"] * x + beta["g[B]"] * (g == "B") + beta["g[C]"] * (g == "C")
    if "w" in beta:
        eta = eta + beta["w"] * np.asarray(frame["w"], dtype=float)
    if "x:g[B]" in beta:
        eta = eta + beta["x:g[B]"] * x * (g == "B") + beta["x:g[C]"] * x * (g == "C")
    return eta


def truecdf(y, eta):
    """
    The true conditional CDF at the points y: U = exp(eta) y is the Coxian.
    """
    u = np.exp(eta) * y
    return np.array([1.0 - TRUEPI @ expm(TRUET * v) @ np.ones(TRUERATES.size) for v in u])


def simulate(study, rng):
    """Simulates one data set of a study, returning the DataFrame."""
    setting = STUDIES[study]
    if study in ("random", "null"):
        groups = np.repeat(np.arange(setting["ngroups"]), setting["groupsize"])
        n = groups.size
    else:
        n = setting["nobs"]
    frame = pd.DataFrame({"x": rng.normal(size=n)})
    frame["g"] = pd.Categorical(rng.choice(LEVELS, size=n), categories=LEVELS)
    if study == "fixed":
        frame["w"] = rng.uniform(-1.0, 1.0, size=n)
    eta = linearpredictor(setting["beta"], frame)
    if study == "random":
        intercepts = rng.normal(scale=TRUESIGMA, size=setting["ngroups"])
        eta = eta + intercepts[groups]
    if study in ("random", "null"):
        frame["group"] = pd.Categorical(groups)
    u = simulatecoxian(n, rng)
    frame["y"] = u * np.exp(-eta)
    return frame


# ------------------------------------------------------------------
# ONE REPLICATE
# ------------------------------------------------------------------

def replicate(task):
    """
    Simulates and fits one replicate, returning a flat dictionary of the
    estimates, standard errors, interval hits and CDF errors.
    """
    study, index, seed, restarts, tolerance = task
    setting = STUDIES[study]
    rng = np.random.default_rng([seed, index])
    frame = simulate(study, rng)

    arguments = dict(formula=setting["formula"], data=frame, restarts=restarts,
                     seed=int(rng.integers(2 ** 31)), tolerance=tolerance, **FIT)

    record = dict(study=study, replicate=index)
    start = time.time()
    buffer = io.StringIO()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with contextlib.redirect_stdout(buffer):
                model = fitreg(**arguments)
    except SystemExit:
        record.update(failed=True, seconds=time.time() - start)
        return record
    record.update(failed=False, seconds=time.time() - start, converged=bool(model.converged),
                  iterations=int(model.iterations), loglik=model.getloglik(),
                  boundary=bool(getattr(model, "boundary", False)),
                  warnings=" | ".join(sorted({str(w.message)[:80] for w in caught})))

    z = norm.ppf(0.975)
    coef, stderr = model.getcoef(), model.getstderr()
    for name, truth in setting["beta"].items():
        record["est_" + name] = coef[name]
        record["se_" + name] = stderr[name]
        record["hit_" + name] = bool(abs(coef[name] - truth) <= z * stderr[name])

    if study == "null":
        # the likelihood-ratio test of sigma = 0, against the 50:50 mixture
        lrt = max(0.0, 2.0 * (model.getloglik() - model.d.fixedloglikelihood))
        record.update(sigma=model.getsigma(), lrt=lrt,
                      pvalue=0.5 * chi2.sf(lrt, 1) if lrt > 0.0 else 1.0)

    scalars = []
    if study == "random":
        scalars.append(("sigma", model.getsigma(), model.sigmastderr, TRUESIGMA))
    for name, value, se, truth in scalars:
        record["est_" + name] = value
        record["se_" + name] = se
        # the standard error is computed for the logarithm, se(log) = se / value
        if se is not None and np.isfinite(se) and value > 0:
            record["hit_" + name] = bool(abs(np.log(value) - np.log(truth)) <= z * se / value)
        else:
            record["hit_" + name] = np.nan

    # the conditional CDF at each profile, on a grid between the true 1% and
    # 99% quantiles; for the random study at the mean intercept b = 0
    for k, profile in enumerate(setting["profiles"]):
        eta = float(linearpredictor(setting["beta"], pd.DataFrame([profile]))[0])
        grid = profilegrid(eta)
        fitted = np.array([model.getcumprob(v, profile) for v in grid])
        record["cdferr_%d" % k] = float(np.max(np.abs(fitted - truecdf(grid, eta))))
    return record


def profilegrid(eta, npoints=50):
    """
    A grid between the true 1% and 99% conditional quantiles, found by
    bisection on the true CDF.
    """
    def quantile(p):
        lower, upper = 0.0, 1.0
        while truecdf(np.array([upper]), eta)[0] < p:
            upper *= 2.0
        for _ in range(60):
            middle = 0.5 * (lower + upper)
            if truecdf(np.array([middle]), eta)[0] < p:
                lower = middle
            else:
                upper = middle
        return 0.5 * (lower + upper)
    return np.linspace(quantile(0.01), quantile(0.99), npoints)


# ------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------

def clopperpearson(hits, n, level):
    """The two-sided Clopper-Pearson interval for a binomial proportion."""
    alpha = 1.0 - level
    lower = 0.0 if hits == 0 else betadist.ppf(alpha / 2.0, hits, n - hits + 1)
    upper = 1.0 if hits == n else betadist.ppf(1.0 - alpha / 2.0, hits + 1, n - hits)
    return lower, upper


def summarize(study, results):
    """Returns the summary of a study as text."""
    setting = STUDIES[study]
    truths = dict(setting["beta"])
    if study == "random":
        truths["sigma"] = TRUESIGMA

    total = len(results)
    usable = results[~results["failed"]]
    lines = ["fitreg validation, study '%s': %s" % (study, setting["formula"]),
             "Replicates: %d, failed: %d, not converged: %d, boundary: %d, median seconds per fit: %.1f"
             % (total, total - len(usable), int((~usable["converged"].astype(bool)).sum()),
                int(usable["boundary"].astype(bool).sum()), usable["seconds"].median()),
             ""]
    level = 1.0 - 0.05 / (len(truths) + (study == "null"))
    lines.append("%-8s %8s %9s %9s %8s %8s %8s %9s %17s  %s" % (
        "param", "truth", "mean", "bias/SD", "emp.SD", "mean SE", "SE/SD", "coverage",
        "CP 95% interval", "flag (Bonferroni)"))
    flagged = []
    for name, truth in truths.items():
        estimates = usable["est_" + name].astype(float).values
        errors = usable["se_" + name].astype(float).values
        hits = usable["hit_" + name].dropna().astype(bool).values
        sd = np.std(estimates, ddof=1)
        lower, upper = clopperpearson(int(hits.sum()), hits.size, 0.95)
        adjlower, adjupper = clopperpearson(int(hits.sum()), hits.size, level)
        flag = not (adjlower <= 0.95 <= adjupper)
        if flag:
            flagged.append(name)
        lines.append("%-8s %8.3f %9.4f %9.3f %8.4f %8.4f %8.3f %9.3f   [%.3f, %.3f]  %s" % (
            name, truth, np.mean(estimates), (np.mean(estimates) - truth) / sd, sd,
            np.nanmean(errors), np.nanmean(errors) / sd, hits.mean(), lower, upper,
            "FLAG" if flag else "ok"))
    lines.append("")
    for k, profile in enumerate(setting["profiles"]):
        errors = usable["cdferr_%d" % k].values
        lines.append("Conditional CDF at %s: sup error mean %.4f, median %.4f, 95th percentile %.4f"
                     % (profile, np.mean(errors), np.median(errors), np.quantile(errors, 0.95)))
    if study == "null":
        boundary = usable["boundary"].astype(bool).values
        lower, upper = clopperpearson(int(boundary.sum()), boundary.size, 0.95)
        lines.append("")
        lines.append("Share with sigma estimated as exactly zero: %.3f [%.3f, %.3f] (asymptotically 0.5)"
                     % (boundary.mean(), lower, upper))
        reject = usable["pvalue"].values < 0.05
        lower, upper = clopperpearson(int(reject.sum()), reject.size, 0.95)
        lines.append("Rejection rate of the likelihood-ratio test at 5%%: %.3f [%.3f, %.3f]" % (reject.mean(), lower, upper))
        adjlower, _ = clopperpearson(int(reject.sum()), reject.size, level)
        if adjlower > 0.05:
            flagged.append("likelihood-ratio test (anti-conservative)")
        lines.append("Median sigma where not zero: %.4f" % np.median(usable["sigma"].values[~boundary])
                     if np.any(~boundary) else "Every estimate of sigma is zero.")
    lines.append("")
    lines.append("Flagged: %s" % (", ".join(flagged) if flagged else "none"))
    return "\n".join(lines)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Statistical validation of fitreg.")
    parser.add_argument("--study", choices=list(STUDIES) + ["all"], default="fixed")
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--first", type=int, default=0, help="index of the first replicate")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output", default=".")
    args = parser.parse_args()

    studies = list(STUDIES) if args.study == "all" else [args.study]
    for study in studies:
        tasks = [(study, index, args.seed, args.restarts, args.tolerance)
                 for index in range(args.first, args.first + args.replicates)]
        path = os.path.join(args.output, "fitreg_%s.csv" % study)
        start = time.time()
        records = []
        with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
            for k, record in enumerate(pool.imap_unordered(replicate, tasks), 1):
                records.append(record)
                if k % 10 == 0 or k == len(tasks):
                    # written as it goes, so an interrupted run keeps its replicates
                    frame = pd.DataFrame(records).sort_values("replicate")
                    if args.first > 0 and os.path.isfile(path):
                        frame = pd.concat([pd.read_csv(path), frame]).drop_duplicates("replicate", keep="last")
                    frame.to_csv(path, index=False)
                    print("%s: %d of %d replicates after %.0f s" % (study, k, len(tasks), time.time() - start), flush=True)
        results = pd.read_csv(path)
        text = summarize(study, results)
        with open(os.path.join(args.output, "fitreg_%s.txt" % study), "w") as handle:
            handle.write(text + "\n")
        print(text, flush=True)


if __name__ == "__main__":
    main()
