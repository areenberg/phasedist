'''
VALIDATION OF THE p-VALUES OF fitreg AGAINST SIMULATED DEVIANCES

fitreg reports Wald p-values: coeftest for each coefficient (z test) and
getwaldtest for each fixed effect of the formula (chi-square test with one
degree of freedom per coefficient of the effect). Both rest on the asymptotic normality of
the estimates and on the standard errors of the class. This script checks them
against p-values computed in an independent way, from the deviance

    D = 2 (loglik of the full model - loglik of the model without the effect),

whose reference distribution under the null hypothesis is obtained by
simulation rather than from asymptotic theory. Nothing in the reference
distribution uses the standard errors or the covariance of the class: only its
log-likelihoods, which the unit test checks against an independent oracle.

Two kinds of study:

    Null studies (nullfixed, nullrandom, nullexp, nulllarge). Data are
    simulated many times from a truth in which the tested effects are zero.
    Every replicate is fitted with and without each tested effect. The
    deviances of all replicates form the simulated null distribution of D,
    and the p-value of replicate i is the share of the replicates whose
    deviance is at least D_i, counting i itself (the likelihood-ratio test
    with a simulated reference). The study reports, for each test,
        - the rejection rates of the Wald test and of the chi-square
          likelihood-ratio test at the 10%, 5% and 1% levels, with
          Clopper-Pearson intervals;
        - a chi-square goodness-of-fit test of the uniformity of the Wald
          p-values (ten equal bins);
        - quantiles of the Wald statistics and of the deviances against the
          chi-square quantiles;
        - the agreement between the Wald p-value and the simulated-deviance
          p-value of the same data set.

    Bootstrap study (bootstrap). For a few data sets and tested effects
    (BOOTCASES), the model without the effect is fitted, and B data sets are
    simulated from that fit (the parametric bootstrap under the null
    hypothesis) and fitted with and without the effect. The bootstrap p-value
    is the share of bootstrap deviances at least the observed one, with a
    Clopper-Pearson interval for its Monte Carlo error, and is compared with
    the Wald p-value of fitreg and the chi-square likelihood-ratio p-value.

Tests (unless stated, the truth is the three-phase Coxian of
validate_fitreg.py, fitted with a three-phase Coxian):
    nullfixed: 400 observations, y ~ x*g + w, with g, w and x:g zero in the
               truth (x has coefficient 0.8). Tests: w in y ~ x*g + w (1 df,
               coeftest), x:g in y ~ x*g + w (2 df, getwaldtest), g in
               y ~ x + g + w (2 df, getwaldtest).
    nullrandom: 40 groups of 10, y ~ x + g + (1|group), sigma = 1, with x and
               g zero in the truth. Tests: x (1 df) and g (2 df), the
               random effect kept in both models.
    nullexp:   as nullfixed without the interaction, but with an exponential
               base distribution fitted with one phase, so that the base
               distribution has no shape to estimate. Tests: w and g.
    nulllarge: the g test of nullfixed with 1600 observations, to see how
               the finite-sample behaviour of nullfixed changes with n.
    bootstrap: three data sets of 400 observations from y ~ x*g + w with
               small w and x:g effects (tests w and x:g), and replicate 1 of
               nullfixed (test g).

A local maximum in either fit can make a deviance negative. A deviance below
-NEGATIVETOL is refitted with REFITRESTARTS starts; a small negative deviance
(the convergence tolerance) is set to zero.

Usage (from the tests/validation folder):
    python validate_fitreg_pvalues.py --study nullfixed --replicates 600 --workers 2
    python validate_fitreg_pvalues.py --study nullrandom --replicates 300 --workers 2
    python validate_fitreg_pvalues.py --study bootstrap --prepare
    python validate_fitreg_pvalues.py --study bootstrap --replicates 200 --workers 2
    python validate_fitreg_pvalues.py --study all --summarize

Replicates are appended as JSON lines to <output>/pvalues_<study><tag>.jsonl,
and the summary reads every such file. A replicate is a deterministic
function of --seed and its index, so runs can be split over machines with
--first and --tag and the files put in one folder. Keep the index set of a
null study complete (no gaps): a replicate that is slow to fit is not a
random one, so dropping replicates that were cut off by a time limit biases
the study. The summary of each study is written to <output>/pvalues_<study>.txt.
'''

import os

# one thread per worker: the work is spread over processes instead
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import contextlib
import glob
import io
import json
import multiprocessing
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import beta as betadist
from scipy.stats import chi2, norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from phasedist.fitreg import fitreg
import validate_fitreg as V


# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------

NEGATIVETOL = 1e-2       # deviances below -NEGATIVETOL are refitted
REFITRESTARTS = 10       # starts of a refit
CONFIDENCE = 0.99        # family-wise level of the Clopper-Pearson intervals
LEVELS = (0.10, 0.05, 0.01)
NBINS = 10               # bins of the uniformity test
# cases of the bootstrap study: (study of the truth, index of the data set,
# tested effects). The first three are data sets 10**6, ... of the bootstrap
# truth, taken in order; the last is replicate 1 of the nullfixed study, whose
# Wald and likelihood-ratio p-values for g disagree most among the first
# replicates (0.025 against 0.14).
BOOTCASES = [("bootstrap", 10 ** 6 + k, ["w", "x:g"]) for k in range(3)] + [("nullfixed", 1, ["g"])]

ZERO = {"x": 0.0, "g[B]": 0.0, "g[C]": 0.0, "w": 0.0, "x:g[B]": 0.0, "x:g[C]": 0.0}

STUDIES = {
    "nullfixed": dict(
        code=11,
        beta=dict(ZERO, x=0.8),
        nobs=400,
        models={"full": "y ~ x*g + w", "now": "y ~ x*g", "noint": "y ~ x + g + w", "nog": "y ~ x + w"},
        # (name, effect in the larger model, larger model, smaller model)
        tests=[("w", "w", "full", "now"), ("x:g", "x:g", "full", "noint"), ("g", "g", "noint", "nog")],
    ),
    "nullrandom": dict(
        code=12,
        beta={"x": 0.0, "g[B]": 0.0, "g[C]": 0.0},
        sigma=1.0,
        ngroups=40,
        groupsize=10,
        models={"full": "y ~ x + g + (1|group)", "nox": "y ~ g + (1|group)", "nog": "y ~ x + (1|group)"},
        tests=[("x", "x", "full", "nox"), ("g", "g", "full", "nog")],
    ),
    "nullexp": dict(
        code=14,
        beta={"x": 0.8, "g[B]": 0.0, "g[C]": 0.0, "w": 0.0},
        nobs=400,
        truth="exponential",
        nphases=1,
        models={"full": "y ~ x + g + w", "now": "y ~ x + g", "nog": "y ~ x + w"},
        tests=[("w", "w", "full", "now"), ("g", "g", "full", "nog")],
    ),
    "nulllarge": dict(
        code=15,
        beta={"x": 0.8, "g[B]": 0.0, "g[C]": 0.0, "w": 0.0},
        nobs=1600,
        models={"noint": "y ~ x + g + w", "nog": "y ~ x + w"},
        tests=[("g", "g", "noint", "nog")],
    ),
    "bootstrap": dict(
        code=13,
        beta={"x": 0.8, "g[B]": -0.6, "g[C]": 0.7, "w": 0.15, "x:g[B]": 0.2, "x:g[C]": -0.1},
        nobs=400,
        models={"full": "y ~ x*g + w", "now": "y ~ x*g", "noint": "y ~ x + g + w"},
        tests=[("w", "w", "full", "now"), ("x:g", "x:g", "full", "noint")],
    ),
}


# ------------------------------------------------------------------
# SIMULATION (built from raw rates; nothing shared with the class)
# ------------------------------------------------------------------

def simulatecovariates(setting, rng):
    """Simulates the covariates (and groups) of one data set."""
    if "ngroups" in setting:
        groups = np.repeat(np.arange(setting["ngroups"]), setting["groupsize"])
        n = groups.size
    else:
        groups, n = None, setting["nobs"]
    frame = pd.DataFrame({"x": rng.normal(size=n)})
    frame["g"] = pd.Categorical(rng.choice(V.LEVELS, size=n), categories=V.LEVELS)
    if "w" in setting["beta"]:
        frame["w"] = rng.uniform(-1.0, 1.0, size=n)
    if groups is not None:
        frame["group"] = pd.Categorical(groups)
    return frame


def linearpredictor(coef, frame):
    """
    Returns x'beta for coefficients named as by fitreg.getcoef ('x', 'g[B]',
    'x:g[C]', ...), built by hand from the columns.
    """
    eta = np.zeros(len(frame))
    for name, value in coef.items():
        column = np.ones(len(frame))
        for part in name.split(":"):
            if "[" in part:
                variable, level = part[:-1].split("[")
                column = column * (np.asarray(frame[variable], dtype=object) == level)
            else:
                column = column * np.asarray(frame[part], dtype=float)
        eta = eta + value * column
    return eta


def simulateph(pi, T, n, rng):
    """
    Simulates n absorption times of PH(pi, T) by running its Markov jump
    process: exponential holding times and the embedded jump chain.
    """
    pi = np.asarray(pi, dtype=float).ravel()
    T = np.asarray(T, dtype=float)
    p = pi.size
    rates = -np.diag(T)
    jumps = np.zeros((p, p + 1))
    for i in range(p):
        row = T[i].copy()
        row[i] = 0.0
        jumps[i, :p] = np.maximum(row, 0.0) / rates[i]
        jumps[i, p] = max(-T[i].sum(), 0.0) / rates[i]
        jumps[i] /= jumps[i].sum()
    out = np.empty(n)
    for k in range(n):
        state = rng.choice(p, p=pi / pi.sum())
        total = 0.0
        while True:
            total += rng.exponential(1.0 / rates[state])
            state = rng.choice(p + 1, p=jumps[state])
            if state == p:
                break
        out[k] = total
    return out


def simulatetruth(study, rng):
    """Simulates one data set from the truth of a study."""
    setting = STUDIES[study]
    frame = simulatecovariates(setting, rng)
    eta = linearpredictor(setting["beta"], frame)
    if "sigma" in setting:
        intercepts = rng.normal(scale=setting["sigma"], size=setting["ngroups"])
        eta = eta + intercepts[np.asarray(frame["group"].cat.codes)]
    if setting.get("truth") == "exponential":
        u = rng.exponential(1.0, size=len(frame))
    else:
        u = V.simulatecoxian(len(frame), rng)
    frame["y"] = u * np.exp(-eta)
    return frame


def simulatefit(model, frame, rng):
    """
    Simulates a response for the covariates of frame from a fitted model
    without random effect: y = U exp(-x'beta), U ~ PH(pi, T) at the
    baseline.
    """
    out = frame.copy()
    eta = linearpredictor(model["coef"], frame)
    out["y"] = simulateph(model["pi"], model["T"], len(frame), rng) * np.exp(-eta)
    return out


# ------------------------------------------------------------------
# FITTING
# ------------------------------------------------------------------

TOLERANCE = 1e-5         # convergence tolerance of the fits, set by --tolerance


def settolerance(tolerance):
    """Sets the convergence tolerance of the fits (also in each worker)."""
    global TOLERANCE
    TOLERANCE = tolerance


def fit(formula, frame, restarts, seed, nphases=3):
    """Fits a model quietly, returning the fitreg object."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return fitreg(formula=formula, data=frame, nphases=nphases, dtype="coxian", restarts=restarts, seed=seed,
                      tolerance=TOLERANCE)


def fitall(setting, frame, restarts, seed, needed):
    """Fits the needed models of a study, returning a dict of fitreg objects."""
    return {name: fit(setting["models"][name], frame, restarts, seed, setting.get("nphases", 3)) for name in needed}


def evaluate(setting, frame, restarts, seed, tests):
    """
    Fits the models of the given tests and returns, for each test, the Wald
    statistic, degrees of freedom and Wald p-value of the larger model, the
    deviance and the chi-square likelihood-ratio p-value. Deviances below
    -NEGATIVETOL trigger a refit of both models with REFITRESTARTS starts.
    """
    needed = sorted({t[2] for t in tests} | {t[3] for t in tests})
    models = fitall(setting, frame, restarts, seed, needed)
    out = {}
    for name, term, large, small in tests:
        refit = False
        D = 2.0 * (models[large].getloglik() - models[small].getloglik())
        if D < -NEGATIVETOL:
            refit = True
            for m in (large, small):
                models[m] = fit(setting["models"][m], frame, REFITRESTARTS, seed + 1, setting.get("nphases", 3))
            D = 2.0 * (models[large].getloglik() - models[small].getloglik())
        W, df, pW = models[large].getwaldtest()[term]
        record = dict(W=W, df=df, pW=pW, D=D, Dclipped=max(D, 0.0), pLRT=float(chi2.sf(max(D, 0.0), df)),
                      refit=refit, degenerate=bool(models[large].d.degenerate or models[small].d.degenerate))
        if df == 1:
            # the coefficient p-value of coeftest must equal the effect test
            column = [c for c in models[large].getcoef() if c == term][0]
            record["pvalue"] = models[large].coeftest(show=False)[column]["pvalue"]
        out[name] = record
    return out, models


def nulltask(task):
    """One replicate of a null study."""
    study, index, seedbase, restarts = task
    setting = STUDIES[study]
    rng = np.random.default_rng([seedbase, setting["code"], index])
    frame = simulatetruth(study, rng)
    seed = int(rng.integers(2 ** 31))
    began = time.time()
    results, _ = evaluate(setting, frame, restarts, seed, setting["tests"])
    return dict(study=study, index=index, seconds=round(time.time() - began, 1), tests=results)


def modelparameters(model):
    """The parameters of a fitted model needed to simulate from it."""
    return dict(coef=model.getcoef(), pi=np.asarray(model.getinitdist()).ravel().tolist(),
                T=np.asarray(model.getphasegen()).tolist(), loglik=model.getloglik())


def prepare(seedbase, output):
    """
    Simulates and fits the original data sets of the bootstrap study (the
    cases of BOOTCASES), and stores the observed tests and the fitted models
    without the tested effects.
    """
    origins = []
    for case, (study, index, testnames) in enumerate(BOOTCASES):
        setting = STUDIES[study]
        tests = [t for t in setting["tests"] if t[0] in testnames]
        rng = np.random.default_rng([seedbase, setting["code"], index])
        frame = simulatetruth(study, rng)
        seed = int(rng.integers(2 ** 31))
        # the observed analysis uses the default number of starts
        results, models = evaluate(setting, frame, 10, seed, tests)
        nulls = {name: modelparameters(models[small]) for name, _, _, small in tests}
        origins.append(dict(case=case, study=study, index=index, tests=testnames,
                            frame=frame.to_dict(orient="list"), observed=results, nulls=nulls))
        print("%s %d: %s" % (study, index, ", ".join("%s: Wald p %.4f, LRT p %.4f" % (n, r["pW"], r["pLRT"])
                                                     for n, r in results.items())), flush=True)
    with open(os.path.join(output, "pvalues_bootstrap_origin.json"), "w") as handle:
        json.dump(origins, handle)


def loadorigins(output):
    """Reads the original data sets of the bootstrap study."""
    with open(os.path.join(output, "pvalues_bootstrap_origin.json")) as handle:
        origins = json.load(handle)
    for origin in origins:
        frame = pd.DataFrame(origin["frame"])
        frame["g"] = pd.Categorical(frame["g"], categories=V.LEVELS)
        origin["frame"] = frame
    return origins


ORIGINS = None


def boottask(task):
    """
    One bootstrap replicate: every case and test at bootstrap index b, or
    those listed in subset ('case:test,...'; empty for all).
    """
    b, seedbase, restarts, output, subset = task
    global ORIGINS
    if ORIGINS is None:
        ORIGINS = loadorigins(output)
    wanted = None if not subset else {tuple(item.split(":", 1)) for item in subset.split(",")}
    began = time.time()
    rows = []
    for origin in ORIGINS:
        setting = STUDIES[origin["study"]]
        for position, test in enumerate([t for t in setting["tests"] if t[0] in origin["tests"]]):
            name = test[0]
            if wanted is not None and (str(origin["case"]), name) not in wanted:
                continue
            rng = np.random.default_rng([seedbase, STUDIES["bootstrap"]["code"], origin["case"], position, b])
            frame = simulatefit(origin["nulls"][name], origin["frame"], rng)
            seed = int(rng.integers(2 ** 31))
            results, _ = evaluate(setting, frame, restarts, seed, [test])
            rows.append(dict(case=origin["case"], test=name, **results[name]))
    return dict(study="bootstrap", index=b, subset=subset, seconds=round(time.time() - began, 1), rows=rows)


# ------------------------------------------------------------------
# SUMMARIES
# ------------------------------------------------------------------

def clopperpearson(k, n, confidence):
    """Clopper-Pearson interval for a binomial proportion."""
    a = 1.0 - confidence
    lower = 0.0 if k == 0 else betadist.ppf(a / 2, k, n - k + 1)
    upper = 1.0 if k == n else betadist.ppf(1 - a / 2, k + 1, n - k)
    return lower, upper


def readrows(output, study):
    """Reads the replicates of a study, dropping duplicates (same index and bootstrap subset)."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(output, "pvalues_%s*.jsonl" % study))):
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                rows[(row["index"], row.get("subset", ""))] = row
    return [rows[i] for i in sorted(rows)]


def summarizenull(study, output):
    """Summarizes a null study."""
    rows = readrows(output, study)
    setting = STUDIES[study]
    ntests = len(setting["tests"])
    # Clopper-Pearson checks: two tests (Wald, LRT) at each level of each test
    nchecks = ntests * 2 * len(LEVELS)
    level = 1.0 - (1.0 - CONFIDENCE) / nchecks
    lines = ["Study %s: %d replicates, mean %.1f s each" % (study, len(rows), np.mean([r["seconds"] for r in rows]))]
    lines.append("Truth: %s" % setting["beta"] + (", sigma = %.1f" % setting["sigma"] if "sigma" in setting else "")
                 + ("; exponential base distribution, fitted with one phase" if setting.get("truth") == "exponential" else ""))
    lines.append("Clopper-Pearson intervals at %.4f%% each, %.0f%% for the %d intervals together (Bonferroni)."
                 % (100 * level, 100 * CONFIDENCE, nchecks))
    flags = []
    for name, term, large, small in setting["tests"]:
        recs = [r["tests"][name] for r in rows]
        M = len(recs)
        W = np.array([r["W"] for r in recs])
        D = np.array([r["Dclipped"] for r in recs])
        Draw = np.array([r["D"] for r in recs])
        pW = np.array([r["pW"] for r in recs])
        pL = np.array([r["pLRT"] for r in recs])
        df = recs[0]["df"]
        # a Wald statistic is NaN where the information was not positive
        # definite (no standard errors); such a test counts as not rejecting
        finite = np.isfinite(W)
        lines.append("")
        lines.append("Test of %s (%d df): %s against %s" % (name, df, setting["models"][large], setting["models"][small]))
        if not np.all(finite):
            lines.append("  Wald statistic unavailable (no standard errors) in %d replicates; they count as not "
                         "rejecting, and the comparisons of W below leave them out" % np.sum(~finite))
        lines.append("  refits after a negative deviance: %d; deviances below 0 after refit: %d (min %.2e); "
                     "fits with a degenerate phase: %d"
                     % (sum(r["refit"] for r in recs), int(np.sum(Draw < 0)), Draw.min(), sum(r["degenerate"] for r in recs)))
        if df == 1:
            gap = max(abs(r["pvalue"] - r["pW"]) for r in recs)
            lines.append("  coeftest vs getwaldtest p-value: largest difference %.1e" % gap)
            if gap > 1e-12:
                flags.append("%s: coeftest differs from getwaldtest" % name)
        lines.append("  rejection rate   Wald  [Clopper-Pearson]           LRT chi2 [Clopper-Pearson]")
        for alpha in LEVELS:
            parts = []
            for p, label in ((pW, "Wald"), (pL, "LRT")):
                k = int(np.sum(p < alpha))
                lo, hi = clopperpearson(k, M, level)
                parts.append("%.3f [%.3f, %.3f]" % (k / M, lo, hi))
                if not lo <= alpha <= hi:
                    flags.append("%s: %s rejection rate %.3f at level %.2f, interval [%.3f, %.3f]"
                                 % (name, label, k / M, alpha, lo, hi))
            lines.append("  at %4.2f          %s      %s" % (alpha, parts[0], parts[1]))
        # uniformity of the p-values
        for p, label in ((pW, "Wald"), (pL, "LRT chi2")):
            counts = np.histogram(p, bins=np.linspace(0, 1, NBINS + 1))[0]
            stat = np.sum((counts - M / NBINS) ** 2 / (M / NBINS))
            lines.append("  uniformity of the %s p-values: chi-square %.2f on %d df, p = %.3f; bin counts %s"
                         % (label, stat, NBINS - 1, chi2.sf(stat, NBINS - 1), counts.tolist()))
        # quantiles
        lines.append("  quantile     chi2(%d)    Wald W   deviance D" % df)
        for q in (0.5, 0.9, 0.95, 0.99):
            lines.append("  %5.2f     %8.3f  %8.3f   %8.3f" % (q, chi2.ppf(q, df), np.quantile(W[finite], q), np.quantile(D, q)))
        lines.append("  mean W - D: %.4f (sd %.4f); correlation of W and D: %.4f"
                     % (np.mean((W - D)[finite]), np.std((W - D)[finite]), np.corrcoef(W[finite], D[finite])[0, 1]))
        # simulated-deviance p-value of each replicate, leaving it out of its reference
        order = np.sort(D)
        atleast = M - np.searchsorted(order, D, side="left")   # D_j >= D_i, including i
        pD = atleast / M            # (1 + #{j != i : D_j >= D_i}) / M
        diff = pW - pD
        lines.append("  Wald p minus simulated-deviance p, same data set: mean %.4f, mean |.| %.4f, largest |.| %.4f"
                     % (np.mean(diff[finite]), np.mean(np.abs(diff[finite])), np.max(np.abs(diff[finite]))))
        small = (pD <= 0.10) & finite
        lines.append("  among the %d data sets with simulated-deviance p <= 0.10: mean |.| %.4f, median ratio Wald/simulated %.3f"
                     % (np.sum(small), np.mean(np.abs(diff[small])), np.median(pW[small] / pD[small])))
        lines.append("  the simulated deviance reference at 5%%: critical value %.3f (chi2 %.3f); "
                     "Wald test and simulated-deviance test agree on %d of %d data sets at 5%%"
                     % (np.quantile(D, 0.95), chi2.ppf(0.95, df), np.sum((pW < 0.05) == (pD < 0.05)), M))
    lines.append("")
    lines.append("Flags: %s" % ("none" if not flags else "; ".join(flags)))
    return "\n".join(lines)


def summarizebootstrap(output):
    """Summarizes the bootstrap study."""
    rows = readrows(output, "bootstrap")
    origins = loadorigins(output)
    ncases = sum(len(o["tests"]) for o in origins)
    level = 1.0 - (1.0 - CONFIDENCE) / ncases
    lines = ["Study bootstrap: %d bootstrap replicates of %d tests on %d data sets, mean %.1f s per replicate"
             % (len(rows), ncases, len(origins), np.mean([r["seconds"] for r in rows]))]
    lines.append("Truth of the bootstrap data sets: %s" % STUDIES["bootstrap"]["beta"])
    lines.append("Clopper-Pearson intervals of the bootstrap p-value at %.4f%% each, %.0f%% for the %d together."
                 % (100 * level, 100 * CONFIDENCE, ncases))
    lines.append("")
    lines.append("%-14s %-5s %3s %8s %8s %8s %8s %24s %8s %6s"
                 % ("data set", "test", "df", "W obs", "D obs", "Wald p", "LRT p", "bootstrap p [CP]", "boot-W p", "refits"))
    flags = []
    pooled = {}
    for origin in origins:
        for name in origin["tests"]:
            obs = origin["observed"][name]
            boot = [x for r in rows for x in r["rows"] if x["case"] == origin["case"] and x["test"] == name]
            B = len(boot)
            Dstar = np.array([x["Dclipped"] for x in boot])
            Wstar = np.array([x["W"] for x in boot])
            pooled.setdefault((name, obs["df"]), []).append((Dstar, Wstar))
            k = int(np.sum(Dstar >= max(obs["D"], 0.0)))
            pboot = (k + 1) / (B + 1)
            lo, hi = clopperpearson(k, B, level)
            pbootW = (np.sum(Wstar >= obs["W"]) + 1) / (B + 1)
            refits = sum(x["refit"] for x in boot)
            label = "%s %d" % (origin["study"], origin["index"])
            lines.append("%-14s %-5s %3d %8.3f %8.3f %8.4f %8.4f %8.4f [%.4f, %.4f] %8.4f %6d"
                         % (label, name, obs["df"], obs["W"], obs["D"], obs["pW"], obs["pLRT"],
                            pboot, lo, hi, pbootW, refits))
            if not lo <= obs["pW"] <= hi:
                flags.append("%s, %s: Wald p %.4f outside [%.4f, %.4f]" % (label, name, obs["pW"], lo, hi))
    lines.append("")
    lines.append("bootstrap p = (1 + #{D* >= D obs}) / (B + 1), B the number of bootstrap replicates; its interval is "
                 "the Clopper-Pearson interval for P(D* >= D obs). boot-W p is the same with the Wald statistics.")
    lines.append("")
    lines.append("Bootstrap null distributions against chi-square (per test, pooled over its data sets):")
    for (name, df), values in pooled.items():
        Dstar = np.concatenate([v[0] for v in values])
        Wstar = np.concatenate([v[1] for v in values])
        lines.append("  %s (%d df), %d values: 95%% quantile D* %.3f, W* %.3f, chi2 %.3f; share above the chi2 5%% "
                     "point: D* %.3f, W* %.3f" % (name, df, Dstar.size, np.quantile(Dstar, 0.95), np.quantile(Wstar, 0.95),
                                                  chi2.ppf(0.95, df), np.mean(Dstar > chi2.ppf(0.95, df)),
                                                  np.mean(Wstar > chi2.ppf(0.95, df))))
    lines.append("")
    lines.append("Flags: %s" % ("none" if not flags else "; ".join(flags)))
    return "\n".join(lines)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default="all", choices=["nullfixed", "nullrandom", "nullexp", "nulllarge", "bootstrap", "all"])
    parser.add_argument("--replicates", type=int, default=0)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--restarts", type=int, default=3, help="starts of each fit of a simulated data set")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tolerance", type=float, default=1e-5, help="convergence tolerance of every fit")
    parser.add_argument("--output", default=".")
    parser.add_argument("--tag", default="", help="suffix of the replicate file, to split runs over machines")
    parser.add_argument("--prepare", action="store_true", help="fit the original data sets of the bootstrap study")
    parser.add_argument("--bootcases", default="", help="bootstrap only these tests, as 'case:test,...' "
                        "(cases numbered from 0 in BOOTCASES); bootstrap replicates are independent of their index, "
                        "so the replicates of a test can be spread over runs with different --first")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    settolerance(args.tolerance)
    studies = ["nullfixed", "nullrandom", "nullexp", "nulllarge", "bootstrap"] if args.study == "all" else [args.study]
    if args.prepare:
        prepare(args.seed, args.output)
        return
    if args.summarize:
        for study in studies:
            text = summarizebootstrap(args.output) if study == "bootstrap" else summarizenull(study, args.output)
            print(text + "\n")
            with open(os.path.join(args.output, "pvalues_%s.txt" % study), "w") as handle:
                handle.write(text + "\n")
        return

    for study in studies:
        indices = range(args.first, args.first + args.replicates)
        if study == "bootstrap":
            tasks, work = [(b, args.seed, args.restarts, args.output, args.bootcases) for b in indices], boottask
        else:
            tasks, work = [(study, i, args.seed, args.restarts) for i in indices], nulltask
        path = os.path.join(args.output, "pvalues_%s%s.jsonl" % (study, args.tag))
        with open(path, "a") as handle, multiprocessing.Pool(args.workers, settolerance, (args.tolerance,)) as pool:
            for row in pool.imap_unordered(work, tasks):
                handle.write(json.dumps(row) + "\n")
                handle.flush()


if __name__ == "__main__":
    main()
