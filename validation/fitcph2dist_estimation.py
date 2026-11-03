"""
Simulation validation of fitcph2dist: Does the fitted phase-type distribution
reproduce the distribution it approximates?

fitcph2dist fits a continuous phase-type (PH) distribution to a given
continuous distribution (the target) instead of to data. The target is
discretised on a grid of `steps` cells up to its `truncation` quantile, plus
one cell for the tail, and the EM algorithm of Bladt and Nielsen (2017,
p. 681) minimises the Kullback-Leibler divergence from it. There is no
sampling noise, so every experiment compares the fit with a known answer:

Experiment 1  Targets that are themselves phase-type (a Coxian, a
              hyperexponential and a generalised Erlang distribution), fitted
              with their own structure. The fitted parameters must be the true
              ones, up to the discretisation, whose error must vanish as the
              grid is refined.
Experiment 2  Four targets that are not phase-type (log-normal, Weibull,
              gamma, normal truncated at 0), fitted with 1 to 8 phases. The
              distance to the target must shrink as phases are added, and the
              fitted mean must equal the target mean, a property of the EM
              fixed point. The EM algorithm converges slowly near the optimum,
              so the fits are also repeated with a tighter tolerance.
Experiment 3  A target with little variation (normal with mean 10 and sd 1).
              A PH distribution with p phases has a squared coefficient of
              variation (variance / mean^2) of at least 1/p, and only the
              Erlang distribution attains it (Aldous and Shepp, 1987). The fit
              must therefore be the Erlang with p phases and the target mean.

The fitted distributions are evaluated here with scipy's matrix exponential,
independently of PhaseDist.

Run from the repository root:  python validation/fitcph2dist_estimation.py
Runtime: about half a minute. The tables are printed and saved, together with the
figures, in validation/output/.

Aldous, D., & Shepp, L. (1987). The least variable phase type distribution is
Erlang. Stochastic Models, 3(3), 467-473.
Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in
Applied Probability. Springer.
"""

import contextlib
import io
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import expm

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "src"))
from phasedist import fitcph2dist

OUTPUT = os.path.join(here, "output")
NAME = "fitcph2dist_estimation"
SEED = 1
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"]
REPORT = []


def report(line=""):
    """Prints a line of the report and keeps it for the saved copy."""
    print(line)
    REPORT.append(line)


def fit(target, nphases, dtype, **settings):
    """
    Fits a PH distribution to a target with fitcph2dist. The target is given
    as a function that configures the fitcph2dist object, e.g.
    lambda model: model.gamma(shape=3, rate=2). Returns the fitted initial
    distribution and generator, and the number of warnings raised.
    """
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        model = fitcph2dist(nphases=nphases, dtype=dtype, seed=SEED, **settings)
        target(model)
        model.fit()
    return np.asarray(model.getinitdist(), dtype=float), np.asarray(model.getphasegen(), dtype=float), len(caught)


def ph_cdf(pi, T, x):
    """The CDF 1 - pi exp(T x) 1 of a PH distribution, by scipy's expm."""
    ones = np.ones(T.shape[0])
    return np.array([1.0 - pi @ expm(T * value) @ ones for value in np.atleast_1d(x)])


def ph_density(pi, T, x):
    """The density pi exp(T x) t of a PH distribution, by scipy's expm."""
    exits = -T @ np.ones(T.shape[0])
    return np.array([pi @ expm(T * value) @ exits for value in np.atleast_1d(x)])


def ph_moments(pi, T):
    """Mean and variance of a PH distribution: E[X^k] = k! pi (-T)^(-k) 1."""
    green = np.linalg.inv(-T)
    ones = np.ones(T.shape[0])
    mean = pi @ green @ ones
    second = 2.0 * pi @ green @ green @ ones
    return mean, second - mean ** 2


# ----------------------------------------------------------------------
# Experiment 1: Targets that are phase-type, fitted with their own structure
# ----------------------------------------------------------------------

def coxian_parameters(pi, T):
    """(rate 1, probability of moving on to phase 2, rate 2) of a 2-phase Coxian."""
    return np.array([-T[0, 0], T[0, 1] / -T[0, 0], -T[1, 1]])


def hyperexp_parameters(pi, T):
    """(probability, rate) of each branch of a 2-phase hyperexponential, the slow branch first."""
    order = np.argsort(-np.diag(T))
    return np.array([pi[order[0]], -T[order[0], order[0]], pi[order[1]], -T[order[1], order[1]]])


def generlang_parameters(pi, T):
    """The two rates of a 2-phase generalised Erlang, the smaller first (their order is not identified)."""
    return np.sort(-np.diag(T))


# Coxian: rate 2, then on to phase 2 with probability 0.5, rate 0.5 there. With
# these rates its representation is unique, so the parameters are identified.
PH_TARGETS = {
    "Coxian": (np.array([1.0, 0.0]), np.array([[-2.0, 1.0], [0.0, -0.5]]), "coxian", coxian_parameters),
    "hyperexponential": (np.array([0.3, 0.7]), np.array([[-5.0, 0.0], [0.0, -0.5]]), "hyperexp", hyperexp_parameters),
    "generalised Erlang": (np.array([1.0, 0.0]), np.array([[-1.0, 1.0], [0.0, -3.0]]), "generlang", generlang_parameters),
}
GRID_STEPS = [50, 100, 200, 400, 800]


def experiment1():
    report("EXPERIMENT 1: Phase-type targets fitted with their own structure")
    report("The largest relative error of the fitted parameters, by the number of grid cells (steps).")
    report("The default is steps = 50. The error should fall about four-fold per doubling (second order).")
    report()
    report(f"{'target':20s} {'true parameters':32s}" + "".join(f"{s:>10d}" for s in GRID_STEPS) + f"{'warnings':>10s}")
    errors = {}
    for name, (pi, T, dtype, parameters) in PH_TARGETS.items():
        truth = parameters(pi, T)
        row = []
        warned = 0
        for steps in GRID_STEPS:
            fitted_pi, fitted_T, count = fit(lambda model: model.phasedist(pi, T), 2, dtype, steps=steps)
            row.append(np.max(np.abs(parameters(fitted_pi, fitted_T) - truth) / truth))
            warned += count
        errors[name] = row
        report(f"{name:20s} {np.array2string(truth, precision=2):32s}" + "".join(f"{e:10.1e}" for e in row) + f"{warned:10d}")
    report()

    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    for color, (name, row) in zip(COLORS, errors.items()):
        axis.loglog(GRID_STEPS, row, "o-", color=color, lw=1.8, ms=6, label=name)
    reference = errors["Coxian"][0] * (np.array(GRID_STEPS) / GRID_STEPS[0]) ** -2.0
    axis.loglog(GRID_STEPS, reference, "--", color="#6b6b6b", lw=1.0, label="slope -2 (error ~ 1/steps$^2$)")
    axis.set_xticks(GRID_STEPS)
    axis.set_xticklabels([str(s) for s in GRID_STEPS])
    axis.minorticks_off()
    axis.set_xlabel("Grid cells (steps)")
    axis.set_ylabel("Largest relative parameter error")
    axis.set_title("Experiment 1: Phase-type targets are recovered", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    save(figure, "experiment1_recovery.png")


# ----------------------------------------------------------------------
# Experiment 2: Targets that are not phase-type, with 1 to 8 phases
# ----------------------------------------------------------------------

# name: (configuration of fitcph2dist, the same distribution in scipy)
TARGETS = {
    "log-normal (0, 0.5)": (lambda model: model.lognorm(mu=0.0, sigma=0.5), stats.lognorm(0.5, scale=1.0)),
    "Weibull (shape 1.5)": (lambda model: model.weibull(shape=1.5, scale=1.0), stats.weibull_min(1.5, scale=1.0)),
    "gamma (shape 2.5)": (lambda model: model.gamma(shape=2.5, rate=1.0), stats.gamma(2.5, scale=1.0)),
    "normal (3, 1) > 0": (lambda model: model.norm(mu=3.0, sigma=1.0), stats.truncnorm(-3.0, np.inf, loc=3.0, scale=1.0)),
}
PHASES = [1, 2, 3, 4, 6, 8]


def experiment2():
    report("EXPERIMENT 2: Targets that are not phase-type, fitted with p phases (dtype 'general', the default)")
    report("KS: the largest distance between the fitted and the target CDF. Mean: relative error of the")
    report("fitted mean, which the EM fixed point makes equal to the target mean. Var: fitted/target")
    report("variance. q95: relative error of the 95% quantile. SCV: fitted variance/mean^2, which is at")
    report("least 1/p for any PH distribution. KS 1e-7: KS with the tolerance tightened from the default")
    report("1e-5 to 1e-7; where it is smaller, the default stopped the slow EM before the optimum.")
    report()
    distances = {}
    tight = {}
    fits = {}
    for name, (configure, target) in TARGETS.items():
        grid = np.linspace(0.0, target.ppf(0.999), 400)
        report(f"{name}: target mean {target.mean():.4f}, variance {target.var():.4f}, SCV {target.var() / target.mean() ** 2:.4f}")
        report(f"{'p':>4s} {'KS':>9s} {'mean':>10s} {'var':>8s} {'q95':>9s} {'SCV':>8s} {'1/p':>8s} {'KS 1e-7':>9s} {'warnings':>9s}")
        distances[name] = []
        tight[name] = []
        for p in PHASES:
            pi, T, warned = fit(configure, p, "general")
            mean, variance = ph_moments(pi, T)
            ks = np.max(np.abs(ph_cdf(pi, T, grid) - target.cdf(grid)))
            q95 = quantile(pi, T, 0.95, target.ppf(0.95))
            tightpi, tightT, tightwarned = fit(configure, p, "general", tolerance=1e-7)
            kstight = np.max(np.abs(ph_cdf(tightpi, tightT, grid) - target.cdf(grid)))
            distances[name].append(ks)
            tight[name].append(kstight)
            fits[(name, p)] = (pi, T)
            report(f"{p:4d} {ks:9.2e} {mean / target.mean() - 1:10.1e} {variance / target.var():8.4f} "
                   f"{q95 / target.ppf(0.95) - 1:9.1e} {variance / mean ** 2:8.4f} {1 / p:8.4f} {kstight:9.2e} "
                   f"{warned + tightwarned:9d}")
        report()

    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    for axis, (name, (configure, target)) in zip(axes.ravel(), TARGETS.items()):
        x = np.linspace(1e-3, target.ppf(0.999), 300)
        axis.plot(x, target.pdf(x), color="#1f1f1f", lw=2.4, label="target")
        for color, p in zip(COLORS, [1, 2, 4, 8]):
            pi, T = fits[(name, p)]
            axis.plot(x, ph_density(pi, T, x), color=color, lw=1.4, label=f"{p} phase{'s' if p > 1 else ''}")
        axis.set_title(name, fontsize=10)
        axis.set_yticks([])
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Experiment 2: Densities of the fits approach the target as phases are added", fontsize=11)
    save(figure, "experiment2_densities.png")

    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    for color, (name, row) in zip(COLORS, distances.items()):
        axis.semilogy(PHASES, row, "o-", color=color, lw=1.8, ms=6, label=name)
        axis.semilogy(PHASES, tight[name], "o--", color=color, lw=1.2, ms=4, alpha=0.8)
    axis.plot([], [], "-", color="#6b6b6b", label="default tolerance (1e-5)")
    axis.plot([], [], "--", color="#6b6b6b", label="tolerance 1e-7")
    axis.set_xlabel("Phases (p)")
    axis.set_ylabel("Largest CDF distance (KS)")
    axis.set_title("Experiment 2: Distance to the target", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    save(figure, "experiment2_distance.png")


def quantile(pi, T, p, start):
    """The p-quantile of a PH distribution by Newton's method from start."""
    x = start
    for _ in range(50):
        step = (ph_cdf(pi, T, x)[0] - p) / ph_density(pi, T, x)[0]
        x = max(x - step, x / 2.0)
        if abs(step) < 1e-12 * x:
            break
    return x


# ----------------------------------------------------------------------
# Experiment 3: The least variable target, and the Erlang bound
# ----------------------------------------------------------------------

def experiment3():
    report("EXPERIMENT 3: Normal (10, 1) target, SCV 0.01, fitted with p phases (dtype 'generlang')")
    report("Every PH distribution with p phases has SCV >= 1/p, with equality only for the Erlang, so the")
    report("fit must be the Erlang with all p rates equal to p/10 (mean 10) and p * SCV = 1.")
    report()
    report(f"{'p':>4s} {'mean':>10s} {'p * SCV':>10s} {'largest |rate - p/10| / (p/10)':>32s} {'warnings':>9s}")
    for p in [2, 4, 8, 16, 32]:
        pi, T, warned = fit(lambda model: model.norm(mu=10.0, sigma=1.0), p, "generlang")
        mean, variance = ph_moments(pi, T)
        rates = -np.diag(T)
        report(f"{p:4d} {mean:10.6f} {p * variance / mean ** 2:10.6f} {np.max(np.abs(rates - p / 10.0)) / (p / 10.0):32.1e} {warned:9d}")
    report()


def save(figure, filename):
    """Saves a figure in the output folder."""
    figure.tight_layout()
    figure.savefig(os.path.join(OUTPUT, f"{NAME}_{filename}"), dpi=130)
    plt.close(figure)


if __name__ == "__main__":
    os.makedirs(OUTPUT, exist_ok=True)
    experiment1()
    experiment2()
    experiment3()
    with open(os.path.join(OUTPUT, f"{NAME}.txt"), "w") as stream:
        stream.write("\n".join(REPORT) + "\n")
