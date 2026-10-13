'''
UNIT TEST FOR THE fitreg CLASS (the user-facing interface that fits a
continuous-time phase-type regression with fixed effects, interactions and an
optional Gaussian random effect, specified by an R-style formula on a pandas
DataFrame).

The first two cases are exact ground truths: with one phase the model is a
classical regression whose maximum likelihood estimates can be computed by
other means, so the class must reproduce them to optimizer precision. The
oracles are independent of the class: statsmodels' Gamma GLM on a design matrix
built by hand, and a direct maximization of the random-effect likelihood
with each group's integral computed by the trapezoidal rule on a fine grid.
Case 3 checks the design with interactions against columns built by hand. The
remaining cases are smoke tests of each functionality on multi-phase data,
using one phase wherever the functionality does not depend on the number of
phases, to keep the run time down. Statistical validation (coverage of the
confidence intervals, recovery of the parameters over many replicates) is too
slow for this file and lives in tests/validation/, see the design note
phreg-fitting-algorithm.md, Section 11.

Ground truth of the multi-phase cases. The data are simulated from the model
the class fits: a three-phase Coxian with distinct rates, scaled by
exp(x'beta + b). The simulator builds the Markov jump process from the raw
rates and shares no code with the class.

Sub-tests:
    Case 1:  With one phase the model is exponential regression: coefficients,
             baseline rate and log-likelihood equal those of a log-link Gamma
             GLM (statsmodels) on a hand-built design with continuous,
             categorical and interaction effects, the coefficients with the sign
             flipped, and the standard errors equal the exact observed
             information of exponential regression.
    Case 2:  With one phase and a random effect, sigma, the baseline rate,
             the coefficient and the marginal log-likelihood equal a direct
             maximization of the marginal likelihood computed by quadrature,
             and the standard errors equal its inverse Hessian.
    Case 3:  Interactions: '*' expands to the main effects and interactions,
             repeated effects are kept once (x:z and z:x being one effect), an
             interaction without its lower-order effects is refused, the same
             model written in another order gives the same fit, and the design
             columns, their names and a prediction row equal hand-built
             products.
    Case 4:  A three-phase fixed-effects fit converges, names the coefficients
             after the variables and levels, and recovers the effects.
    Case 5:  Variable types come from the DataFrame: the 'category' dtype makes
             a variable categorical, its first category is the reference
             level (reordering a two-level factor flips the sign of its
             coefficient and leaves the fit unchanged), unused categories are
             ignored, and text, boolean and non-categorical grouping columns
             are refused.
    Case 6:  Shifting a continuous covariate leaves the centred design, the
             coefficients and the log-likelihood unchanged, and the shifted
             model at x = shift is the original model at x = 0.
    Case 7:  A random effect on three-phase data: sigma is recovered and
             every group gets a predicted intercept. When the groups vary less
             than chance would make them, sigma is exactly zero and the fit is
             that of the model without random effect.
    Case 8:  Predictions: getdist, getdensity and getcumprob agree with each
             other and with PH(pi, exp(x'beta) T), interactions included, the
             density is the derivative of the CDF, a group's predicted
             intercept enters the linear predictor, and bad covariates are
             refused.
    Case 9:  The formula parser accepts the supported notation and refuses the
             unsupported notation with a message.
    Case 10: Infeasible input terminates, and the arguments of earlier
             versions are refused.
    Case 11: summary prints, plot writes a file, AIC and BIC are the same
             parameter count read two ways, the confidence intervals are Wald
             intervals, the p-values and getwaldtest are Wald tests (the
             latter checked against the exact covariance of case 1), the
             summary reports the model and its effects but no individual
             coefficients: likelihood-ratio tests of every effect by default,
             Wald or none on request, the test of sigma = 0 (against the fit
             without the random effect, and at the boundary), the
             log-likelihood and deviance, the parameters by kind and the
             starts; coeftest returns and prints the Wald test, standard
             error and interval of every coefficient; and verbose decides
             whether anything is printed.
    Case 12: Observations at zero are a point mass, as in fit: its
             probability is the share of zeros, the regression is the fit to
             the positive observations, and the log-likelihood, parameter
             count, BIC, initial distribution, CDF, density and summary
             include the point mass; with and without a random effect.
    Case 13: getlrtest. With one phase the likelihood-ratio statistics of
             effects, including effects that are part of an interaction (whose
             columns alone are dropped), equal those of exponential
             regression computed from Gamma GLMs on hand-built designs; the
             p-value is the chi-square tail. Zeros leave the test unchanged,
             a random effect and a model left with no covariates work,
             and on three-phase data the statistics are not negative. The
             simulated p-value has the stated form and Monte Carlo interval,
             is reproducible with its seed, is the one the summary prints,
             and contains the chi-square p-value, which is accurate for
             one-phase data (a statistical test at a set level); the
             simulators behind it follow their laws (statistical tests).
             Bad arguments are refused.

References:
    Bladt, M. (2022). Phase-type distributions for claim severity regression
    modeling. ASTIN Bulletin, 52(2), 417-448.
    https://doi.org/10.1017/asb.2021.40
'''

import contextlib
import io
import os
import sys
import warnings

import numpy as np
import matplotlib

# a backend that needs no display, since case 11 only writes the figure to file
matplotlib.use("Agg")

from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist
from scipy.linalg import expm
from scipy.stats import chi2, norm
from scipy.stats import t as t_dist

# Load phasedist from the src-folder so the test can be run without installing
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from phasedist.fitreg import fitreg
from phasedist.dist import dist
from phasedist.regdesign import _regformula, _regdesign

try:
    import pandas as pd
    import statsmodels.api as sm
except ImportError:
    sys.exit("Validation test failed at initialization: the test needs pandas and statsmodels, which are test dependencies.")


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

TOL = 1e-10          # tolerance for comparisons that are exact in theory
TOLEXACT = 1e-8      # fit against a closed-form oracle (case 1)
TOLFIT = 1e-3        # loose convergence tolerance, enough for a smoke test
TOLTIGHT = 1e-10     # convergence tolerance of the ground-truth fits

# The oracle of case 2 is a numerical optimizer of its own, so the comparison
# is only as good as it is. Measured over seeds 0 and 1: it agrees to 1.2e-7
# in the parameters.
TOLOPT = 1e-5

# Two fits whose inputs differ only by rounding (the centred design of case 6
# differs by about 4e-15) take slightly different paths and stop at slightly
# different points of the flat ridge around the optimum, the representation
# not being identified. As in test_fit, two runs are asked to agree to 1e-4.
# Measured over seeds 0-3, the worst disagreement was 4.5e-7 in a coefficient
# and 2.3e-6 in the CDF (seed 3).
TOLRUN = 1e-4

# Standard errors from finite differences of two likelihoods computed in
# different ways (case 2). Measured over seeds 0 and 1 they agreed to six
# decimals; 1e-4 relative leaves room for the finite-difference error.
STEP = 1e-4
TOLSE = 1e-4

SEED = 0
NOBS = 300           # observations of the multi-phase fixed-effects data
NEXACT = 400         # observations of the one-phase data of case 1
NGROUPS = 40         # groups of the multi-phase random-effect data
GROUPSIZE = 10
NGROUPSEXACT = 20    # groups of the one-phase random-effect data of case 2
GROUPSIZEEXACT = 6
SHIFT = 10.0         # shift of the continuous covariate in case 6

# The simulated p-value of case 13: NSIMLR simulated data sets, on one-phase
# data of NLR observations with a moderate effect of w, so that the simulated
# p-value is small and a simulation from the wrong model (the full one) would
# move it far from the chi-square p-value. The chi-square reference is
# accurate for one-phase data (validation: 1000 data sets of 400 observations
# rejected at 5.3% and 4.0% at the 5% level), so the chi-square p-value must
# lie in the Clopper-Pearson interval of the simulated one, taken at level
# 1 - ALPHALR.
NLR = 200
NSIMLR = 40
LREFFECT = 0.35
# The simulators behind the simulated p-value are checked against theory by
# statistical tests too: NDRAWS absorption times of the true Coxian (mean by a
# t interval, distribution by a chi-square test on NBINSLR bins), and NREPLR
# simulated data sets of a one-phase random-effect model (the mean of
# log(y) + eta and the variance of its group means, by t intervals). With the
# Clopper-Pearson check of the p-value these are NCHECKSLR checks, sharing
# the level ALPHALR (Bonferroni).
NDRAWS = 20000
NBINSLR = 10
NREPLR = 400
ALPHALR = 0.01
NCHECKSLR = 5

# The multi-phase truth: a Coxian of three phases with distinct rates, so the
# phases are not exchangeable, and effects large enough that their signs are
# clear at NOBS observations.
TRUERATES = np.array([3.0, 1.0, 0.4])
TRUECONTINUE = np.array([0.6, 0.5])
TRUEBETA = {"x": 0.8, "g[B]": -0.6, "g[C]": 0.7}

# The random effect must be identifiable from the dependence within groups,
# and the true Coxian is wide (log U has a standard deviation of about 1.5),
# so a small sigma would leave the within-group correlation near zero and the
# estimate of sigma at the boundary. With sigma = 1 and groups of ten the
# correlation of log Y within a group is about 0.3.
TRUESIGMA = 1.0

# The one-phase truths of cases 1 and 2
EXACTRATE = 0.5
EXACTBETA = dict(x=0.6, gb=-0.4, gc=0.5, w=0.3, hq=0.3, xgb=0.4, xgc=-0.3, xw=0.5, gbhq=0.2, gchq=-0.2)
EXACTSIGMA = 0.8

# Every public method of fitreg. Checked against the class below, so that a
# method added later cannot go untested unnoticed.
COVEREDMETHODS = {"getcoef", "getstderr", "coeftest", "getwaldtest", "getlrtest", "getsigma", "getzeroprob",
                  "getranef", "getinitdist", "getphasegen", "getexitrates",
                  "getloglik", "getaic", "getbic", "getdist", "getdensity",
                  "getcumprob", "summary", "plot"}
publicmethods = {name for name in dir(fitreg) if not name.startswith("_")}
if publicmethods != COVEREDMETHODS:
    sys.exit("Validation test failed at initialization: the public methods of fitreg are %s, but the test covers %s." % (sorted(publicmethods), sorted(COVEREDMETHODS)))

if len(set(TRUERATES)) != TRUERATES.size:
    sys.exit("Validation test failed at initialization: the true rates must stay apart, so that the truth is not exchangeable between phases.")


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def simulatecoxian(n, rng):
    """
    Simulates n absorption times of the true Coxian by running its Markov jump
    process from the raw rates, sharing no code with the class under test.
    """
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


def quietfit(**arguments):
    """
    Fits without letting the class print or warn into the test output, and
    returns the model with whatever it printed and warned.
    """
    buffer = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(buffer):
            model = fitreg(**arguments)
    return model, buffer.getvalue(), caught


def terminates(**arguments):
    """Returns whether fitreg terminates on the given arguments."""
    buffer = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(buffer):
                fitreg(**arguments)
    except SystemExit:
        return True
    return False


def categorical(values, categories=None):
    """Returns the values as a pandas categorical column."""
    return pd.Categorical(values, categories=categories)


rng = np.random.default_rng(SEED)

# The one-phase data of case 1: two continuous and two categorical
# covariates, with interactions between all kinds
xe = rng.normal(size=NEXACT)
we = rng.uniform(-1.0, 1.0, size=NEXACT)
ge = rng.choice(["a", "b", "c"], size=NEXACT)
he = rng.choice(["p", "q"], size=NEXACT)
gb, gc, hq = (ge == "b") * 1.0, (ge == "c") * 1.0, (he == "q") * 1.0
# the design of 'y ~ x*g + w + x:w + g*h', built by hand in the order and with
# the names the class is documented to use
HANDNAMES = ["x", "g[b]", "g[c]", "w", "h[q]", "x:g[b]", "x:g[c]", "x:w", "g[b]:h[q]", "g[c]:h[q]"]
HANDX = np.column_stack([xe, gb, gc, we, hq, xe * gb, xe * gc, xe * we, gb * hq, gc * hq])
HANDBETA = np.array([EXACTBETA[k] for k in ("x", "gb", "gc", "w", "hq", "xgb", "xgc", "xw", "gbhq", "gchq")])
ye = rng.exponential(size=NEXACT) / (EXACTRATE * np.exp(HANDX @ HANDBETA))
DATAEXACT = pd.DataFrame({"y": ye, "x": xe, "w": we, "g": categorical(ge), "h": categorical(he)})
FORMULAEXACT = "y ~ x*g + w + x:w + g*h"

# The one-phase random-effect data of case 2
groupsexact = np.repeat(np.arange(NGROUPSEXACT), GROUPSIZEEXACT)
xre = rng.normal(size=groupsexact.size)
intercepts = rng.normal(scale=EXACTSIGMA, size=NGROUPSEXACT)
yre = rng.exponential(size=groupsexact.size) / (EXACTRATE * np.exp(EXACTBETA["x"] * xre + intercepts[groupsexact]))
DATARANDOMEXACT = pd.DataFrame({"y": yre, "x": xre, "u": categorical(groupsexact)})

# The multi-phase fixed-effects data: a continuous covariate and a
# three-level categorical
x = rng.normal(size=NOBS)
g = rng.choice(["A", "B", "C"], size=NOBS)
eta = TRUEBETA["x"] * x + TRUEBETA["g[B]"] * (g == "B") + TRUEBETA["g[C]"] * (g == "C")
y = simulatecoxian(NOBS, rng) * np.exp(-eta)
DATA = pd.DataFrame({"y": y, "x": x, "g": categorical(g)})

# The multi-phase random-effect data
groups = np.repeat(np.arange(NGROUPS), GROUPSIZE)
xr = rng.normal(size=groups.size)
groupintercepts = rng.normal(scale=TRUESIGMA, size=NGROUPS)
yr = simulatecoxian(groups.size, rng) * np.exp(-(TRUEBETA["x"] * xr + groupintercepts[groups]))
DATARANDOM = pd.DataFrame({"y": yr, "x": xr, "group": categorical(groups)})

# The same design without a random effect, for the boundary of case 7
# Data whose groups vary less than chance would make them, for the boundary of
# case 7: the observations are dealt to the groups in the order of their
# baseline times, so that every group gets a similar mix of short and long
# times. The derivative of the likelihood in sigma^2 at zero is then clearly
# negative and the estimate of sigma is exactly zero. (With groups drawn at
# random and no random effect, about half the data sets would give zero.)
uz = simulatecoxian(groups.size, rng)
yz = uz * np.exp(-TRUEBETA["x"] * xr)
dealt = np.empty(groups.size, dtype=int)
dealt[np.argsort(uz)] = np.arange(groups.size) % NGROUPS
DATANOINTERCEPT = pd.DataFrame({"y": yz, "x": xr, "group": categorical(dealt)})


for frame in (DATAEXACT, DATARANDOMEXACT, DATA, DATARANDOM, DATANOINTERCEPT):
    if not np.all(frame["y"].values > 0):
        sys.exit("Validation test failed at initialization: the simulated responses are not all positive.")
if set(g) != {"A", "B", "C"} or set(ge) != {"a", "b", "c"} or set(he) != {"p", "q"}:
    sys.exit("Validation test failed at initialization: a simulated categorical covariate does not contain all its levels.")
# every combination of the interacted factors of case 1 must occur, or the
# design would be rank deficient for a reason that is not under test
if len(set(zip(ge, he))) != 6:
    sys.exit("Validation test failed at initialization: an interaction cell of case 1 is empty.")

COMMON = dict(nphases=3, dtype="coxian", restarts=1, seed=SEED, tolerance=TOLFIT)
EXACT = dict(nphases=1, restarts=1, seed=SEED, tolerance=TOLTIGHT)


# ------------------------------------------------------------------
# CASE 1: One phase is exponential regression, i.e. a Gamma GLM
# ------------------------------------------------------------------

modelexact, _, _ = quietfit(formula=FORMULAEXACT, data=DATAEXACT, **EXACT)
coefexact = modelexact.getcoef()

if list(coefexact) != HANDNAMES:
    sys.exit("Validation test failed at case 1: the coefficients are named %s, not %s." % (list(coefexact), HANDNAMES))

# the oracle: a log-link Gamma GLM, whose estimating equations for the
# coefficients are those of exponential regression whatever the dispersion
design = np.column_stack([np.ones(NEXACT), HANDX])
glm = sm.GLM(ye, design, family=sm.families.Gamma(link=sm.families.links.Log())).fit(tol=1e-14, maxiter=1000)
estimates = np.array([coefexact[name] for name in HANDNAMES])

if np.max(np.abs(estimates + glm.params[1:])) > TOLEXACT:
    sys.exit("Validation test failed at case 1: the coefficients differ from minus those of the Gamma GLM by %.3e." % np.max(np.abs(estimates + glm.params[1:])))
rate = -modelexact.getphasegen()[0, 0]
if abs(rate / np.exp(-glm.params[0]) - 1.0) > TOLEXACT:
    sys.exit("Validation test failed at case 1: the baseline rate is %.10f, against exp(-intercept) = %.10f." % (rate, np.exp(-glm.params[0])))

# the exact log-likelihood and observed information of exponential regression
u = rate * np.exp(HANDX @ estimates) * ye
loglik = np.sum(np.log(rate) + HANDX @ estimates - u)
if abs(modelexact.getloglik() - loglik) > TOLEXACT * abs(loglik):
    sys.exit("Validation test failed at case 1: the log-likelihood is %.10f, against %.10f by hand." % (modelexact.getloglik(), loglik))
# kept for the tests of case 11: the exact covariance of the coefficients
EXACTCOVARIANCE = np.linalg.inv((design * u[:, None]).T @ design)[1:, 1:]
stderr = np.sqrt(np.diag(EXACTCOVARIANCE))
reported = np.array([modelexact.getstderr()[name] for name in HANDNAMES])
if np.max(np.abs(reported / stderr - 1.0)) > TOLEXACT:
    sys.exit("Validation test failed at case 1: the standard errors differ from the observed information of exponential regression by a factor of up to %.3e." % np.max(np.abs(reported / stderr - 1.0)))


# ------------------------------------------------------------------
# CASE 2: One phase with a random effect against a quadrature likelihood
# ------------------------------------------------------------------

modelrandomexact, _, _ = quietfit(formula="y ~ x + (1|u)", data=DATARANDOMEXACT, quadnodes=15, **EXACT)


# The integrand of a group is closed form: with a_j = sum(lograte + beta x)
# and s_j = sum(exp(lograte + beta x) y) over its observations, the log of the
# group's likelihood given b is a_j + n_j b - exp(b) s_j. Each integral is
# computed by the trapezoidal rule on a fine fixed grid, which for a smooth
# integrand vanishing at both ends converges geometrically, and shares nothing
# with the adaptive Gauss-Hermite rule of the class.
# The grid spacing is under a thirtieth of the posterior standard deviation
# of an intercept near the optimum, where the error of the rule is far below double
# precision.
GRID = np.linspace(-15.0, 15.0, 3001)
EXPGRID = np.exp(GRID)
LOGTRAPEZOID = np.log(np.r_[0.5, np.ones(GRID.size - 2), 0.5] * (GRID[1] - GRID[0]))
SIZES = np.bincount(groupsexact).astype(float)


def randomnegloglik(v):
    """
    Negative marginal log-likelihood of exponential regression with a Gaussian
    random effect, (log rate, beta, log sigma).
    """
    lograte, beta, logsigma = v
    sigma = np.exp(logsigma)
    linear = lograte + beta * xre
    a = np.bincount(groupsexact, weights=linear)
    s = np.bincount(groupsexact, weights=np.exp(linear) * yre)
    logintegrand = (a[:, None] + SIZES[:, None] * GRID[None, :] - s[:, None] * EXPGRID[None, :]
                    - 0.5 * GRID[None, :] ** 2 / sigma ** 2 - 0.5 * np.log(2.0 * np.pi * sigma ** 2))
    return -np.sum(logsumexp(logintegrand + LOGTRAPEZOID[None, :], axis=1))


oracle = minimize(randomnegloglik, np.array([0.0, 0.0, np.log(0.5)]), method="Nelder-Mead",
                  options=dict(xatol=1e-9, fatol=1e-11, maxiter=4000))
fitted = np.array([np.log(-modelrandomexact.getphasegen()[0, 0]), modelrandomexact.getcoef()["x"],
                   np.log(modelrandomexact.getsigma())])
if np.max(np.abs(fitted - oracle.x)) > TOLOPT:
    sys.exit("Validation test failed at case 2: the random-effect estimates differ from a direct maximization of the quadrature likelihood by %.3e." % np.max(np.abs(fitted - oracle.x)))
if abs(modelrandomexact.getloglik() + oracle.fun) > TOLOPT:
    sys.exit("Validation test failed at case 2: the marginal log-likelihood is %.8f, against %.8f by quadrature." % (modelrandomexact.getloglik(), -oracle.fun))

# the standard errors are those of the full observed information, the rate
# included: the inverse Hessian of the oracle's likelihood at its maximum. The
# standard error of log(sigma) is sigmastderr / sigma. (With the rate held at
# its estimate, the standard error of log(sigma) would be too small.)
hessian = np.zeros((3, 3))
for i in range(3):
    ei = np.zeros(3)
    ei[i] = STEP
    hessian[i, i] = -(randomnegloglik(oracle.x + 2 * ei) - 2 * oracle.fun + randomnegloglik(oracle.x - 2 * ei)) / (4 * STEP ** 2)
    for j in range(i + 1, 3):
        ej = np.zeros(3)
        ej[j] = STEP
        hessian[i, j] = hessian[j, i] = -(randomnegloglik(oracle.x + ei + ej) - randomnegloglik(oracle.x + ei - ej)
                                          - randomnegloglik(oracle.x - ei + ej) + randomnegloglik(oracle.x - ei - ej)) / (4 * STEP ** 2)
oraclestderr = np.sqrt(np.diag(np.linalg.inv(-hessian)))[1:]
reported = np.array([modelrandomexact.getstderr()["x"], modelrandomexact.sigmastderr / modelrandomexact.getsigma()])
if np.max(np.abs(reported / oraclestderr - 1.0)) > TOLSE:
    sys.exit("Validation test failed at case 2: the standard errors of the coefficient and log(sigma) are %s, against %s from the oracle's information." % (reported, oraclestderr))


# ------------------------------------------------------------------
# CASE 3: Interactions
# ------------------------------------------------------------------

parsed = _regformula("y ~ x*w*g + w:x + (1|u)")
expected = [("x",), ("w",), ("g",), ("x", "w"), ("x", "g"), ("w", "g"), ("x", "w", "g")]
if parsed.terms != expected or parsed.group != "u":
    sys.exit("Validation test failed at case 3: 'x*w*g + w:x' was parsed as %s, not as every main effect and interaction once." % parsed.terms)
try:
    _regformula("y ~ x + x:g")
    sys.exit("Validation test failed at case 3: an interaction without its lower-order effect was accepted.")
except ValueError as error:
    if "x*g" not in str(error):
        sys.exit("Validation test failed at case 3: the refusal of an incomplete interaction does not say how to write it.")

# x:z and z:x are one effect: within a formula a repeated interaction is kept
# once whatever the order of its variables, and it is named in the order the
# variables first appear, as in R
for formula in ("y ~ x*z + z:x", "y ~ x + z + z:x", "y ~ x + z + x:z + z:x"):
    if _regformula(formula).terms != [("x",), ("z",), ("x", "z")]:
        sys.exit("Validation test failed at case 3: '%s' was parsed as %s." % (formula, _regformula(formula).terms))
if _regformula("y ~ z*x").terms != [("z",), ("x",), ("z", "x")]:
    sys.exit("Validation test failed at case 3: 'z*x' does not keep the order it is written in.")

# and the same model written in another order is the same fit: the same
# log-likelihood, coefficients, standard errors and effect tests, the names
# only following the order of the formula. The comparison goes by the set of
# variables and levels in each name, so that 'x:g[b]' matches 'g[b]:x'
modelreordered, _, _ = quietfit(formula="y ~ h*g + w*x + g*x", data=DATAEXACT, **EXACT)


def byparts(values):
    """Keys a dictionary by the set of parts of each name."""
    return {frozenset(name.split(":")): value for name, value in values.items()}


if abs(modelreordered.getloglik() - modelexact.getloglik()) > TOLEXACT * abs(modelexact.getloglik()):
    sys.exit("Validation test failed at case 3: writing the interactions in another order changed the log-likelihood.")
for getter in ("getcoef", "getstderr", "_getwaldcoeftest"):
    original, reordered = byparts(getattr(modelexact, getter)()), byparts(getattr(modelreordered, getter)())
    if set(original) != set(reordered) or max(abs(original[k] - reordered[k]) for k in original) > TOLEXACT:
        sys.exit("Validation test failed at case 3: writing the interactions in another order changed %s." % getter)
original = byparts(modelexact.getwaldtest())
reordered = byparts(modelreordered.getwaldtest())
if set(original) != set(reordered) or max(abs(original[k][0] - reordered[k][0]) for k in original) > TOLEXACT * 100:
    sys.exit("Validation test failed at case 3: writing the interactions in another order changed the effect tests.")

# the design of case 1 is the hand-built one, column by column
if np.max(np.abs(modelexact.design.X - HANDX)) > TOL:
    sys.exit("Validation test failed at case 3: the design matrix with interactions differs from the hand-built one.")

# a prediction row is the products of the covariate values
row = modelexact.design.row({"x": 2.0, "w": -0.5, "g": "c", "h": "q"})
handrow = np.array([2.0, 0.0, 1.0, -0.5, 1.0, 0.0, 2.0, -1.0, 0.0, 1.0])
if np.max(np.abs(row - handrow)) > TOL:
    sys.exit("Validation test failed at case 3: the prediction row with interactions is %s, not %s." % (row, handrow))


# ------------------------------------------------------------------
# CASE 4: A three-phase fixed-effects fit
# ------------------------------------------------------------------

model, _, _ = quietfit(formula="y ~ x + g", data=DATA, **COMMON)
coef = model.getcoef()

if list(coef) != ["x", "g[B]", "g[C]"]:
    sys.exit("Validation test failed at case 4: the coefficients are named %s, not after the variable and the non-reference levels." % list(coef))
if not all(np.isfinite(v) for v in coef.values()) or not all(np.isfinite(v) and v > 0 for v in model.getstderr().values()):
    sys.exit("Validation test failed at case 4: a coefficient or standard error is not finite and positive.")
if not model.converged or not np.isfinite(model.getloglik()):
    sys.exit("Validation test failed at case 4: the fit did not converge to a finite log-likelihood.")
for name, truth in TRUEBETA.items():
    if abs(coef[name] - truth) > 4.0 * model.getstderr()[name]:
        sys.exit("Validation test failed at case 4: the coefficient of %s is %.3f, against a true value of %.3f and a standard error of %.3f." % (name, coef[name], truth, model.getstderr()[name]))
if model.getsigma() is not None or model.getranef() is not None:
    sys.exit("Validation test failed at case 4: a model without random effect reports a sigma or random effects.")


# ------------------------------------------------------------------
# CASE 5: Variable types come from the DataFrame
# ------------------------------------------------------------------

# the reference level is the first category: reordering a two-level factor
# flips the sign of its coefficient and leaves the fit unchanged. The type
# handling does not depend on the number of phases, so one phase suffices,
# whose unique optimum makes the comparison exact
modelpq, _, _ = quietfit(formula="y ~ x + h", data=DATAEXACT, **EXACT)
modelqp, _, _ = quietfit(formula="y ~ x + h",
                         data=DATAEXACT.assign(h=DATAEXACT["h"].cat.reorder_categories(["q", "p"])), **EXACT)
if list(modelpq.getcoef()) != ["x", "h[q]"] or list(modelqp.getcoef()) != ["x", "h[p]"]:
    sys.exit("Validation test failed at case 5: the reference level is not the first category.")
if abs(modelqp.getcoef()["h[p]"] + modelpq.getcoef()["h[q]"]) > TOLEXACT or \
        abs(modelqp.getcoef()["x"] - modelpq.getcoef()["x"]) > TOLEXACT or \
        abs(modelqp.getloglik() - modelpq.getloglik()) > TOLEXACT * abs(modelpq.getloglik()):
    sys.exit("Validation test failed at case 5: changing the reference level changed the fit.")
# the baseline is now the other level, so the rate absorbs the coefficient
if abs(np.log(modelqp.getphasegen()[0, 0] / modelpq.getphasegen()[0, 0]) - modelpq.getcoef()["h[q]"]) > TOLEXACT:
    sys.exit("Validation test failed at case 5: the baseline rate did not move to the new reference level.")

# An unused category is ignored, and integer codes are categorical once given
# the category dtype. The fit is a deterministic function of the response and
# the design, so equal designs are equal fits
TERMS = [("x",), ("g",)]
designused = _regdesign(DATA, "y", TERMS)
designunused = _regdesign(DATA.assign(g=DATA["g"].cat.add_categories(["Z"])), "y", TERMS)
if designunused.columns != designused.columns or np.max(np.abs(designunused.X - designused.X)) > 0.0:
    sys.exit("Validation test failed at case 5: an unused category changed the design.")
designcodes = _regdesign(DATA.assign(g=categorical(DATA["g"].cat.codes.values)), "y", TERMS)
if designcodes.columns != ["x", "g[1]", "g[2]"] or np.max(np.abs(designcodes.X - designused.X)) > 0.0:
    sys.exit("Validation test failed at case 5: integer codes with the category dtype are not read as the same categorical variable.")

# any other type must be converted first
# (pandas 1.5 with numpy 1.24 warns inside astype(str) of a categorical)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    textcolumn = DATA["g"].astype(str)
refused = {
    "a text column": dict(formula="y ~ x + g", data=DATA.assign(g=textcolumn)),
    "a boolean column": dict(formula="y ~ b", data=DATA.assign(b=DATA["x"] > 0)),
    "a numeric grouping variable": dict(formula="y ~ x + (1|group)", data=DATARANDOM.assign(group=groups)),
    "a categorical response": dict(formula="y ~ x", data=DATA.assign(y=categorical(DATA["y"].round(1)))),
}
for name, arguments in refused.items():
    if not terminates(**arguments, **COMMON):
        sys.exit("Validation test failed at case 5: %s was accepted." % name)


# ------------------------------------------------------------------
# CASE 6: Shifting a continuous covariate
# ------------------------------------------------------------------

modelshift, _, _ = quietfit(formula="y ~ x + g", data=DATA.assign(x=DATA["x"] + SHIFT), **COMMON)

if np.max(np.abs(modelshift.design.Xtilde - model.design.Xtilde)) > TOL:
    sys.exit("Validation test failed at case 6: shifting a covariate changed the centred design.")
if abs(modelshift.getloglik() - model.getloglik()) > TOLRUN:
    sys.exit("Validation test failed at case 6: shifting a covariate changed the log-likelihood by %.3e." % abs(modelshift.getloglik() - model.getloglik()))
if max(abs(modelshift.getcoef()[k] - coef[k]) for k in coef) > TOLRUN:
    sys.exit("Validation test failed at case 6: shifting a covariate changed the coefficients by %.3e." % max(abs(modelshift.getcoef()[k] - coef[k]) for k in coef))
# the shifted model at x = SHIFT is the original model at x = 0. The
# distributions are compared rather than the generators, the representation
# not being identified: two runs that stop a few iterations apart agree on the
# distribution while their generators may still drift
for v in np.quantile(DATA["y"], [0.1, 0.5, 0.9]):
    if abs(modelshift.getcumprob(v, {"x": SHIFT, "g": "A"}) - model.getcumprob(v, {"x": 0.0, "g": "A"})) > TOLRUN:
        sys.exit("Validation test failed at case 6: the shifted model at x = %g is not the original model at x = 0." % SHIFT)


# ------------------------------------------------------------------
# CASE 7: A random effect on three-phase data
# ------------------------------------------------------------------

modelrandom, _, _ = quietfit(formula="y ~ x + (1|group)", data=DATARANDOM, **COMMON)

sigma = modelrandom.getsigma()
if sigma is None or not (np.isfinite(sigma) and sigma > 0.0) or not np.isfinite(modelrandom.sigmastderr):
    sys.exit("Validation test failed at case 7: sigma or its standard error is not finite and positive.")
if abs(sigma - TRUESIGMA) > 4.0 * modelrandom.sigmastderr:
    sys.exit("Validation test failed at case 7: sigma is %.3f, against a true value of %.3f and a standard error of %.3f." % (sigma, TRUESIGMA, modelrandom.sigmastderr))
ranef = modelrandom.getranef()
if ranef is None or sorted(ranef) != list(range(NGROUPS)) or not all(np.isfinite(v) for v in ranef.values()):
    sys.exit("Validation test failed at case 7: not every group has a finite predicted intercept.")
if np.corrcoef([ranef[k] for k in range(NGROUPS)], groupintercepts)[0, 1] < 0.5:
    sys.exit("Validation test failed at case 7: the predicted intercepts are hardly related to the true ones.")
if abs(modelrandom.getcoef()["x"] - TRUEBETA["x"]) > 4.0 * modelrandom.getstderr()["x"]:
    sys.exit("Validation test failed at case 7: the coefficient of x is not recovered.")

# At the boundary sigma is exactly zero and the fit is that of the model
# without random effect: the same coefficients, standard errors and
# log-likelihood, every predicted intercept zero, a warning, and no standard
# error for sigma
modelzero, _, caughtzero = quietfit(formula="y ~ x + (1|group)", data=DATANOINTERCEPT, **COMMON)
modelnogroup, _, _ = quietfit(formula="y ~ x", data=DATANOINTERCEPT, **COMMON)
if not modelzero.boundary or modelzero.getsigma() != 0.0:
    sys.exit("Validation test failed at case 7: sigma is %.3g, not zero at the boundary." % modelzero.getsigma())
if not any("standard deviation of the random effect is zero" in str(w.message) for w in caughtzero):
    sys.exit("Validation test failed at case 7: the boundary was reached without a warning.")
if modelzero.sigmastderr is None or np.isfinite(modelzero.sigmastderr):
    sys.exit("Validation test failed at case 7: a standard error is reported for sigma at the boundary.")
# (the class orders the observations by group, so the two fits see the data
# in a different order and agree to TOLRUN rather than to rounding)
if abs(modelzero.getloglik() - modelnogroup.getloglik()) > TOLRUN or \
        abs(modelzero.getcoef()["x"] - modelnogroup.getcoef()["x"]) > TOLRUN or \
        abs(modelzero.getstderr()["x"] - modelnogroup.getstderr()["x"]) > TOLRUN:
    sys.exit("Validation test failed at case 7: the fit at the boundary is not that of the model without random effect.")
if any(v != 0.0 for v in modelzero.getranef().values()):
    sys.exit("Validation test failed at case 7: a predicted intercept is not zero at the boundary.")
# one parameter more than the model without random effect, as in lme4
if modelzero.nparam != modelnogroup.nparam + 1:
    sys.exit("Validation test failed at case 7: sigma is not counted as a parameter at the boundary.")


# ------------------------------------------------------------------
# CASE 8: Predictions
# ------------------------------------------------------------------

baseline = model.getdist()
if not isinstance(baseline, dist) or np.max(np.abs(baseline.getphasegen() - model.getphasegen())) > TOL:
    sys.exit("Validation test failed at case 8: getdist without covariates is not the baseline distribution.")

covariates = {"x": 0.5, "g": "C"}
etahand = 0.5 * coef["x"] + coef["g[C]"]
conditional = model.getdist(covariates)
if np.max(np.abs(conditional.getphasegen() - np.exp(etahand) * model.getphasegen())) > TOL * np.max(np.abs(model.getphasegen())) * np.exp(abs(etahand)):
    sys.exit("Validation test failed at case 8: getdist is not PH(pi, exp(x'beta) T).")
for v in np.quantile(y, [0.1, 0.5, 0.9]):
    if abs(conditional.getcumprob(v) - model.getcumprob(v, covariates)) > 1e-9 or \
            abs(conditional.getdensity(v) - model.getdensity(v, covariates)) > 1e-9:
        sys.exit("Validation test failed at case 8: getcumprob or getdensity disagrees with the distribution from getdist.")
    # and, independently of dist, the density is the derivative of the CDF
    h = 1e-5 * v
    numeric = (model.getcumprob(v + h, covariates) - model.getcumprob(v - h, covariates)) / (2.0 * h)
    if abs(numeric - model.getdensity(v, covariates)) > 1e-6 * max(1.0, numeric):
        sys.exit("Validation test failed at case 8: the density is not the derivative of the CDF.")

# with interactions the linear predictor is the sum over every column
values = {"x": 2.0, "w": -0.5, "g": "c", "h": "q"}
etahand = float(handrow @ np.array([modelexact.getcoef()[name] for name in HANDNAMES]))
rate = -modelexact.getphasegen()[0, 0]
for v in (0.3, 1.0, 3.0):
    if abs(modelexact.getcumprob(v, values) - (1.0 - np.exp(-rate * np.exp(etahand) * v))) > 1e-12:
        sys.exit("Validation test failed at case 8: the CDF with interactions is not the exponential with rate exp(x'beta) times the baseline.")

label = 3
withgroup = modelrandom.getdist({"x": 0.0}, group=label)
expectedgen = np.exp(modelrandom.getranef()[label]) * modelrandom.getphasegen()
if np.max(np.abs(withgroup.getphasegen() - expectedgen)) > TOL * np.max(np.abs(expectedgen)):
    sys.exit("Validation test failed at case 8: a group's predicted intercept does not enter the linear predictor.")

buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    bad = [model.getdist({"x": 0.5}), model.getdist({"x": 0.5, "g": "D"}),
           model.getdist({"x": 0.5, "g": "C", "w": 1.0}), model.getdist(group=1),
           modelrandom.getdist(group=NGROUPS + 5)]
if any(b is not None for b in bad) or buffer.getvalue().count("Error") != len(bad):
    sys.exit("Validation test failed at case 8: a missing variable, an unknown level, an unknown variable, a group in a model without groups or an unknown group was not refused.")


# ------------------------------------------------------------------
# CASE 9: The formula parser
# ------------------------------------------------------------------

accepted = {
    "y ~ x + z": ("y", [("x",), ("z",)], None),
    "y~x+z+(1|g)": ("y", [("x",), ("z",)], "g"),
    " y ~ 1 + x ": ("y", [("x",)], None),
    "y ~ (1 | g)": ("y", [], "g"),
    "y ~ x.1 + z_2 + ( 1|grp )": ("y", [("x.1",), ("z_2",)], "grp"),
    "y ~ x*z": ("y", [("x",), ("z",), ("x", "z")], None),
    "y ~ z:x + x + z": ("y", [("x",), ("z",), ("z", "x")], None),
    "y ~ x + x": ("y", [("x",)], None),
}
for formula, (response, terms, group) in accepted.items():
    parsed = _regformula(formula)
    if (parsed.response, parsed.terms, parsed.group) != (response, terms, group):
        sys.exit("Validation test failed at case 9: '%s' was parsed as %s." % (formula, (parsed.response, parsed.terms, parsed.group)))

refused = ["y x + z", "y ~ x ~ z", "y ~ x:z", "y ~ log(x)", "y ~ x - 1", "y ~ 0 + x",
           "y ~ x + (x|g)", "y ~ x + (1|g) + (1|h)", "y ~ x + (1||g)", "y ~ x + y",
           "y ~ g + (1|g)", "y ~ 1", "log(y) ~ x", "y ~ x*z:w", "y ~ x:x", 42]
for formula in refused:
    try:
        _regformula(formula)
    except ValueError as error:
        if not str(error):
            sys.exit("Validation test failed at case 9: '%s' was refused without a message." % formula)
        continue
    sys.exit("Validation test failed at case 9: '%s' was accepted." % formula)


# ------------------------------------------------------------------
# CASE 10: Infeasible input terminates
# ------------------------------------------------------------------

infeasible = {
    "no formula": dict(data=DATA),
    "no data": dict(formula="y ~ x"),
    "a dictionary as data": dict(formula="y ~ x", data={"y": y, "x": x}),
    "an unknown variable": dict(formula="y ~ w", data=DATA),
    "a negative response": dict(formula="y ~ x", data=DATA.assign(y=DATA["y"] - DATA["y"].median())),
    "only one positive response": dict(formula="y ~ x", data=DATA.assign(y=np.where(np.arange(NOBS) == 3, 1.0, 0.0))),
    "a missing value": dict(formula="y ~ x", data=DATA.assign(x=np.where(np.arange(NOBS) == 3, np.nan, x))),
    "a missing level": dict(formula="y ~ g", data=DATA.assign(g=DATA["g"].where(np.arange(NOBS) != 3))),
    "an empty interaction cell": dict(formula="y ~ g*k", data=DATA.assign(k=categorical(np.where(g == "A", "r", "s")))),
    "an unknown structure": dict(formula="y ~ x", data=DATA, dtype="erlang"),
    "one quadrature node": dict(formula="y ~ x + (1|group)", data=DATARANDOM, quadnodes=1),
    "an integer tolerance": dict(formula="y ~ x", data=DATA, tolerance=1),
    "zero phases": dict(formula="y ~ x", data=DATA, nphases=0),
}
for name, arguments in infeasible.items():
    if not terminates(**arguments):
        sys.exit("Validation test failed at case 10: %s did not terminate." % name)

# two random effects: refused, with a message saying that one is supported
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    try:
        fitreg(formula="y ~ x + (1|group) + (1|g)", data=DATARANDOM.assign(g=categorical(np.arange(groups.size) % 7)))
        sys.exit("Validation test failed at case 10: two random effects were accepted.")
    except SystemExit as stop:
        if stop.code != 1:
            raise
if "supports one random effect" not in buffer.getvalue() or "(1|group), (1|g)" not in buffer.getvalue():
    sys.exit("Validation test failed at case 10: the refusal of two random effects does not say that one is supported.")

# arguments of earlier versions are gone, so passing one is an error
removed = {"categorical": ["g"], "reference": {"g": "B"}, "inhomogeneity": "weibull"}
for name, value in removed.items():
    try:
        terminates(formula="y ~ x + g", data=DATA, **{name: value})
    except TypeError:
        continue
    sys.exit("Validation test failed at case 10: the removed argument %s was accepted." % name)


# ------------------------------------------------------------------
# CASE 11: summary, plot, AIC, BIC, intervals and verbose
# ------------------------------------------------------------------

buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    # without effect tests, which would refit this three-phase random-effect
    # model; they are checked on the one-phase model below
    modelrandom.summary(effecttest=None)
text = buffer.getvalue()
if "Formula: y ~ x + (1|group)" not in text or "sigma" not in text or "Log-likelihood" not in text:
    sys.exit("Validation test failed at case 11: summary does not report the formula, sigma and the log-likelihood.")

filename = "fitreg_test_plot.png"
try:
    model.plot(confint=True, filename=filename)
    if not os.path.isfile(filename):
        sys.exit("Validation test failed at case 11: plot did not write a file.")
finally:
    if os.path.isfile(filename):
        os.remove(filename)

k = (model.getaic() - (-2.0 * model.getloglik())) / 2.0
if abs(k - round(k)) > TOL or abs(model.getbic() - (-2.0 * model.getloglik() + round(k) * np.log(NOBS))) > TOL:
    sys.exit("Validation test failed at case 11: AIC and BIC do not use the same parameter count.")
# one phase: the rate and the coefficients, nothing else
if (modelexact.getaic() + 2.0 * modelexact.getloglik()) / 2.0 != 1 + len(HANDNAMES):
    sys.exit("Validation test failed at case 11: the one-phase model does not count one rate plus its coefficients.")

intervals = model._getwaldconfint(0.9)
z = norm.ppf(0.95)
for name in coef:
    lower, upper = intervals[name]
    if abs(lower - (coef[name] - z * model.getstderr()[name])) > TOL or abs(upper - (coef[name] + z * model.getstderr()[name])) > TOL:
        sys.exit("Validation test failed at case 11: the confidence interval of %s is not the Wald interval." % name)

# p-values: two-sided Wald tests of each coefficient
for name, p in model._getwaldcoeftest().items():
    if abs(p - 2.0 * norm.sf(abs(coef[name] / model.getstderr()[name]))) > TOL:
        sys.exit("Validation test failed at case 11: the p-value of %s is not that of the Wald test." % name)
# an effect with one coefficient: the square of its z value, with the same p-value
statistic, df, p = model.getwaldtest()["x"]
if df != 1 or abs(statistic - (coef["x"] / model.getstderr()["x"]) ** 2) > TOL * statistic or abs(p - model._getwaldcoeftest()["x"]) > TOL:
    sys.exit("Validation test failed at case 11: the Wald test of a one-coefficient effect is not the square of its z value.")
# effects with several coefficients, against the exact covariance of the
# one-phase model of case 1: the levels of g, and the interaction x:g
tests = modelexact.getwaldtest()
if list(tests) != ["x", "g", "w", "h", "x:g", "x:w", "g:h"]:
    sys.exit("Validation test failed at case 11: the effect tests are %s, not one per effect of the formula." % list(tests))
for term, index in (("g", [1, 2]), ("x:g", [5, 6]), ("g:h", [8, 9])):
    b = np.array([modelexact.getcoef()[HANDNAMES[k]] for k in index])
    hand = float(b @ np.linalg.solve(EXACTCOVARIANCE[np.ix_(index, index)], b))
    statistic, df, p = tests[term]
    if df != 2 or abs(statistic / hand - 1.0) > TOLEXACT or abs(p - chi2.sf(hand, 2)) > TOLEXACT:
        sys.exit("Validation test failed at case 11: the Wald test of %s is %.6f, against %.6f by hand." % (term, statistic, hand))
# the effect table of summary: likelihood-ratio tests of every effect by
# default, Wald tests as an option, none on request; the coefficient table
# has no tests (those are in coeftest)
def termrows(text, header):
    """Returns the rows of the effect table under the given header, as {effect: fields}."""
    lines = text.splitlines()
    if header not in lines:
        return None
    rows = {}
    for line in lines[lines.index(header) + 2:]:
        if not line.strip() or line.startswith("(") or line.startswith("Simulated"):
            break
        fields = line.split()
        rows[fields[0]] = fields[1:]
    return rows

def summarytext(model, **arguments):
    """Returns what summary prints."""
    buffer = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(buffer):
            model.summary(**arguments)
    return buffer.getvalue()

LRHEADER = "Likelihood-ratio tests of the fixed effects:"
WALDHEADER = "Wald tests of the fixed effects:"
ALLTERMS = ["x", "g", "w", "h", "x:g", "x:w", "g:h"]
text = summarytext(modelexact)
rows = termrows(text, LRHEADER)
if rows is None or list(rows) != ALLTERMS or WALDHEADER in text or "Pr(>|z|)" in text:
    sys.exit("Validation test failed at case 11: summary does not report likelihood-ratio tests of every effect, and only those, by default.")
# no individual coefficients in the summary (they are in coeftest), but the
# fit: deviance = -2 log-likelihood, the parameter count by kind, the starts
if any(line.split()[:1] == [name] for line in text.splitlines() for name in HANDNAMES if "[" in name):
    sys.exit("Validation test failed at case 11: summary reports individual coefficients.")
fitline = [line for line in text.splitlines() if line.startswith("Log-likelihood:")]
if len(fitline) != 1 or abs(float(fitline[0].split()[1]) - modelexact.getloglik()) > 5e-5 or \
        abs(float(fitline[0].split()[-1]) + 2.0 * modelexact.getloglik()) > 5e-5:
    sys.exit("Validation test failed at case 11: summary does not report the log-likelihood and the deviance -2 log-likelihood.")
parline = [line for line in text.splitlines() if "Parameters:" in line]
counts = [int(v) for v in parline[0].split("Parameters:")[1].replace("(", " ").replace(",", " ").split() if v.isdigit()] if parline else []
if len(counts) < 3 or counts[0] != modelexact.nparam or sum(counts[1:]) != counts[0] or \
        "1 phase-type, %d coefficients" % len(HANDNAMES) not in parline[0]:
    sys.exit("Validation test failed at case 11: summary does not break the %d parameters down by kind." % modelexact.nparam)
startline = [line for line in text.splitlines() if "Starts:" in line]
if len(startline) != 1:
    sys.exit("Validation test failed at case 11: summary does not report the starts.")
started = int(startline[0].split("Starts:")[1].split(",")[0])
reachedbest = int(startline[0].split("of which")[1].split()[0])
if started != len(modelexact.d.startstatus) or not 1 <= reachedbest <= started:
    sys.exit("Validation test failed at case 11: summary reports %d starts with %d at the best, for %d starts." % (started, reachedbest, len(modelexact.d.startstatus)))

# the likelihood-ratio test of the random effect: twice the gain over the
# fit without (1|u), against 0.5 chi2(0) + 0.5 chi2(1); none at the boundary
modelnorandom, _, _ = quietfit(formula="y ~ x", data=DATARANDOMEXACT, **EXACT)
hand = 2.0 * (modelrandomexact.getloglik() - modelnorandom.getloglik())
statistic, pvalue = modelrandomexact._fitreg__sigmatest()
if abs(statistic - hand) > TOLOPT * max(1.0, hand) or abs(pvalue / (0.5 * chi2.sf(hand, 1)) - 1.0) > TOLOPT * max(1.0, hand):
    sys.exit("Validation test failed at case 11: the test of sigma = 0 has statistic %.8f and p-value %.3e, against %.8f and %.3e from the fit without the random effect." % (statistic, pvalue, hand, 0.5 * chi2.sf(hand, 1)))
text = summarytext(modelrandomexact, effecttest=None)
sigmaline = [line for line in text.splitlines() if "Likelihood-ratio test of sigma = 0" in line]
if len(sigmaline) != 1 or abs(float(sigmaline[0].split("LR stat")[1].split(",")[0]) - hand) > 0.005:
    sys.exit("Validation test failed at case 11: summary does not report the likelihood-ratio test of sigma = 0.")
statistic, pvalue = modelzero._fitreg__sigmatest()
text = summarytext(modelzero, effecttest=None)
if statistic != 0.0 or pvalue != 1.0 or "LR stat 0.00, p-value 1.000" not in text or "at the boundary" not in text:
    sys.exit("Validation test failed at case 11: at the boundary sigma = 0 the test of sigma = 0 is not LR 0 with p-value 1.")

buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    lrexact = modelexact.getlrtest()
for term, fields in rows.items():
    if abs(float(fields[0]) - lrexact[term]["statistic"]) > 0.005 or int(fields[1]) != lrexact[term]["df"]:
        sys.exit("Validation test failed at case 11: the summary's likelihood-ratio test of %s is not that of getlrtest." % term)
rows = termrows(summarytext(modelexact, effecttest="wald"), WALDHEADER)
if rows is None or list(rows) != ALLTERMS or any(abs(float(f[0]) - tests[t][0]) > 0.005 for t, f in rows.items()):
    sys.exit("Validation test failed at case 11: summary(effecttest='wald') does not report the Wald tests of the effects.")
text = summarytext(modelexact, effecttest=None)
if LRHEADER in text or WALDHEADER in text or "Log-likelihood" not in text:
    sys.exit("Validation test failed at case 11: summary(effecttest=None) still reports effect tests, or no longer the rest.")
for arguments in (dict(effecttest="score"), dict(nsim=-1), dict(nsim=2, effecttest="wald"), dict(seed=1.5)):
    text = summarytext(modelexact, **arguments)
    if "Error" not in text or "Formula" in text:
        sys.exit("Validation test failed at case 11: summary accepted the arguments %s." % arguments)

# coeftest: the Wald test, standard error and interval of every coefficient,
# returned and printed
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    table = modelexact.coeftest(confidence=0.9)
printedtable = buffer.getvalue()
intervals = modelexact._getwaldconfint(0.9)
if list(table) != HANDNAMES:
    sys.exit("Validation test failed at case 11: coeftest does not return every coefficient.")
for name, row in table.items():
    se = modelexact.getstderr()[name]
    if abs(row["estimate"] - modelexact.getcoef()[name]) > TOL or abs(row["stderr"] - se) > TOL or \
            abs(row["zvalue"] - row["estimate"] / se) > TOL or abs(row["pvalue"] - 2.0 * norm.sf(abs(row["estimate"] / se))) > TOL or \
            max(abs(row["confint"][k] - intervals[name][k]) for k in (0, 1)) > TOL:
        sys.exit("Validation test failed at case 11: coeftest's row of %s is not the Wald test and interval of the coefficient." % name)
    fields = [line.split() for line in printedtable.splitlines() if line.split()[:1] == [name]]
    if len(fields) != 1 or abs(float(fields[0][1]) - row["estimate"]) > 5e-5 or abs(float(fields[0][-1]) - row["confint"][1]) > 5e-5:
        sys.exit("Validation test failed at case 11: coeftest does not print the row of %s." % name)
if "90% conf. int." not in printedtable or "Pr(>|z|)" not in printedtable:
    sys.exit("Validation test failed at case 11: coeftest does not label the p-values and the intervals at the requested level.")
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    silent = modelexact.coeftest(show=False)
    refused = modelexact.coeftest(confidence=1)
if buffer.getvalue().count("Error") != 1 or refused is not None or silent["x"]["pvalue"] != table["x"]["pvalue"]:
    sys.exit("Validation test failed at case 11: coeftest(show=False) prints, or a bad confidence level is accepted.")

_, printed, _ = quietfit(formula="y ~ x + g", data=DATAEXACT, verbose=False, **EXACT)
_, printedverbose, _ = quietfit(formula="y ~ x + g", data=DATAEXACT, verbose=True, **EXACT)
if printed != "" or printedverbose == "":
    sys.exit("Validation test failed at case 11: verbose does not decide whether anything is printed while fitting.")


# ------------------------------------------------------------------
# CASE 12: Observations at zero
# ------------------------------------------------------------------

# The model is the mixture p (zero) + (1 - p) (phase-type regression), whose
# likelihood factorizes: p is the share of zeros, the rest is the fit to the
# positive observations, and the zeros add n0 log p + n+ log(1 - p)
iszero = np.arange(NEXACT) % 10 == 3
withzeros = DATAEXACT.assign(y=np.where(iszero, 0.0, DATAEXACT["y"]))
modelzeros, _, _ = quietfit(formula=FORMULAEXACT, data=withzeros, **EXACT)
modelpositive, _, _ = quietfit(formula=FORMULAEXACT, data=withzeros[~iszero], **EXACT)
p0 = iszero.mean()
n0, npositive = int(iszero.sum()), int((~iszero).sum())

if modelzeros.getzeroprob() != p0 or modelexact.getzeroprob() != 0.0:
    sys.exit("Validation test failed at case 12: the probability of a zero is %s, not the share of zeros %s." % (modelzeros.getzeroprob(), p0))
if max(abs(modelzeros.getcoef()[k] - modelpositive.getcoef()[k]) for k in HANDNAMES) > TOL or \
        max(abs(modelzeros.getstderr()[k] - modelpositive.getstderr()[k]) for k in HANDNAMES) > TOL:
    sys.exit("Validation test failed at case 12: the regression is not the fit to the positive observations.")
loglik = modelpositive.getloglik() + n0 * np.log(p0) + npositive * np.log(1.0 - p0)
if abs(modelzeros.getloglik() - loglik) > TOL * abs(loglik):
    sys.exit("Validation test failed at case 12: the log-likelihood %.10f leaves out the point mass (%.10f)." % (modelzeros.getloglik(), loglik))
k = modelpositive.nparam + 1
if modelzeros.nparam != k or abs(modelzeros.getaic() - (-2.0 * loglik + 2.0 * k)) > TOL * abs(loglik) or \
        abs(modelzeros.getbic() - (-2.0 * loglik + k * np.log(NEXACT))) > TOL * abs(loglik):
    sys.exit("Validation test failed at case 12: the point mass is not counted as a parameter, or BIC does not use every observation.")
if abs(np.sum(modelzeros.getinitdist()) - (1.0 - p0)) > TOL:
    sys.exit("Validation test failed at case 12: the initial distribution does not leave the point mass at zero.")
point = {"x": 0.5, "w": 0.2, "g": "b", "h": "q"}
if modelzeros.getcumprob(-1.0, point) != 0.0 or abs(modelzeros.getcumprob(0.0, point) - p0) > TOL:
    sys.exit("Validation test failed at case 12: the CDF is not 0 below zero and p at zero.")
for v in (0.2, 1.0, 4.0):
    if abs(modelzeros.getcumprob(v, point) - (p0 + (1.0 - p0) * modelpositive.getcumprob(v, point))) > TOL or \
            abs(modelzeros.getdensity(v, point) - (1.0 - p0) * modelpositive.getdensity(v, point)) > TOL:
        sys.exit("Validation test failed at case 12: the CDF or density is not the mixture with the point mass.")
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    modelzeros.summary(effecttest=None)
if "Point mass at zero" not in buffer.getvalue() or "(%d at zero)" % n0 not in buffer.getvalue() or \
        "1 zero share" not in buffer.getvalue():
    sys.exit("Validation test failed at case 12: summary does not report the point mass at zero.")

# the same with a random effect
iszero = np.arange(groupsexact.size) % 9 == 4
randomzeros = DATARANDOMEXACT.assign(y=np.where(iszero, 0.0, DATARANDOMEXACT["y"]))
modelzeros, _, _ = quietfit(formula="y ~ x + (1|u)", data=randomzeros, **EXACT)
modelpositive, _, _ = quietfit(formula="y ~ x + (1|u)", data=randomzeros[~iszero], **EXACT)
p0 = iszero.mean()
loglik = modelpositive.getloglik() + iszero.sum() * np.log(p0) + (~iszero).sum() * np.log(1.0 - p0)
if abs(modelzeros.getloglik() - loglik) > TOL * abs(loglik) or abs(modelzeros.getsigma() - modelpositive.getsigma()) > TOL:
    sys.exit("Validation test failed at case 12: with a random effect the zeros are not a point mass beside the fit to the positive observations.")


# ------------------------------------------------------------------
# CASE 13: Likelihood-ratio tests
# ------------------------------------------------------------------

def exponentialloglik(X, response):
    """
    The maximized log-likelihood of exponential regression of the response on
    the columns X (and an intercept), from a log-link Gamma GLM whose
    estimating equations are those of exponential regression.
    """
    full = np.column_stack([np.ones(response.size), X])
    fitted = sm.GLM(response, full, family=sm.families.Gamma(link=sm.families.links.Log())).fit(tol=1e-14, maxiter=1000)
    mean = np.exp(full @ fitted.params)
    return float(np.sum(-np.log(mean) - response / mean))


def quiet(function, *arguments, **keywords):
    """Calls a method without letting it print, returning its result and what it printed."""
    buffer = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(buffer):
            result = function(*arguments, **keywords)
    return result, buffer.getvalue()


# exact: the columns of an effect are dropped, whether or not the effect is part of
# an interaction (w is part of x:w, g of x:g and g:h)
lrtests, printed = quiet(modelexact.getlrtest, ["w", "g", "x:g"])
if printed != "":
    sys.exit("Validation test failed at case 13: getlrtest printed '%s' for a fit at its maximum." % printed.strip())
fullloglik = exponentialloglik(HANDX, ye)
for term, index in (("w", [3]), ("g", [1, 2]), ("x:g", [5, 6])):
    keep = [k for k in range(len(HANDNAMES)) if k not in index]
    hand = 2.0 * (fullloglik - exponentialloglik(HANDX[:, keep], ye))
    test = lrtests[term]
    if test["df"] != len(index) or abs(test["statistic"] - hand) > TOLEXACT * max(1.0, hand):
        sys.exit("Validation test failed at case 13: the likelihood-ratio statistic of %s is %.10f (%d df), against %.10f (%d df) by hand." % (term, test["statistic"], test["df"], hand, len(index)))
    if abs(test["pvalue"] - chi2.sf(test["statistic"], test["df"])) > TOL or "simpvalue" in test:
        sys.exit("Validation test failed at case 13: the p-value of %s is not the chi-square tail, or a simulated p-value was reported without simulations." % term)
if list(quiet(modelexact.getlrtest)[0]) != list(modelexact.getwaldtest()):
    sys.exit("Validation test failed at case 13: getlrtest does not test every effect of the formula by default.")

# zeros: the point mass contributes the same factor with and without an effect
iszero = np.arange(NEXACT) % 10 == 3
withzeros = DATAEXACT.assign(y=np.where(iszero, 0.0, DATAEXACT["y"]))
modelzeros, _, _ = quietfit(formula=FORMULAEXACT, data=withzeros, **EXACT)
modelpositive, _, _ = quietfit(formula=FORMULAEXACT, data=withzeros[~iszero], **EXACT)
# (a statistic computed from the log-likelihood with the point mass would
# make the model without the effect look better than the fit, which getlrtest
# would report as a local maximum; it must print nothing)
(zerotest, printed), (positivetest, _) = quiet(modelzeros.getlrtest, "x:w"), quiet(modelpositive.getlrtest, "x:w")
zerotest, positivetest = zerotest["x:w"], positivetest["x:w"]
if printed != "":
    sys.exit("Validation test failed at case 13: with zeros getlrtest printed '%s'." % printed.strip())
if abs(zerotest["statistic"] - positivetest["statistic"]) > TOL * max(1.0, positivetest["statistic"]):
    sys.exit("Validation test failed at case 13: the likelihood-ratio statistic changes with the zeros (%.10f against %.10f without them)." % (zerotest["statistic"], positivetest["statistic"]))

# a random effect, and a model left without covariates: the statistic of
# x is twice the gain in the marginal log-likelihood over the fit of y ~ (1|u)
modelonly, _, _ = quietfit(formula="y ~ (1|u)", data=DATARANDOMEXACT, quadnodes=15, **EXACT)
hand = 2.0 * (modelrandomexact.getloglik() - modelonly.getloglik())
test = quiet(modelrandomexact.getlrtest, "x")[0]["x"]
if abs(test["statistic"] - hand) > TOLOPT * max(1.0, hand):
    sys.exit("Validation test failed at case 13: with a random effect the statistic of x is %.8f, against %.8f from the fit without x." % (test["statistic"], hand))

# three phases: the model without an effect never ends above the full model
for term, test in quiet(model.getlrtest)[0].items():
    if not test["statistic"] >= 0.0 or not 0.0 <= test["pvalue"] <= 1.0:
        sys.exit("Validation test failed at case 13: the three-phase test of %s has statistic %s and p-value %s." % (term, test["statistic"], test["pvalue"]))

# the simulated p-value, on one-phase data with a moderate effect of w
lrrng = np.random.default_rng([SEED, 13])
xl = lrrng.normal(size=NLR)
wl = lrrng.uniform(-1.0, 1.0, size=NLR)
yl = lrrng.exponential(size=NLR) / (EXACTRATE * np.exp(EXACTBETA["x"] * xl + LREFFECT * wl))
modellr, _, _ = quietfit(formula="y ~ x + w", data=pd.DataFrame({"y": yl, "x": xl, "w": wl}),
                         nphases=1, restarts=1, seed=SEED, tolerance=TOLFIT)
levellr = ALPHALR / NCHECKSLR
confidence = 1.0 - levellr
test = quiet(modellr.getlrtest, "w", nsim=NSIMLR, seed=SEED, confidence=confidence)[0]["w"]
if test["nsim"] != NSIMLR:
    sys.exit("Validation test failed at case 13: %d of %d simulated data sets were fitted." % (test["nsim"], NSIMLR))
k = test["simpvalue"] * (NSIMLR + 1) - 1
if abs(k - round(k)) > 1e-9 or not 0 <= round(k) <= NSIMLR:
    sys.exit("Validation test failed at case 13: the simulated p-value %.6f is not (1 + k) / (nsim + 1)." % test["simpvalue"])
k = int(round(k))
lower = 0.0 if k == 0 else beta_dist.ppf(levellr / 2.0, k, NSIMLR - k + 1)
upper = 1.0 if k == NSIMLR else beta_dist.ppf(1.0 - levellr / 2.0, k + 1, NSIMLR - k)
if abs(test["siminterval"][0] - lower) > TOL or abs(test["siminterval"][1] - upper) > TOL:
    sys.exit("Validation test failed at case 13: the Monte Carlo interval is not the Clopper-Pearson interval at the requested level.")
if not lower <= test["pvalue"] <= upper:
    sys.exit("Validation test failed at case 13: the chi-square p-value %.4f of w lies outside the Monte Carlo interval [%.4f, %.4f] of the simulated p-value." % (test["pvalue"], lower, upper))
again = quiet(modellr.getlrtest, "w", nsim=3, seed=SEED + 1)[0]["w"]
twice = quiet(modellr.getlrtest, "w", nsim=3, seed=SEED + 1)[0]["w"]
if again["simpvalue"] != twice["simpvalue"] or again["statistic"] != test["statistic"]:
    sys.exit("Validation test failed at case 13: the simulated p-value is not reproducible with its seed.")

# the Monte Carlo interval by hand: Clopper-Pearson, at both ends and in
# between (the test above has k = 0 for most seeds, which leaves the lower
# limit untested). Exact values from the closed forms of the beta quantiles:
# k = 0 gives (0, 1 - (alpha/2)^(1/m)), k = m gives ((alpha/2)^(1/m), 1).
for k, m, level in ((0, 20, 0.9), (20, 20, 0.9), (3, 20, 0.95), (7, 11, 0.99)):
    lower, upper = fitreg._fitreg__clopperpearson(k, m, level)
    a = 1.0 - level
    handlower = 0.0 if k == 0 else (a / 2.0) ** (1.0 / m) if k == m else beta_dist.ppf(a / 2.0, k, m - k + 1)
    handupper = 1.0 if k == m else 1.0 - (a / 2.0) ** (1.0 / m) if k == 0 else beta_dist.ppf(1.0 - a / 2.0, k + 1, m - k)
    if abs(lower - handlower) > TOL or abs(upper - handupper) > TOL:
        sys.exit("Validation test failed at case 13: the Clopper-Pearson interval for %d of %d at %.2f is (%.6f, %.6f), against (%.6f, %.6f)." % (k, m, level, lower, upper, handlower, handupper))
# a binomial count inside: k = 3 of 20 at 95% is (0.0321, 0.3789) by the
# tables of the exact interval
lower, upper = fitreg._fitreg__clopperpearson(3, 20, 0.95)
if abs(lower - 0.0321) > 5e-5 or abs(upper - 0.3789) > 5e-5:
    sys.exit("Validation test failed at case 13: the Clopper-Pearson interval for 3 of 20 is (%.4f, %.4f), not (0.0321, 0.3789)." % (lower, upper))

# the simulator of absorption times against the true Coxian, built from the
# raw rates: its mean (t interval) and its distribution (chi-square test on
# bins of equal true probability)
truet = -np.diag(TRUERATES)
for i in range(TRUERATES.size - 1):
    truet[i, i + 1] = TRUECONTINUE[i] * TRUERATES[i]
truepi = np.eye(TRUERATES.size)[0]
draws = fitreg._fitreg__simulateph(truepi, truet, NDRAWS, np.random.default_rng([SEED, 131]))
truemean = float(truepi @ np.linalg.solve(-truet, np.ones(TRUERATES.size)))
half = t_dist.ppf(1.0 - levellr / 2.0, NDRAWS - 1) * np.std(draws, ddof=1) / np.sqrt(NDRAWS)
if abs(np.mean(draws) - truemean) > half:
    sys.exit("Validation test failed at case 13: the simulated absorption times have mean %.4f, against %.4f (half-width %.4f)." % (np.mean(draws), truemean, half))
truecdf = lambda v: 1.0 - truepi @ expm(truet * v) @ np.ones(TRUERATES.size)
edges = [0.0]
for q in np.arange(1, NBINSLR) / NBINSLR:
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if truecdf(mid) < q else (lo, mid)
    edges.append((lo + hi) / 2.0)
counts = np.histogram(draws, bins=np.r_[edges, np.inf])[0]
expected = NDRAWS / NBINSLR
if expected < 5 or NBINSLR - 1 < 2:
    sys.exit("Validation test failed at case 13: the chi-square test of the simulator has too few expected counts or degrees of freedom.")
statistic = float(np.sum((counts - expected) ** 2 / expected))
if statistic > chi2.ppf(1.0 - levellr, NBINSLR - 1):
    sys.exit("Validation test failed at case 13: the simulated absorption times do not follow the Coxian (chi-square %.2f on %d df)." % (statistic, NBINSLR - 1))

# the simulated responses of a one-phase random-effect model: z = log(y) +
# eta0 = log(U) - b, U exponential with rate RATE and b ~ N(0, SIGMA^2) per
# group, so z has mean -gamma - log(RATE), and the group means of z have
# variance SIGMA^2 + (pi^2 / 6) / (group size)
simrate, simsigma = 0.7, 0.9
order = modelrandomexact.order
simgroups = modelrandomexact._fitreg__groups
simeta = 0.5 + 0.8 * DATARANDOMEXACT["x"].to_numpy()[order]
simrng = np.random.default_rng([SEED, 132])
means, variances = [], []
for _ in range(NREPLR):
    z = np.log(modelrandomexact._fitreg__simulatenull(dict(pi=np.ones(1), T=-simrate * np.ones((1, 1))),
                                                       simeta, simsigma, simrng)) + simeta
    groupmeans = np.bincount(simgroups, weights=z) / np.bincount(simgroups)
    means.append(np.mean(z))
    variances.append(np.var(groupmeans, ddof=1))
for label, values, theory in (("mean of log(y) + eta", means, -np.euler_gamma - np.log(simrate)),
                              ("variance of the group means", variances, simsigma ** 2 + (np.pi ** 2 / 6.0) / GROUPSIZEEXACT)):
    half = t_dist.ppf(1.0 - levellr / 2.0, NREPLR - 1) * np.std(values, ddof=1) / np.sqrt(NREPLR)
    if abs(np.mean(values) - theory) > half:
        sys.exit("Validation test failed at case 13: the simulated responses have %s %.4f, against %.4f (half-width %.4f)." % (label, np.mean(values), theory, half))

# the simulated p-value in the summary is that of getlrtest with the same seed
modelsummary, _, _ = quietfit(formula="y ~ x + g", data=DATAEXACT, nphases=1, restarts=1, seed=SEED, tolerance=TOLFIT)
rows = termrows(summarytext(modelsummary, nsim=3, seed=SEED), LRHEADER)
direct = quiet(modelsummary.getlrtest, "g", nsim=3, seed=SEED)[0]["g"]
if rows is None or len(rows["g"]) != 6 or abs(float(rows["g"][3]) - direct["simpvalue"]) > 5e-5 or \
        abs(float(rows["g"][4]) - direct["siminterval"][0]) > 5e-5 or abs(float(rows["g"][5]) - direct["siminterval"][1]) > 5e-5:
    sys.exit("Validation test failed at case 13: the summary's simulated p-value is not that of getlrtest with the same seed.")

# bad arguments are refused with a message and no result
for arguments in (dict(effects="z"), dict(effects=["x", 3]), dict(effects=5), dict(nsim=-1), dict(nsim=2.5),
                  dict(seed="a"), dict(confidence=1.0), dict(confidence=1)):
    result, printed = quiet(modellr.getlrtest, **arguments)
    if result is not None or "Error" not in printed:
        sys.exit("Validation test failed at case 13: getlrtest accepted the arguments %s." % arguments)


print("fitreg: All tests passed.")
