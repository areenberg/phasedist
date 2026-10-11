'''
UNIT TEST FOR THE fitreg CLASS (the user-facing interface that fits a
continuous-time phase-type regression with fixed effects, interactions and an
optional Gaussian random intercept, specified by an R-style formula on a pandas
DataFrame).

The first two cases are exact ground truths: with one phase the model is a
classical regression whose maximum likelihood estimates can be computed by
other means, so the class must reproduce them to optimizer precision. The
oracles are independent of the class: statsmodels' Gamma GLM on a design matrix
built by hand, and a direct maximization of the random-intercept likelihood
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
             categorical and interaction terms, the coefficients with the sign
             flipped, and the standard errors equal the exact observed
             information of exponential regression.
    Case 2:  With one phase and a random intercept, sigma, the baseline rate,
             the coefficient and the marginal log-likelihood equal a direct
             maximization of the marginal likelihood computed by quadrature,
             and the standard errors equal its inverse Hessian.
    Case 3:  Interactions: '*' expands to the main effects and interactions,
             repeated terms are kept once (x:z and z:x being one term), an
             interaction without its lower-order terms is refused, the same
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
    Case 7:  A random intercept on three-phase data: sigma is recovered and
             every group gets a predicted intercept. When the groups vary less
             than chance would make them, sigma is exactly zero and the fit is
             that of the model without random intercept.
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
             intervals, the p-values and the term tests are Wald tests (the
             latter checked against the exact covariance of case 1), and
             verbose decides whether anything is printed.

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
from scipy.stats import chi2, norm

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
NGROUPS = 40         # groups of the multi-phase random-intercept data
GROUPSIZE = 10
NGROUPSEXACT = 20    # groups of the one-phase random-intercept data of case 2
GROUPSIZEEXACT = 6
SHIFT = 10.0         # shift of the continuous covariate in case 6

# The multi-phase truth: a Coxian of three phases with distinct rates, so the
# phases are not exchangeable, and effects large enough that their signs are
# clear at NOBS observations.
TRUERATES = np.array([3.0, 1.0, 0.4])
TRUECONTINUE = np.array([0.6, 0.5])
TRUEBETA = {"x": 0.8, "g[B]": -0.6, "g[C]": 0.7}

# The random intercept must be identifiable from the dependence within groups,
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
COVEREDMETHODS = {"getcoef", "getstderr", "getconfint", "getpvalue", "getwaldtest", "getsigma",
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

# The one-phase random-intercept data of case 2
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

# The multi-phase random-intercept data
groups = np.repeat(np.arange(NGROUPS), GROUPSIZE)
xr = rng.normal(size=groups.size)
groupintercepts = rng.normal(scale=TRUESIGMA, size=NGROUPS)
yr = simulatecoxian(groups.size, rng) * np.exp(-(TRUEBETA["x"] * xr + groupintercepts[groups]))
DATARANDOM = pd.DataFrame({"y": yr, "x": xr, "group": categorical(groups)})

# The same design without a random intercept, for the boundary of case 7
# Data whose groups vary less than chance would make them, for the boundary of
# case 7: the observations are dealt to the groups in the order of their
# baseline times, so that every group gets a similar mix of short and long
# times. The derivative of the likelihood in sigma^2 at zero is then clearly
# negative and the estimate of sigma is exactly zero. (With groups drawn at
# random and no random intercept, about half the data sets would give zero.)
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
# CASE 2: One phase with a random intercept against a quadrature likelihood
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
    random intercept, (log rate, beta, log sigma).
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
    sys.exit("Validation test failed at case 2: the random-intercept estimates differ from a direct maximization of the quadrature likelihood by %.3e." % np.max(np.abs(fitted - oracle.x)))
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
    sys.exit("Validation test failed at case 3: an interaction without its lower-order term was accepted.")
except ValueError as error:
    if "x*g" not in str(error):
        sys.exit("Validation test failed at case 3: the refusal of an incomplete interaction does not say how to write it.")

# x:z and z:x are one term: within a formula a repeated interaction is kept
# once whatever the order of its variables, and it is named in the order the
# variables first appear, as in R
for formula in ("y ~ x*z + z:x", "y ~ x + z + z:x", "y ~ x + z + x:z + z:x"):
    if _regformula(formula).terms != [("x",), ("z",), ("x", "z")]:
        sys.exit("Validation test failed at case 3: '%s' was parsed as %s." % (formula, _regformula(formula).terms))
if _regformula("y ~ z*x").terms != [("z",), ("x",), ("z", "x")]:
    sys.exit("Validation test failed at case 3: 'z*x' does not keep the order it is written in.")

# and the same model written in another order is the same fit: the same
# log-likelihood, coefficients, standard errors and term tests, the names
# only following the order of the formula. The comparison goes by the set of
# variables and levels in each name, so that 'x:g[b]' matches 'g[b]:x'
modelreordered, _, _ = quietfit(formula="y ~ h*g + w*x + g*x", data=DATAEXACT, **EXACT)


def byparts(values):
    """Keys a dictionary by the set of parts of each name."""
    return {frozenset(name.split(":")): value for name, value in values.items()}


if abs(modelreordered.getloglik() - modelexact.getloglik()) > TOLEXACT * abs(modelexact.getloglik()):
    sys.exit("Validation test failed at case 3: writing the interactions in another order changed the log-likelihood.")
for getter in ("getcoef", "getstderr", "getpvalue"):
    original, reordered = byparts(getattr(modelexact, getter)()), byparts(getattr(modelreordered, getter)())
    if set(original) != set(reordered) or max(abs(original[k] - reordered[k]) for k in original) > TOLEXACT:
        sys.exit("Validation test failed at case 3: writing the interactions in another order changed %s." % getter)
original = byparts(modelexact.getwaldtest())
reordered = byparts(modelreordered.getwaldtest())
if set(original) != set(reordered) or max(abs(original[k][0] - reordered[k][0]) for k in original) > TOLEXACT * 100:
    sys.exit("Validation test failed at case 3: writing the interactions in another order changed the term tests.")

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
    sys.exit("Validation test failed at case 4: a model without random intercept reports a sigma or random intercepts.")


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
# CASE 7: A random intercept on three-phase data
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
# without random intercept: the same coefficients, standard errors and
# log-likelihood, every predicted intercept zero, a warning, and no standard
# error for sigma
modelzero, _, caughtzero = quietfit(formula="y ~ x + (1|group)", data=DATANOINTERCEPT, **COMMON)
modelnogroup, _, _ = quietfit(formula="y ~ x", data=DATANOINTERCEPT, **COMMON)
if not modelzero.boundary or modelzero.getsigma() != 0.0:
    sys.exit("Validation test failed at case 7: sigma is %.3g, not zero at the boundary." % modelzero.getsigma())
if not any("standard deviation of the random intercept is zero" in str(w.message) for w in caughtzero):
    sys.exit("Validation test failed at case 7: the boundary was reached without a warning.")
if modelzero.sigmastderr is None or np.isfinite(modelzero.sigmastderr):
    sys.exit("Validation test failed at case 7: a standard error is reported for sigma at the boundary.")
# (the class orders the observations by group, so the two fits see the data
# in a different order and agree to TOLRUN rather than to rounding)
if abs(modelzero.getloglik() - modelnogroup.getloglik()) > TOLRUN or \
        abs(modelzero.getcoef()["x"] - modelnogroup.getcoef()["x"]) > TOLRUN or \
        abs(modelzero.getstderr()["x"] - modelnogroup.getstderr()["x"]) > TOLRUN:
    sys.exit("Validation test failed at case 7: the fit at the boundary is not that of the model without random intercept.")
if any(v != 0.0 for v in modelzero.getranef().values()):
    sys.exit("Validation test failed at case 7: a predicted intercept is not zero at the boundary.")
# one parameter more than the model without random intercept, as in lme4
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
    "a non-positive response": dict(formula="y ~ x", data=DATA.assign(y=DATA["y"] - DATA["y"].max())),
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
    modelrandom.summary()
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

intervals = model.getconfint(0.9)
z = norm.ppf(0.95)
for name in coef:
    lower, upper = intervals[name]
    if abs(lower - (coef[name] - z * model.getstderr()[name])) > TOL or abs(upper - (coef[name] + z * model.getstderr()[name])) > TOL:
        sys.exit("Validation test failed at case 11: the confidence interval of %s is not the Wald interval." % name)

# p-values: two-sided Wald tests of each coefficient
for name, p in model.getpvalue().items():
    if abs(p - 2.0 * norm.sf(abs(coef[name] / model.getstderr()[name]))) > TOL:
        sys.exit("Validation test failed at case 11: the p-value of %s is not that of the Wald test." % name)
# a term with one coefficient: the square of its z value, with the same p-value
statistic, df, p = model.getwaldtest()["x"]
if df != 1 or abs(statistic - (coef["x"] / model.getstderr()["x"]) ** 2) > TOL * statistic or abs(p - model.getpvalue()["x"]) > TOL:
    sys.exit("Validation test failed at case 11: the Wald test of a one-coefficient term is not the square of its z value.")
# terms with several coefficients, against the exact covariance of the
# one-phase model of case 1: the levels of g, and the interaction x:g
tests = modelexact.getwaldtest()
if list(tests) != ["x", "g", "w", "h", "x:g", "x:w", "g:h"]:
    sys.exit("Validation test failed at case 11: the term tests are %s, not one per term of the formula." % list(tests))
for term, index in (("g", [1, 2]), ("x:g", [5, 6]), ("g:h", [8, 9])):
    b = np.array([modelexact.getcoef()[HANDNAMES[k]] for k in index])
    hand = float(b @ np.linalg.solve(EXACTCOVARIANCE[np.ix_(index, index)], b))
    statistic, df, p = tests[term]
    if df != 2 or abs(statistic / hand - 1.0) > TOLEXACT or abs(p - chi2.sf(hand, 2)) > TOLEXACT:
        sys.exit("Validation test failed at case 11: the Wald test of %s is %.6f, against %.6f by hand." % (term, statistic, hand))
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    modelexact.summary()
if "Wald tests of the terms" not in buffer.getvalue() or "x:g" not in buffer.getvalue():
    sys.exit("Validation test failed at case 11: summary does not report the tests of the terms with several coefficients.")

_, printed, _ = quietfit(formula="y ~ x + g", data=DATAEXACT, verbose=False, **EXACT)
_, printedverbose, _ = quietfit(formula="y ~ x + g", data=DATAEXACT, verbose=True, **EXACT)
if printed != "" or printedverbose == "":
    sys.exit("Validation test failed at case 11: verbose does not decide whether anything is printed while fitting.")


print("fitreg: All tests passed.")
