'''
UNIT TEST FOR THE fitreg CLASS (phase-type regression from an R-style formula
on a pandas DataFrame). Statistical
validation (coverage over many replicates) is kept outside the repository.

Data are simulated from the model: one-phase truths for the exact oracles,
and a three-phase Coxian with distinct rates, built from its raw rates
without the class, for the smoke tests.

Sub-tests:
    Case 1:  One phase equals a log-link Gamma GLM (statsmodels): coefficients,
             rate, log-likelihood; standard errors equal the exact information.
    Case 2:  One phase with a random effect equals a direct maximization of
             a quadrature likelihood, standard errors included.
    Case 3:  Interactions: '*' expansion, x:z = z:x, refusal of incomplete
             three-way interactions, order invariance, design columns.
    Case 4:  A three-phase fit recovers the effects; its standard errors are
             those of the information with the phase-type parameters
             profiled out (an independent expm likelihood).
    Case 5:  Variable types from the dtypes; reference level; refusals.
    Case 6:  Shifting a covariate leaves the fit unchanged.
    Case 7:  Random effect on three-phase data; sigma = 0 at the boundary.
    Case 8:  Predictions (getdist, getdensity, getcumprob).
    Case 9:  The formula parser accepts and refuses with messages.
    Case 10: Infeasible input terminates.
    Case 11: summary, plot, AIC/BIC, Wald intervals and tests, coeftest,
             verbose.
    Case 12: Zeros are a point mass; the summary notes the assumption.
    Case 13: getlrtest: exact statistics, zeros, random effect, simulated
             p-values and their simulators (statistical tests).
    Case 14: A slope of x per level of g ('x:g', 'x + x:g'): Gamma GLM,
             reference-level invariance, origin dependence, refusals.
    Case 15: Right-censoring, one phase: Poisson GLM; full-information
             standard errors; likelihood-ratio test.
    Case 16: All kinds of censoring, one phase: concave direct maximization.
    Case 17: Empty censoring columns give the uncensored fit bit for bit;
             three-phase censored fit; the engine's censored rows against an
             independent expm evaluation.
    Case 18: Censoring with a random effect: quadrature likelihood.
    Case 19: Censoring refusals, zeros, simulated p-values, plot.
    Case 20: The score test of the random effect against a hand computation
             with dense matrices, uncensored and censored; the
             likelihood-ratio alternative; refusals.

References:
    Bladt, M. (2022). Phase-type distributions for claim severity regression
    modeling. ASTIN Bulletin, 52(2), 417-448.
'''

import contextlib
import io
import os
import sys
import warnings

import numpy as np
import matplotlib

# no display needed: case 11 only writes the figure to file
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
from phasedist.regtools import _quadform

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

TOLOPT = 1e-5        # against a numerical optimizer (case 2; measured 1.2e-7)

TOLRUN = 1e-4        # two runs differing by rounding (case 6; measured 2.3e-6)

STEP = 1e-4          # finite-difference step of the oracles' Hessians
TOLSCORE = 1e-8      # score test of sigma = 0 against dense matrices, relative (measured 1e-14
                     # for z; the p-values to the integral's accuracy, 1e-11 absolute)
TOLSE = 1e-4         # standard errors, relative: the class takes them from a
                     # finite-difference Hessian (measured 3.3e-7 or better)

SEED = 0
NOBS = 300           # observations of the multi-phase fixed-effects data
NEXACT = 400         # observations of the one-phase data of case 1
NGROUPS = 40         # groups of the multi-phase random-effect data
GROUPSIZE = 10
NGROUPSEXACT = 20    # groups of the one-phase random-effect data of case 2
GROUPSIZEEXACT = 6
SHIFT = 10.0         # shift of the continuous covariate in case 6

# Case 13: the chi-square p-value (accurate for one phase) must lie in the
# Clopper-Pearson interval of the simulated one; NCHECKSLR statistical checks
# share the level ALPHALR (Bonferroni).
NLR = 200
NSIMLR = 40
LREFFECT = 0.35
NDRAWS = 20000
NBINSLR = 10
NREPLR = 400
ALPHALR = 0.01
NCHECKSLR = 5

# Case 14: distinct slopes per level; shifting x by ORIGINSHIFT changed the
# log-likelihood by 77.6-137.1 over seeds 0-29 (threshold ORIGINGAP).
SLOPES = {"a": 0.3, "b": -0.5, "c": 0.8}
SLOPEMEAN = 1.0
ORIGINSHIFT = -3.0
ORIGINGAP = 10.0

# Cases 15-19: limits independent of the times: a uniform right-censoring
# limit, or inspections on CENSGRID (exact, right, inspected: CENSSCHEME).
CENSRIGHT = (0.5, 6.0)
CENSGRID = np.array([0.0, 0.5, 1.0, 2.0, 4.0, np.inf])
CENSSCHEME = (0.4, 0.3, 0.3)

# The multi-phase truth: distinct rates, so the phases are not exchangeable
TRUERATES = np.array([3.0, 1.0, 0.4])
TRUECONTINUE = np.array([0.6, 0.5])
TRUEBETA = {"x": 0.8, "g[B]": -0.6, "g[C]": 0.7}

TRUESIGMA = 1.0      # within-group correlation of log Y about 0.3: identifiable

# The one-phase truths of cases 1 and 2
EXACTRATE = 0.5
EXACTBETA = dict(x=0.6, gb=-0.4, gc=0.5, w=0.3, hq=0.3, xgb=0.4, xgc=-0.3, xw=0.5, gbhq=0.2, gchq=-0.2)
EXACTSIGMA = 0.8

# every public method of fitreg, checked against the class
COVEREDMETHODS = {"getcoef", "getstderr", "coeftest", "getwaldtest", "getlrtest", "getsigma", "getsigmatest", "getzeroprob",
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
# the design of 'y ~ x*g + w + x:w + h + g:h', built by hand in the order and with
# the names the class is documented to use
HANDNAMES = ["x", "g[b]", "g[c]", "w", "h[q]", "x:g[b]", "x:g[c]", "x:w", "g[b]:h[q]", "g[c]:h[q]"]
HANDX = np.column_stack([xe, gb, gc, we, hq, xe * gb, xe * gc, xe * we, gb * hq, gc * hq])
HANDBETA = np.array([EXACTBETA[k] for k in ("x", "gb", "gc", "w", "hq", "xgb", "xgc", "xw", "gbhq", "gchq")])
ye = rng.exponential(size=NEXACT) / (EXACTRATE * np.exp(HANDX @ HANDBETA))
DATAEXACT = pd.DataFrame({"y": ye, "x": xe, "w": we, "g": categorical(ge), "h": categorical(he)})
FORMULAEXACT = "y ~ x*g + w + x:w + h + g:h"

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

# Case 7 boundary data: rows dealt to groups by their baseline times, so the
# groups vary less than chance and sigma-hat is exactly zero
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
if np.max(np.abs(reported / stderr - 1.0)) > TOLSE:
    sys.exit("Validation test failed at case 1: the standard errors differ from the observed information of exponential regression by a factor of up to %.3e." % np.max(np.abs(reported / stderr - 1.0)))


# ------------------------------------------------------------------
# CASE 2: One phase with a random effect against a quadrature likelihood
# ------------------------------------------------------------------

modelrandomexact, _, _ = quietfit(formula="y ~ x + (1|u)", data=DATARANDOMEXACT, quadnodes=15, **EXACT)


# A group's integrand is closed form, a_j + n_j b - exp(b) s_j; each integral by
# the trapezoidal rule on a fine fixed grid (independent of the class's AGHQ)
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

# standard errors: the inverse Hessian of the oracle, rate included
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

parsed = _regformula("y ~ x*w*g + (1|u)")
expected = [("x",), ("w",), ("g",), ("x", "w"), ("x", "g"), ("w", "g"), ("x", "w", "g")]
if parsed.terms != expected or parsed.group != "u":
    sys.exit("Validation test failed at case 3: 'x*w*g' was parsed as %s, not as every main effect and interaction once." % parsed.terms)
# an interaction of three variables needs all its lower-order effects (those
# of two variables are checked by the design, which knows the types: case 14)
try:
    _regformula("y ~ x + w + g + x:w + x:g + x:w:g")
    sys.exit("Validation test failed at case 3: an interaction without its lower-order effect was accepted.")
except ValueError as error:
    if "x*w*g" not in str(error) or "'w:g'" not in str(error):
        sys.exit("Validation test failed at case 3: the refusal of an incomplete interaction does not say what is missing and how to write it.")

# x:z and z:x are one effect, named in the order of first appearance; an
# effect given twice is refused (test_regformula covers the details)
if _regformula("y ~ x + z + z:x").terms != [("x",), ("z",), ("x", "z")]:
    sys.exit("Validation test failed at case 3: 'x + z + z:x' was parsed as %s." % _regformula("y ~ x + z + z:x").terms)
for formula in ("y ~ x*z + z:x", "y ~ x + z + x:z + z:x"):
    try:
        _regformula(formula)
        sys.exit("Validation test failed at case 3: '%s' gives x:z twice but was accepted." % formula)
    except ValueError:
        pass
if _regformula("y ~ z*x").terms != [("z",), ("x",), ("z", "x")]:
    sys.exit("Validation test failed at case 3: 'z*x' does not keep the order it is written in.")

# the same model in another order is the same fit (names compared as sets of parts)
modelreordered, _, _ = quietfit(formula="y ~ h*g + w*x + g:x", data=DATAEXACT, **EXACT)


def byparts(values):
    """Keys a dictionary by the set of parts of each name."""
    return {frozenset(name.split(":")): value for name, value in values.items()}


if abs(modelreordered.getloglik() - modelexact.getloglik()) > TOLEXACT * abs(modelexact.getloglik()):
    sys.exit("Validation test failed at case 3: writing the interactions in another order changed the log-likelihood.")
# (the standard errors come from a finite-difference Hessian, whose rounding
# depends on the order of the columns)
for getter, tolerance in (("getcoef", TOLEXACT), ("getstderr", TOLSE), ("_getwaldcoeftest", TOLSE)):
    original, reordered = byparts(getattr(modelexact, getter)()), byparts(getattr(modelreordered, getter)())
    if set(original) != set(reordered) or max(abs(original[k] - reordered[k]) for k in original) > tolerance:
        sys.exit("Validation test failed at case 3: writing the interactions in another order changed %s." % getter)
original = byparts(modelexact.getwaldtest())
reordered = byparts(modelreordered.getwaldtest())
if set(original) != set(reordered) or max(abs(original[k][0] / reordered[k][0] - 1.0) for k in original) > TOLSE:
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


def coxianloglik(v, y, X, nexit, offmask):
    """
    Log-likelihood of PH(e_1, exp(x'beta) T) by scipy's expm, for v = (log
    exit rates, log off-diagonal rates, beta); the initial phase is the first.
    """
    p = offmask.shape[0]
    t = np.exp(v[:p]) if nexit == p else None
    T = np.zeros((p, p))
    T[offmask] = np.exp(v[p:p + int(offmask.sum())])
    np.fill_diagonal(T, -(T.sum(axis=1) + t))
    beta = v[p + int(offmask.sum()):]
    scale = np.exp(X @ beta)
    alive = expm(T[None, :, :] * (scale * y)[:, None, None])[:, 0, :]
    return float(np.sum(np.log(scale * (alive @ t))))


def fdhessian(function, x0, h):
    """Central-difference Hessian of a function of a vector."""
    k = x0.size
    H = np.zeros((k, k))
    centre = function(x0)
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h
        H[i, i] = (function(x0 + 2 * ei) - 2 * centre + function(x0 - 2 * ei)) / (4 * h * h)
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = h
            H[i, j] = H[j, i] = (function(x0 + ei + ej) - function(x0 + ei - ej)
                                 - function(x0 - ei + ej) + function(x0 - ei - ej)) / (4 * h * h)
    return H


# standard errors: the observed information of the engine's parameters (on
# the centred, scaled design), with the phase-type rates profiled out. The
# fitted Coxian must start in phase 1 and use every exit rate, as the oracle assumes.
engine = model.d
pifit, Tfit, tfit = np.asarray(engine.pi, dtype=float), np.asarray(engine.phgen, dtype=float), np.asarray(engine.exitrates, dtype=float)
offmask = (Tfit > 0.0) & ~np.eye(Tfit.shape[0], dtype=bool)
if abs(pifit[0] - 1.0) > TOL or not np.all(tfit > 0.0):
    sys.exit("Validation test failed at case 4: the fitted Coxian (pi %s, exit rates %s) is not of the form the oracle assumes." % (pifit, tfit))
v0 = np.r_[np.log(tfit), np.log(Tfit[offmask]), np.asarray(engine.beta, dtype=float)]
m = tfit.size + int(offmask.sum())
H = -fdhessian(lambda v: coxianloglik(v, model.design.y, model.design.Xtilde, tfit.size, offmask), v0, STEP)
A, B, C = H[m:, m:], H[m:, :m], H[:m, :m]
oracle = np.sqrt(np.diag(np.linalg.inv(A - B @ np.linalg.solve(C, B.T)))) / model.design.scale
reported = np.array([model.getstderr()[name] for name in coef])
if np.max(np.abs(reported / oracle - 1.0)) > TOLSE:
    sys.exit("Validation test failed at case 4: the standard errors are %s, against %s from the profiled information." % (reported, oracle))
# the information with (pi, T) held fixed must differ, or the check has no power
conditional = np.sqrt(np.diag(np.linalg.inv(A))) / model.design.scale
if np.max(np.abs(conditional / oracle - 1.0)) < 10 * TOLSE:
    sys.exit("Validation test failed at case 4: the data do not separate the profiled from the conditional information.")


# ------------------------------------------------------------------
# CASE 5: Variable types come from the DataFrame
# ------------------------------------------------------------------

# reordering a two-level factor flips its coefficient; one phase makes it exact
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

# unused categories are ignored; integer codes as categories give the same design
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
# the shifted model at x = SHIFT is the original at x = 0 (compared as distributions)
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

# at the boundary: the fit without random effect, zero intercepts, a warning,
# no standard error for sigma
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
    # the design decides whether an interaction may leave out its main
    # effects (case 14), so the parser lets it pass
    "y ~ x:z": ("y", [("x", "z")], None),
}
for formula, (response, terms, group) in accepted.items():
    parsed = _regformula(formula)
    if (parsed.response, parsed.terms, parsed.group) != (response, terms, group):
        sys.exit("Validation test failed at case 9: '%s' was parsed as %s." % (formula, (parsed.response, parsed.terms, parsed.group)))

refused = ["y x + z", "y ~ x ~ z", "y ~ x:z:w", "y ~ log(x)", "y ~ x - 1", "y ~ 0 + x",
           "y ~ x + (x|g)", "y ~ x + (1|g) + (1|h)", "y ~ x + (1||g)", "y ~ x + y",
           "y ~ g + (1|g)", "y ~ 1", "log(y) ~ x", "y ~ x*z:w", "y ~ x:x", "y ~ x + x", 42]
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
if "Formula: y ~ x + (1|group)" not in text or "Random effect (1|group): Std. dev. = " not in text or "Log-likelihood" not in text:
    sys.exit("Validation test failed at case 11: summary does not report the formula, the random effect std. dev. and the log-likelihood.")

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
    if df != 2 or abs(statistic / hand - 1.0) > TOLSE or abs(p - chi2.sf(statistic, 2)) > TOL:
        sys.exit("Validation test failed at case 11: the Wald test of %s is %.6f, against %.6f by hand." % (term, statistic, hand))
# effect table: LRT by default, Wald on request, none with effecttest=None
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

# the likelihood-ratio test of the random effect (getsigmatest('lrt')):
# twice the gain over the fit without (1|u), against 0.5 chi2(0) + 0.5 chi2(1);
# none at the boundary
modelnorandom, _, _ = quietfit(formula="y ~ x", data=DATARANDOMEXACT, **EXACT)
hand = 2.0 * (modelrandomexact.getloglik() - modelnorandom.getloglik())
test = modelrandomexact.getsigmatest("lrt")
statistic, pvalue = test["statistic"], test["pvalue"]
if test["method"] != "lrt" or abs(statistic - hand) > TOLOPT * max(1.0, hand) or abs(pvalue / (0.5 * chi2.sf(hand, 1)) - 1.0) > TOLOPT * max(1.0, hand):
    sys.exit("Validation test failed at case 11: the test of sigma = 0 has statistic %.8f and p-value %.3e, against %.8f and %.3e from the fit without the random effect." % (statistic, pvalue, hand, 0.5 * chi2.sf(hand, 1)))
test = modelzero.getsigmatest("lrt")
if test["statistic"] != 0.0 or test["pvalue"] != 1.0:
    sys.exit("Validation test failed at case 11: at the boundary sigma = 0 the likelihood-ratio test of sigma = 0 is not 0 with p-value 1.")
# the summary prints the score test, the default of getsigmatest (case 20),
# also at the boundary
for name, fitted in (("the random-effect fit", modelrandomexact), ("the boundary fit", modelzero)):
    score = fitted.getsigmatest()
    text = summarytext(fitted, effecttest=None)
    sigmaline = [line for line in text.splitlines() if "Score test of std. dev. = 0: z = " in line]
    if score["method"] != "score" or len(sigmaline) != 1 or \
            abs(float(sigmaline[0].split("z = ")[1].split(",")[0]) - score["statistic"]) > 0.005 or \
            sigmaline[0].split("p-value ")[1].strip() != ("<0.001" if score["pvalue"] < 0.001 else "%.3f" % score["pvalue"]):
        sys.exit("Validation test failed at case 11: the summary of %s does not report the score test of sigma = 0." % name)
    if "Likelihood-ratio test of std. dev." in text:
        sys.exit("Validation test failed at case 11: the summary of %s still reports the likelihood-ratio test of sigma = 0." % name)
if "at the boundary" not in summarytext(modelzero, effecttest=None):
    sys.exit("Validation test failed at case 11: the summary of the boundary fit does not say sigma = 0 is at the boundary.")

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

# the likelihood factorizes: p = proportion of zeros, the rest is the fit to the
# positive observations
iszero = np.arange(NEXACT) % 10 == 3
withzeros = DATAEXACT.assign(y=np.where(iszero, 0.0, DATAEXACT["y"]))
modelzeros, _, _ = quietfit(formula=FORMULAEXACT, data=withzeros, **EXACT)
modelpositive, _, _ = quietfit(formula=FORMULAEXACT, data=withzeros[~iszero], **EXACT)
p0 = iszero.mean()
n0, npositive = int(iszero.sum()), int((~iszero).sum())

if modelzeros.getzeroprob() != p0 or modelexact.getzeroprob() != 0.0:
    sys.exit("Validation test failed at case 12: the probability of a zero is %s, not the proportion of zeros %s." % (modelzeros.getzeroprob(), p0))
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
ZEROLINE = "Probability of observations being zero: p = %.4f" % p0
if ZEROLINE not in buffer.getvalue() or "(%d at zero)" % n0 not in buffer.getvalue() or \
        "1 zero prob." not in buffer.getvalue():
    sys.exit("Validation test failed at case 12: summary does not report the point mass at zero.")
# the constant probability of a zero is a strong assumption, so the summary
# notes it, and only when there are zeros
ZERONOTE = "Note: The probability of a zero is assumed constant"
if ZERONOTE not in buffer.getvalue():
    sys.exit("Validation test failed at case 12: summary does not note that the probability of a zero is assumed constant.")
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    modelexact.summary(effecttest=None)
if ZERONOTE in buffer.getvalue() or "Probability of observations being zero" in buffer.getvalue():
    sys.exit("Validation test failed at case 12: summary reports a point mass at zero in a fit without zeros.")

# the same with a random effect
iszero = np.arange(groupsexact.size) % 9 == 4
randomzeros = DATARANDOMEXACT.assign(y=np.where(iszero, 0.0, DATARANDOMEXACT["y"]))
modelzeros, _, _ = quietfit(formula="y ~ x + (1|u)", data=randomzeros, **EXACT)
modelpositive, _, _ = quietfit(formula="y ~ x + (1|u)", data=randomzeros[~iszero], **EXACT)
p0 = iszero.mean()
loglik = modelpositive.getloglik() + iszero.sum() * np.log(p0) + (~iszero).sum() * np.log(1.0 - p0)
if abs(modelzeros.getloglik() - loglik) > TOL * abs(loglik) or abs(modelzeros.getsigma() - modelpositive.getsigma()) > TOL:
    sys.exit("Validation test failed at case 12: with a random effect the zeros are not a point mass beside the fit to the positive observations.")
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    modelzeros.summary(effecttest=None)
if "Probability of observations being zero: p = %.4f" % p0 not in buffer.getvalue() or ZERONOTE not in buffer.getvalue():
    sys.exit("Validation test failed at case 12: with a random effect the summary does not report the point mass at zero, or does not note that its probability is assumed constant.")


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

# the runtime note: printed (without error) when the refits are expected to
# take longer than lrnotice, and the tests are unchanged
modelexact.lrnotice = -1.0
noted, printed = quiet(modelexact.getlrtest, ["w", "g", "x:g"])
modelexact.lrnotice = 10.0
NOTE = "Note: The likelihood-ratio tests are being computed. This can take about "
if not printed.startswith(NOTE) or not printed.rstrip().endswith(" seconds.") or "summary" in printed:
    sys.exit("Validation test failed at case 13: the runtime note of getlrtest is '%s'." % printed.strip())
if noted != lrtests:
    sys.exit("Validation test failed at case 13: the runtime note changed the likelihood-ratio tests.")

# zeros: the point mass contributes the same factor with and without an effect
iszero = np.arange(NEXACT) % 10 == 3
withzeros = DATAEXACT.assign(y=np.where(iszero, 0.0, DATAEXACT["y"]))
modelzeros, _, _ = quietfit(formula=FORMULAEXACT, data=withzeros, **EXACT)
modelpositive, _, _ = quietfit(formula=FORMULAEXACT, data=withzeros[~iszero], **EXACT)
# (getlrtest must print nothing: the point mass must not enter the statistic)
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

# Clopper-Pearson by hand: closed forms at k = 0 and k = m, tables in between
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

# the absorption-time simulator: mean (t interval) and distribution (chi-square)
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

# the response simulator: z = log(y) + eta0 has mean -gamma - log(RATE), and its
# group means variance SIGMA^2 + (pi^2/6)/(group size)
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


# ------------------------------------------------------------------
# CASE 14: A separate slope of x for each level of g
# ------------------------------------------------------------------

sloperng = np.random.default_rng([SEED, 14])
xs = SLOPEMEAN + sloperng.normal(size=NEXACT)
gs = sloperng.choice(["a", "b", "c"], size=NEXACT)
if set(gs) != set(SLOPES):
    sys.exit("Validation test failed at case 14: the simulated g does not contain all its levels.")
levelcols = np.column_stack([(gs == level) * 1.0 for level in ("a", "b", "c")])
ys = sloperng.exponential(size=NEXACT) / (EXACTRATE * np.exp(xs * (levelcols @ np.array([SLOPES[k] for k in "abc"]))))
DATASLOPES = pd.DataFrame({"y": ys, "x": xs, "g": categorical(gs)})


def slopesoracle(X, response, model, names, label):
    """
    Checks a one-phase fit against exponential regression on the hand-built
    columns X: the coefficients are minus those of a log-link Gamma GLM, and
    the log-likelihood and standard errors are those of exponential
    regression at the estimates.
    """
    coef = model.getcoef()
    if list(coef) != names:
        sys.exit("Validation test failed at case 14: the coefficients of %s are named %s, not %s." % (label, list(coef), names))
    full = np.column_stack([np.ones(response.size), X])
    oracle = sm.GLM(response, full, family=sm.families.Gamma(link=sm.families.links.Log())).fit(tol=1e-14, maxiter=1000)
    estimates = np.array([coef[name] for name in names])
    if np.max(np.abs(estimates + oracle.params[1:])) > TOLEXACT:
        sys.exit("Validation test failed at case 14: the coefficients of %s differ from minus those of the Gamma GLM by %.3e." % (label, np.max(np.abs(estimates + oracle.params[1:]))))
    rate = -model.getphasegen()[0, 0]
    u = rate * np.exp(X @ estimates) * response
    loglik = np.sum(np.log(rate) + X @ estimates - u)
    if abs(model.getloglik() - loglik) > TOLEXACT * abs(loglik):
        sys.exit("Validation test failed at case 14: the log-likelihood of %s is %.10f, against %.10f by hand." % (label, model.getloglik(), loglik))
    stderr = np.sqrt(np.diag(np.linalg.inv((full * u[:, None]).T @ full)))[1:]
    reported = np.array([model.getstderr()[name] for name in names])
    if np.max(np.abs(reported / stderr - 1.0)) > TOLSE:
        sys.exit("Validation test failed at case 14: the standard errors of %s differ from those of exponential regression by a factor of up to %.3e." % (label, np.max(np.abs(reported / stderr - 1.0))))
    return estimates


def levelslopes(coef, levels):
    """The slope of x for each level of g, from either coding."""
    if "x" in coef:
        return {level: coef["x"] + coef.get("x:g[%s]" % level, 0.0) for level in levels}
    return {level: coef["x:g[%s]" % level] for level in levels}


# 'x:g': one column per level, since g alone is not in the model
modelfull, _, _ = quietfit(formula="y ~ x:g", data=DATASLOPES, **EXACT)
slopesoracle(xs[:, None] * levelcols, ys, modelfull, ["x:g[a]", "x:g[b]", "x:g[c]"], "'y ~ x:g'")

# 'x + x:g': the common slope and the differences against the reference level,
# since x alone is in the model
modelcontrast, _, _ = quietfit(formula="y ~ x + x:g", data=DATASLOPES, **EXACT)
slopesoracle(np.column_stack([xs, xs * levelcols[:, 1], xs * levelcols[:, 2]]), ys, modelcontrast,
             ["x", "x:g[b]", "x:g[c]"], "'y ~ x + x:g'")

# the two are one model: the same likelihood and the same slope for each level
if abs(modelfull.getloglik() - modelcontrast.getloglik()) > TOLEXACT * abs(modelfull.getloglik()):
    sys.exit("Validation test failed at case 14: 'y ~ x:g' and 'y ~ x + x:g' have different log-likelihoods.")
slopesfull = levelslopes(modelfull.getcoef(), "abc")
slopescontrast = levelslopes(modelcontrast.getcoef(), "abc")
if max(abs(slopesfull[k] - slopescontrast[k]) for k in "abc") > TOLEXACT:
    sys.exit("Validation test failed at case 14: 'y ~ x:g' and 'y ~ x + x:g' give different slopes for the levels of g: %s and %s." % (slopesfull, slopescontrast))

# 'g + x:g': g by treatment dummies, x:g one column per level, which spans the
# same columns as 'x*g'
modelwithg, _, _ = quietfit(formula="y ~ g + x:g", data=DATASLOPES, **EXACT)
modelstar, _, _ = quietfit(formula="y ~ x*g", data=DATASLOPES, **EXACT)
slopesoracle(np.column_stack([levelcols[:, 1:], xs[:, None] * levelcols]), ys, modelwithg,
             ["g[b]", "g[c]", "g[a]:x", "g[b]:x", "g[c]:x"], "'y ~ g + x:g'")
if abs(modelwithg.getloglik() - modelstar.getloglik()) > TOLEXACT * abs(modelstar.getloglik()):
    sys.exit("Validation test failed at case 14: 'y ~ g + x:g' is not the model 'y ~ x*g'.")

# the reference level of g does not matter: with c first the fit is the same,
# slope for slope, and in 'x + x:g' the coefficient of x is now the slope of c
reordered = DATASLOPES.assign(g=DATASLOPES["g"].cat.reorder_categories(["c", "a", "b"]))
for formula, original in (("y ~ x:g", slopesfull), ("y ~ x + x:g", slopescontrast)):
    model, _, _ = quietfit(formula=formula, data=reordered, **EXACT)
    slopes = levelslopes(model.getcoef(), "abc")
    if abs(model.getloglik() - modelfull.getloglik()) > TOLEXACT * abs(modelfull.getloglik()) or \
            max(abs(slopes[k] - original[k]) for k in "abc") > TOLEXACT:
        sys.exit("Validation test failed at case 14: '%s' depends on the reference level of g." % formula)
    if formula == "y ~ x + x:g" and (list(model.getcoef()) != ["x", "x:g[a]", "x:g[b]"] or abs(model.getcoef()["x"] - original["c"]) > TOLEXACT):
        sys.exit("Validation test failed at case 14: with c as the reference level, 'y ~ x + x:g' is not coded against c.")

# the fit depends on the origin of x: on a shifted x it is the GLM on shifted columns
shifted = xs + ORIGINSHIFT
modelshifted, _, _ = quietfit(formula="y ~ x:g", data=DATASLOPES.assign(x=shifted), **EXACT)
slopesoracle(shifted[:, None] * levelcols, ys, modelshifted, ["x:g[a]", "x:g[b]", "x:g[c]"], "'y ~ x:g' on a shifted x")
if abs(modelshifted.getloglik() - modelfull.getloglik()) < ORIGINGAP:
    sys.exit("Validation test failed at case 14: shifting x changed the log-likelihood by only %.3f, so the fit does not depend on the origin of x." % abs(modelshifted.getloglik() - modelfull.getloglik()))

# a prediction row follows the coding, and so does the CDF
for model, covariates, handrow in ((modelfull, {"x": 2.0, "g": "b"}, [0.0, 2.0, 0.0]),
                                   (modelcontrast, {"x": 2.0, "g": "b"}, [2.0, 2.0, 0.0]),
                                   (modelcontrast, {"x": 2.0, "g": "a"}, [2.0, 0.0, 0.0]),
                                   (modelwithg, {"x": 2.0, "g": "c"}, [0.0, 1.0, 0.0, 0.0, 2.0])):
    row = model.design.row(covariates)
    if np.max(np.abs(row - np.array(handrow))) > TOL:
        sys.exit("Validation test failed at case 14: the prediction row of '%s' at %s is %s, not %s." % (model.formula, covariates, row, handrow))
    rate = -model.getphasegen()[0, 0] * np.exp(row @ np.array(list(model.getcoef().values())))
    if abs(model.getcumprob(1.5, covariates) - (1.0 - np.exp(-1.5 * rate))) > TOL:
        sys.exit("Validation test failed at case 14: the CDF of '%s' at %s is not that of the exponential distribution with the predicted rate." % (model.formula, covariates))

# the likelihood-ratio test of x:g in 'x + x:g' drops the differences, leaving
# one slope for all levels: the test of equal slopes
lrslopes = quiet(modelcontrast.getlrtest, "x:g")[0]["x:g"]
hand = 2.0 * (exponentialloglik(xs[:, None] * levelcols, ys) - exponentialloglik(xs[:, None], ys))
if lrslopes["df"] != 2 or abs(lrslopes["statistic"] - hand) > TOLEXACT * max(1.0, hand):
    sys.exit("Validation test failed at case 14: the likelihood-ratio statistic of equal slopes is %.10f (%d df), against %.10f (2 df) by hand." % (lrslopes["statistic"], lrslopes["df"], hand))

# missing main effects are allowed only for continuous x categorical
DATAKINDS = DATASLOPES.assign(z=sloperng.normal(size=NEXACT), h=categorical(sloperng.choice(["p", "q"], size=NEXACT)))
for formula, missing, written in (("y ~ x:z", "'x'", "x*z"), ("y ~ x + x:z", "'z'", "x*z"),
                                  ("y ~ g:h", "'g'", "g*h"), ("y ~ g + g:h", "'h'", "g*h"),
                                  ("y ~ x + g + h + x:g + x:h + x:g:h", "'g:h'", "x*g*h")):
    try:
        parsed = _regformula(formula)
        _regdesign(DATAKINDS, parsed.response, parsed.terms, parsed.group)
        sys.exit("Validation test failed at case 14: the design accepted '%s'." % formula)
    except ValueError as error:
        if missing not in str(error) or written not in str(error):
            sys.exit("Validation test failed at case 14: the refusal of '%s' does not name %s and '%s': %s" % (formula, missing, written, error))
    if not terminates(formula=formula, data=DATAKINDS, **EXACT):
        sys.exit("Validation test failed at case 14: '%s' did not terminate." % formula)


# ------------------------------------------------------------------
# CASES 15-19: Censored observations
# ------------------------------------------------------------------

def censor(times, rng, scheme=CENSSCHEME):
    """
    Censors the times with limits that do not depend on them: each row is
    exact, right-censored at a uniform limit, or inspected on CENSGRID.
    Returns the response (NaN where censored) and the lower and upper limits
    (NaN where there is none).
    """
    n = times.size
    kind = rng.choice(3, size=n, p=scheme)
    limit = rng.uniform(*CENSRIGHT, size=n)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    right = (kind == 1) & (times > limit)
    lower[right] = limit[right]
    for i in np.flatnonzero(kind == 2):
        k = np.searchsorted(CENSGRID, times[i]) - 1
        if CENSGRID[k + 1] < np.inf:
            upper[i] = CENSGRID[k + 1]
        if k > 0:
            lower[i] = CENSGRID[k]
    response = np.where(np.isnan(lower) & np.isnan(upper), times, np.nan)
    return response, lower, upper


def censoredkinds(lower, upper):
    """Kind of each row: 0 exact, 1 right, 2 left, 3 interval."""
    return np.where(np.isnan(lower), np.where(np.isnan(upper), 0, 2), np.where(np.isnan(upper), 1, 3))


def exponentialcensored(D, kind, lower, upper, response):
    """
    The one-phase censored MLE by Newton's method on (log rate, beta): the
    log-likelihood is concave in z = D c. Returns
    the estimates, the log-likelihood and the observed information.
    """
    lo = np.where(kind == 0, response, np.where(kind == 2, upper, lower))
    hi = upper

    def parts(c):
        z = D @ c
        r = np.exp(z)
        value, g, h = np.zeros(z.size), np.zeros(z.size), np.zeros(z.size)
        e = kind == 0
        value[e], g[e], h[e] = z[e] - r[e] * lo[e], 1.0 - r[e] * lo[e], -r[e] * lo[e]
        e = kind == 1
        value[e], g[e], h[e] = -r[e] * lo[e], -r[e] * lo[e], -r[e] * lo[e]
        e = kind >= 2
        a = np.where(kind[e] == 3, r[e] * lo[e], 0.0)
        b = np.where(kind[e] == 3, r[e] * hi[e], r[e] * lo[e])
        P = np.exp(-a) - np.exp(-b)
        G = b * np.exp(-b) - a * np.exp(-a)
        H = G - b ** 2 * np.exp(-b) + a ** 2 * np.exp(-a)
        value[e], g[e], h[e] = np.log(P), G / P, H / P - (G / P) ** 2
        return np.sum(value), D.T @ g, (D * h[:, None]).T @ D

    c = np.zeros(D.shape[1])
    c[0] = -np.log(np.mean(lo))
    for _ in range(200):
        value, g, H = parts(c)
        step = np.linalg.solve(-H, g)
        scale = 1.0
        while parts(c + scale * step)[0] < value and scale > 1e-12:
            scale /= 2.0
        c = c + scale * step
        if np.max(np.abs(scale * step)) < 1e-13:
            break
    value, g, H = parts(c)
    return c, value, -H


censrng = np.random.default_rng([SEED, 15])
xc = censrng.normal(size=NEXACT)
gc = censrng.choice(["a", "b", "c"], size=NEXACT)
Dc = np.column_stack([np.ones(NEXACT), xc, (gc == "b") * 1.0, (gc == "c") * 1.0])
tc = censrng.exponential(size=NEXACT) / (EXACTRATE * np.exp(Dc[:, 1:] @ np.array([0.6, -0.4, 0.5])))
CENSNAMES = ["x", "g[b]", "g[c]"]


# ------------------------------------------------------------------
# CASE 15: One phase with right-censoring is a Poisson GLM
# ------------------------------------------------------------------

response, lower, upper = censor(tc, censrng, scheme=(0.0, 1.0, 0.0))
kind = censoredkinds(lower, upper)
if set(kind) != {0, 1}:
    sys.exit("Validation test failed at case 15: the data are not a mix of exact and right-censored rows.")
DATARIGHT = pd.DataFrame({"y": response, "x": xc, "g": categorical(gc), "lo": lower, "hi": upper})
modelright, _, _ = quietfit(formula="y ~ x + g", data=DATARIGHT, censoring=("lo", "hi"), **EXACT)

# right-censored exponential regression is a Poisson GLM with offset log(time)
event = (kind == 0) * 1.0
time = np.where(kind == 0, response, lower)
poisson = sm.GLM(event, Dc, family=sm.families.Poisson(), offset=np.log(time)).fit(tol=1e-14, maxiter=1000)
estimates = np.array([modelright.getcoef()[k] for k in CENSNAMES])
if list(modelright.getcoef()) != CENSNAMES or np.max(np.abs(estimates - poisson.params[1:])) > TOLEXACT:
    sys.exit("Validation test failed at case 15: the coefficients with right-censoring differ from the Poisson GLM by %.3e." % np.max(np.abs(estimates - poisson.params[1:])))
if abs(-modelright.getphasegen()[0, 0] / np.exp(poisson.params[0]) - 1.0) > TOLEXACT:
    sys.exit("Validation test failed at case 15: the baseline rate with right-censoring differs from the Poisson GLM.")
mu = np.exp(Dc @ poisson.params) * time
loglik = np.sum(event * (Dc @ poisson.params) - mu)
if abs(modelright.getloglik() - loglik) > TOLEXACT * abs(loglik):
    sys.exit("Validation test failed at case 15: the log-likelihood with right-censoring is %.10f, against %.10f." % (modelright.getloglik(), loglik))

# standard errors: the full information with the rate profiled out (the
# conditional information is invalid with censoring and must differ)
information = (Dc * mu[:, None]).T @ Dc
full = np.sqrt(np.diag(np.linalg.inv(information)))[1:]
Dcentred = Dc[:, 1:] - Dc[:, 1:].mean(axis=0)
conditional = np.sqrt(np.diag(np.linalg.inv((Dcentred * mu[:, None]).T @ Dcentred)))
reported = np.array([modelright.getstderr()[k] for k in CENSNAMES])
if np.max(np.abs(reported / full - 1.0)) > TOLSE:
    sys.exit("Validation test failed at case 15: the standard errors with censoring are %s, against %s from the profiled information." % (reported, full))
# (measured: conditional / full = 0.978 for x at 30% censoring)
if np.max(np.abs(conditional / full - 1.0)) < 10 * TOLSE:
    sys.exit("Validation test failed at case 15: the data do not separate the full from the conditional information.")

# the likelihood-ratio test of x is the difference of the Poisson deviances
reducedpoisson = sm.GLM(event, Dc[:, [0, 2, 3]], family=sm.families.Poisson(), offset=np.log(time)).fit(tol=1e-14, maxiter=1000)
lrright = quiet(modelright.getlrtest, "x")[0]["x"]
if abs(lrright["statistic"] - 2.0 * (poisson.llf - reducedpoisson.llf)) > TOLEXACT * max(1.0, lrright["statistic"]):
    sys.exit("Validation test failed at case 15: the likelihood-ratio statistic with censoring is %.10f, against %.10f." % (lrright["statistic"], 2.0 * (poisson.llf - reducedpoisson.llf)))


# ------------------------------------------------------------------
# CASE 16: One phase with every kind of censoring
# ------------------------------------------------------------------

response, lower, upper = censor(tc, censrng)
kind = censoredkinds(lower, upper)
if set(kind) != {0, 1, 2, 3}:
    sys.exit("Validation test failed at case 16: the data do not contain every kind of row.")
DATACENSORED = pd.DataFrame({"y": response, "x": xc, "g": categorical(gc), "lo": lower, "hi": upper})
modelcensored, _, _ = quietfit(formula="y ~ x + g", data=DATACENSORED, censoring=("lo", "hi"), **EXACT)

oraclec, oraclell, oracleinformation = exponentialcensored(Dc, kind, lower, upper, response)
estimates = np.array([modelcensored.getcoef()[k] for k in CENSNAMES])
if np.max(np.abs(estimates - oraclec[1:])) > TOLEXACT or abs(-modelcensored.getphasegen()[0, 0] / np.exp(oraclec[0]) - 1.0) > TOLEXACT:
    sys.exit("Validation test failed at case 16: the estimates with every kind of censoring differ from the one-phase MLE by %.3e." % np.max(np.abs(estimates - oraclec[1:])))
if abs(modelcensored.getloglik() - oraclell) > TOLEXACT * abs(oraclell):
    sys.exit("Validation test failed at case 16: the log-likelihood with every kind of censoring is %.10f, against %.10f." % (modelcensored.getloglik(), oraclell))
full = np.sqrt(np.diag(np.linalg.inv(oracleinformation)))[1:]
reported = np.array([modelcensored.getstderr()[k] for k in CENSNAMES])
if np.max(np.abs(reported / full - 1.0)) > TOLSE:
    sys.exit("Validation test failed at case 16: the standard errors with every kind of censoring are %s, against %s." % (reported, full))

counts = [int(np.count_nonzero(kind == k)) for k in (1, 2, 3)]
text = io.StringIO()
with contextlib.redirect_stdout(text):
    modelcensored.summary(effecttest=None)
expected = "Observations: %d (%d right-censored, %d left-censored, %d interval-censored)" % (NEXACT, *counts)
if expected not in text.getvalue():
    sys.exit("Validation test failed at case 16: the summary does not count the censored observations ('%s')." % expected)


# ------------------------------------------------------------------
# CASE 17: Censoring leaves the uncensored path alone; three-phase rows
# ------------------------------------------------------------------

# censoring columns in which no row is censored give the uncensored fit,
# identical to the last bit
plain, _, _ = quietfit(formula="y ~ x + g", data=DATA, **COMMON)
nocensoring, _, _ = quietfit(formula="y ~ x + g", data=DATA.assign(lo=np.nan, hi=np.nan), censoring=("lo", "hi"), **COMMON)
for getter in ("getcoef", "getstderr"):
    if getattr(plain, getter)() != getattr(nocensoring, getter)():
        sys.exit("Validation test failed at case 17: censoring columns without censored rows changed %s." % getter)
if plain.getloglik() != nocensoring.getloglik() or not np.array_equal(plain.getphasegen(), nocensoring.getphasegen()) \
        or not np.array_equal(plain.getinitdist(), nocensoring.getinitdist()) or plain.iterations != nocensoring.iterations:
    sys.exit("Validation test failed at case 17: censoring columns without censored rows changed the fit.")

# three phases with every kind of censoring: the fit converges and the
# effects are recovered
threerng = np.random.default_rng([SEED, 17])
response, lower, upper = censor(DATA["y"].to_numpy(), threerng)
kind3 = censoredkinds(lower, upper)
if set(kind3) != {0, 1, 2, 3}:
    sys.exit("Validation test failed at case 17: the three-phase data do not contain every kind of row.")
modelthree, _, _ = quietfit(formula="y ~ x + g", data=DATA.assign(y=response, lo=lower, hi=upper), censoring=("lo", "hi"), **COMMON)
for name, truth in TRUEBETA.items():
    if not modelthree.converged or abs(modelthree.getcoef()[name] - truth) > 4.0 * modelthree.getstderr()[name]:
        sys.exit("Validation test failed at case 17: the censored three-phase fit gives %s = %.3f (s.e. %.3f), truth %.3f." % (name, modelthree.getcoef()[name], modelthree.getstderr()[name], truth))

# the engine's rows at the true Coxian against scipy's expm: log-likelihood, l',
# l'', Fisher's identity, sum E(Z) = u (right), sum E(N) = 1 (left, interval)
engine = modelthree.d
truthT = np.diag(-TRUERATES) + np.diag(TRUERATES[:-1] * TRUECONTINUE, 1)
truths = dict(pi=np.eye(3)[0], T=truthT, t=-truthT.sum(axis=1), beta=np.array([0.3, -0.2, 0.1]), sigma=None)
etas = engine.X @ truths["beta"]
rowsll = engine._fitcphreg__censoredloglik(truths, etas, 1)
d1, d2 = engine._fitcphreg__rowderivs(truths, etas)
rowsll2, e = engine._fitcphreg__censoredestep(truths, etas, 1)
weighed = engine._fitcphreg__weigh(truths, e, np.ones(engine.n))
kinde = engine.kind
lo, hi = engine.lo, engine.hi


def rowloglik(i, eta):
    """Log-likelihood of engine row i at the true Coxian, by scipy's expm."""
    survival = lambda c: truths["pi"] @ expm(truthT * np.exp(eta) * c) @ np.ones(3)
    if kinde[i] == 0:
        return eta + np.log(truths["pi"] @ expm(truthT * np.exp(eta) * lo[i]) @ truths["t"])
    if kinde[i] == 1:
        return np.log(survival(lo[i]))
    if kinde[i] == 2:
        return np.log(1.0 - survival(lo[i]))
    return np.log(survival(lo[i]) - survival(hi[i]))


worst = dict(loglik=0.0, derivative=0.0, second=0.0, fisher=0.0, window=0.0, absorbed=0.0)
for i in range(engine.n):
    value = rowloglik(i, etas[i])
    worst["loglik"] = max(worst["loglik"], abs(rowsll[i] - value), abs(rowsll2[i] - value))
    h = 1e-4
    numeric = (8 * (rowloglik(i, etas[i] + h) - rowloglik(i, etas[i] - h))
               - (rowloglik(i, etas[i] + 2 * h) - rowloglik(i, etas[i] - 2 * h))) / (12 * h)
    worst["derivative"] = max(worst["derivative"], abs(d1[i] - numeric))
    h = 1e-3
    second = (16 * (rowloglik(i, etas[i] + h) + rowloglik(i, etas[i] - h)) - 30 * value
              - (rowloglik(i, etas[i] + 2 * h) + rowloglik(i, etas[i] - 2 * h))) / (12 * h * h)
    worst["second"] = max(worst["second"], abs(d2[i] - second))
    exposure = weighed["zrow"][i] @ (-np.diag(truthT))
    worst["fisher"] = max(worst["fisher"], abs(weighed["jumps"][i] - exposure - d1[i]))
    if kinde[i] == 1:
        worst["window"] = max(worst["window"], abs(np.sum(weighed["zrow"][i]) - np.exp(etas[i]) * lo[i]))
    if kinde[i] >= 2:
        worst["absorbed"] = max(worst["absorbed"], abs((e["a"][i] @ truths["t"]) * e["invf"][i] - 1.0))
if max(worst["loglik"], worst["fisher"], worst["window"], worst["absorbed"]) > TOL or worst["derivative"] > 1e-7 \
        or worst["second"] > 1e-6:
    sys.exit("Validation test failed at case 17: the censored rows of the engine disagree with the independent evaluation: %s." % worst)


# ------------------------------------------------------------------
# CASE 18: One phase with a random effect and censoring against quadrature
# ------------------------------------------------------------------

response, lower, upper = censor(yre, np.random.default_rng([SEED, 18]))
kindre = censoredkinds(lower, upper)
if set(kindre) != {0, 1, 2, 3}:
    sys.exit("Validation test failed at case 18: the data do not contain every kind of row.")
modelrandomcens, _, _ = quietfit(formula="y ~ x + (1|u)", data=DATARANDOMEXACT.assign(y=response, lo=lower, hi=upper),
                                 censoring=("lo", "hi"), quadnodes=15, **EXACT)
lore = np.where(kindre == 0, response, np.where(kindre == 2, upper, lower))


def randomcensnegloglik(v):
    """
    Negative marginal log-likelihood of exponential regression with a Gaussian
    random effect and censored rows, each group's integral by the trapezoidal
    rule on GRID; a row's contribution is its log-density or the log of its
    probability, evaluated at every grid point.
    """
    lograte, beta, logsigma = v
    sigma = np.exp(logsigma)
    z = (lograte + beta * xre)[:, None] + GRID[None, :]
    r = np.exp(z)
    rows = np.empty(z.shape)
    e = kindre == 0
    rows[e] = z[e] - r[e] * lore[e, None]
    e = kindre == 1
    rows[e] = -r[e] * lore[e, None]
    e = kindre == 2
    rows[e] = np.log(-np.expm1(-r[e] * lore[e, None]))
    e = kindre == 3
    rows[e] = -r[e] * lore[e, None] + np.log(-np.expm1(-r[e] * (upper[e] - lore[e])[:, None]))
    logintegrand = np.add.reduceat(rows, np.flatnonzero(np.r_[True, np.diff(groupsexact) != 0]), axis=0)
    logintegrand += -0.5 * GRID[None, :] ** 2 / sigma ** 2 - 0.5 * np.log(2.0 * np.pi * sigma ** 2)
    return -np.sum(logsumexp(logintegrand + LOGTRAPEZOID[None, :], axis=1))


oracle = minimize(randomcensnegloglik, np.array([0.0, 0.0, np.log(0.5)]), method="Nelder-Mead",
                  options=dict(xatol=1e-9, fatol=1e-11, maxiter=4000))
fitted = np.array([np.log(-modelrandomcens.getphasegen()[0, 0]), modelrandomcens.getcoef()["x"],
                   np.log(modelrandomcens.getsigma())])
if np.max(np.abs(fitted - oracle.x)) > TOLOPT or abs(modelrandomcens.getloglik() + oracle.fun) > TOLOPT:
    sys.exit("Validation test failed at case 18: with censoring the random-effect fit differs from a direct maximization of the quadrature likelihood by %.3e." % np.max(np.abs(fitted - oracle.x)))
hessian = np.zeros((3, 3))
for i in range(3):
    ei = np.zeros(3)
    ei[i] = STEP
    hessian[i, i] = -(randomcensnegloglik(oracle.x + 2 * ei) - 2 * oracle.fun + randomcensnegloglik(oracle.x - 2 * ei)) / (4 * STEP ** 2)
    for j in range(i + 1, 3):
        ej = np.zeros(3)
        ej[j] = STEP
        hessian[i, j] = hessian[j, i] = -(randomcensnegloglik(oracle.x + ei + ej) - randomcensnegloglik(oracle.x + ei - ej)
                                          - randomcensnegloglik(oracle.x - ei + ej) + randomcensnegloglik(oracle.x - ei - ej)) / (4 * STEP ** 2)
oraclestderr = np.sqrt(np.diag(np.linalg.inv(-hessian)))[1:]
reported = np.array([modelrandomcens.getstderr()["x"], modelrandomcens.sigmastderr / modelrandomcens.getsigma()])
if np.max(np.abs(reported / oraclestderr - 1.0)) > TOLSE:
    sys.exit("Validation test failed at case 18: with censoring the standard errors are %s, against %s from the oracle." % (reported, oraclestderr))


# ------------------------------------------------------------------
# CASE 19: Censoring: refusals, zeros, simulated p-values and the plot
# ------------------------------------------------------------------

bad = {
    "a single column name": dict(censoring="lo"),
    "the same column twice": dict(censoring=("lo", "lo")),
    "three columns": dict(censoring=("lo", "hi", "y")),
    "an unknown column": dict(censoring=("lo", "upper")),
    "a categorical column": dict(censoring=("lo", "g")),
    "a column of the formula": dict(censoring=("x", "hi")),
    "a negative lower limit": dict(data=DATACENSORED.assign(lo=np.where(np.arange(NEXACT) == 1, -1.0, DATACENSORED["lo"]))),
    "a lower limit above the upper": dict(data=DATACENSORED.assign(lo=np.where(np.arange(NEXACT) == 1, 3.0, DATACENSORED["lo"]),
                                                                    hi=np.where(np.arange(NEXACT) == 1, 2.0, DATACENSORED["hi"]))),
    "an upper limit of zero": dict(data=DATACENSORED.assign(lo=np.where(np.arange(NEXACT) == 1, np.nan, DATACENSORED["lo"]),
                                                             hi=np.where(np.arange(NEXACT) == 1, 0.0, DATACENSORED["hi"]))),
    "a missing response in an exact row": dict(data=DATACENSORED.assign(y=np.where(kind == 0, np.nan, DATACENSORED["y"]))),
    "only right-censored rows": dict(data=DATACENSORED.assign(lo=1.0, hi=np.nan)),
}
for name, change in bad.items():
    arguments = dict(formula="y ~ x + g", data=DATACENSORED, censoring=("lo", "hi"), **EXACT)
    arguments.update(change)
    if not terminates(**arguments):
        sys.exit("Validation test failed at case 19: censoring with %s did not terminate." % name)

# zeros with censored rows: a censored row is positive whatever its response,
# and the point mass is the proportion of exact zeros, as in fit
zero = (kind == 0) & (np.arange(NEXACT) % 7 == 3)
withzeros = DATACENSORED.assign(y=np.where(zero, 0.0, np.where(kind == 0, DATACENSORED["y"], 0.0)))
modelzc, _, _ = quietfit(formula="y ~ x + g", data=withzeros, censoring=("lo", "hi"), **EXACT)
modelpc, _, _ = quietfit(formula="y ~ x + g", data=withzeros[~zero], censoring=("lo", "hi"), **EXACT)
p0, n0 = zero.mean(), int(zero.sum())
loglik = modelpc.getloglik() + n0 * np.log(p0) + (NEXACT - n0) * np.log(1.0 - p0)
if abs(modelzc.getzeroprob() - p0) > TOL or abs(modelzc.getloglik() - loglik) > TOL * abs(loglik) or \
        max(abs(modelzc.getcoef()[k] - modelpc.getcoef()[k]) for k in CENSNAMES) > TOL:
    sys.exit("Validation test failed at case 19: with censored rows the zeros are not the point mass of the exact zeros.")

# simulated p-values need a model of the censoring, so they are refused
result, printed = quiet(modelcensored.getlrtest, "x", nsim=2)
if result is not None or "Error" not in printed:
    sys.exit("Validation test failed at case 19: getlrtest gave simulated p-values with censored observations.")
if "Error" not in quiet(modelcensored.summary, nsim=2)[1]:
    sys.exit("Validation test failed at case 19: summary gave simulated p-values with censored observations.")

# the transforms are censored, so the PP plot is left out with a warning
plotfile = "fitreg_test_censored_plot.png"
try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        modelcensored.plot(filename=plotfile)
    if os.path.exists(plotfile) or not any("censored" in str(w.message) for w in caught):
        sys.exit("Validation test failed at case 19: the plot of a censored fit was drawn, or no warning was given.")
finally:
    if os.path.exists(plotfile):
        os.remove(plotfile)


# ------------------------------------------------------------------
# CASE 20: The score test of the random effect
# ------------------------------------------------------------------


def scoretestbyhand(score, kinds, X, groupcodes):
    """
    The score test of sigma = 0 with dense matrices, from the rows' scores in
    eta at the fit without random effect: the scores standardized by their
    variance pooled within each kind of row (times n / (n - rank [1, X])),
    e = D^(-1/2) l'; their covariance I - H, H the projection on D^(1/2) A,
    A = [1, X]; the p-value P(e' S e / e' e >= c) for the observed ratio c, S
    the same-group indicator (diagonal included), from the eigenvalues of
    (I - H)(S - c I)(I - H); and T = e' (S - I) e / 2 standardized by the
    eigenvalues of (I - H)(S - I)(I - H) / 2. Returns the standardized
    statistic and the p-value.
    """
    n = score.size
    A = np.column_stack([np.ones(n), X])
    d = np.empty(n)
    for k in np.unique(kinds):
        d[kinds == k] = np.mean(score[kinds == k] ** 2)
    d *= n / (n - np.linalg.matrix_rank(A))
    e = score / np.sqrt(d)
    S = (groupcodes[:, None] == groupcodes[None, :]).astype(float)
    B = np.sqrt(d)[:, None] * A
    L = np.eye(n) - B @ np.linalg.inv(B.T @ B) @ B.T
    c = float(e @ S @ e) / float(e @ e)
    pvalue = _quadform(np.linalg.eigvalsh(L @ (S - c * np.eye(n)) @ L), np.ones(n)).sf(0.0)
    moments = _quadform(np.linalg.eigvalsh(L @ (0.5 * (S - np.eye(n))) @ L), np.ones(n))
    T = 0.5 * float(e @ (S - np.eye(n)) @ e)
    return (T - moments.mean) / np.sqrt(moments.var), pvalue


# uncensored: for one phase the score of a row is 1 - lambda exp(x beta) y;
# on the data of case 2 (sigma = 0.8) and on the same design without group
# effect, where the p-value is moderate
yre0 = np.random.default_rng([SEED, 20]).exponential(size=xre.size) / (EXACTRATE * np.exp(EXACTBETA["x"] * xre))
for label, response0, fitted in (("the data of case 2", yre, modelrandomexact),
                                 ("data without group effect", yre0,
                                  quietfit(formula="y ~ x + (1|u)", data=DATARANDOMEXACT.assign(y=yre0), **EXACT)[0])):
    nogroups, _, _ = quietfit(formula="y ~ x", data=DATARANDOMEXACT.assign(y=response0), **EXACT)
    rate = -nogroups.getphasegen()[0, 0]
    scores = 1.0 - rate * np.exp(nogroups.getcoef()["x"] * xre) * response0
    hand = scoretestbyhand(scores, np.zeros(xre.size), (xre - xre.mean())[:, None], groupsexact)
    test = fitted.getsigmatest()
    if test["method"] != "score" or abs(test["statistic"] - hand[0]) > TOLSCORE * max(1.0, abs(hand[0])) or \
            abs(test["pvalue"] - hand[1]) > TOLSCORE * max(hand[1], 1e-3):
        sys.exit("Validation test failed at case 20: on %s the score test is z = %.8f, p = %.8e, against %.8f, %.8e by hand." % (label, test["statistic"], test["pvalue"], hand[0], hand[1]))
    if label == "data without group effect" and not 0.01 < hand[1] < 0.99:
        sys.exit("Validation test failed at case 20: the data without group effect give the p-value %.4f, not a moderate one." % hand[1])
# it comes from the fit without random effect, so it exists at the boundary
test = modelzero.getsigmatest()
if not (np.isfinite(test["statistic"]) and 0.0 < test["pvalue"] < 1.0):
    sys.exit("Validation test failed at case 20: the score test of the boundary fit is z = %s, p = %s." % (test["statistic"], test["pvalue"]))

# censored: the scores of right-, left- and interval-censored rows, with r
# the rate times exp(x beta), and the variance pooled within each kind
censoredrandom = DATARANDOMEXACT.assign(y=response, lo=lower, hi=upper)
modelcensnorandom, _, _ = quietfit(formula="y ~ x", data=censoredrandom, censoring=("lo", "hi"), **EXACT)
r = -modelcensnorandom.getphasegen()[0, 0] * np.exp(modelcensnorandom.getcoef()["x"] * xre)
scores = np.empty(xre.size)
e = kindre == 0
scores[e] = 1.0 - r[e] * lore[e]
e = kindre == 1
scores[e] = -r[e] * lore[e]
e = kindre == 2
scores[e] = r[e] * lore[e] * np.exp(-r[e] * lore[e]) / -np.expm1(-r[e] * lore[e])
e = kindre == 3
sl, su = np.exp(-r[e] * lore[e]), np.exp(-r[e] * upper[e])
scores[e] = (r[e] * upper[e] * su - r[e] * lore[e] * sl) / (sl - su)
hand = scoretestbyhand(scores, kindre, (xre - xre.mean())[:, None], groupsexact)
test = modelrandomcens.getsigmatest()
if abs(test["statistic"] - hand[0]) > TOLSCORE * max(1.0, abs(hand[0])) or abs(test["pvalue"] - hand[1]) > TOLSCORE * max(hand[1], 1e-3):
    sys.exit("Validation test failed at case 20: with censoring the score test is z = %.8f, p = %.8e, against %.8f, %.8e by hand." % (test["statistic"], test["pvalue"], hand[0], hand[1]))
# pooling within kinds matters here (the hand computation with one pooled
# variance must differ), or the check above has no power (measured: z 7.77
# against 7.87)
pooled = scoretestbyhand(scores, np.zeros(xre.size), (xre - xre.mean())[:, None], groupsexact)
if abs(pooled[0] - hand[0]) < 100 * TOLOPT:
    sys.exit("Validation test failed at case 20: the censored data do not separate the variances pooled within kinds from one pooled variance.")

# no random effect, or an unknown method: None, with an error for the method
if quiet(modelexact.getsigmatest)[0] is not None:
    sys.exit("Validation test failed at case 20: getsigmatest of a model without random effect is not None.")
result, printed = quiet(modelrandomexact.getsigmatest, "wald")
if result is not None or "Error" not in printed:
    sys.exit("Validation test failed at case 20: getsigmatest accepted an unknown method.")


print("fitreg: All tests passed.")
