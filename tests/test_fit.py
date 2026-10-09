'''
UNIT TEST FOR THE fit CLASS (the user-facing interface that fits a continuous or
discrete phase-type distribution to data and returns it as a dist object).

The classes fit delegates to -- the E-steps, the M-steps, the random
initializers and dist -- have test files of their own, and the estimation
itself is checked there. This file therefore exercises each of fit's own
functionalities once rather than in combination: it is the wiring between the
arguments and the nested classes that is under test, not the EM algorithm.

Ground truth. The continuous data are a generalized Erlang of three phases with
distinct rates, and the discrete data a negative binomial of three phases, both
of which lie exactly inside the "generlang" family that is fitted to them. The
rates are kept distinct so the truth is not exchangeable between phases, and an
exponential sample is deliberately not used: a general phase-type model fitted
to exponential data has a flat ridge of optima and the EM crawls along it for
tens of thousands of iterations.

The two accuracy cases compare a fitted CDF against the empirical one through
Clopper-Pearson intervals rather than a hand-picked tolerance. One failed
interval fails the test, so the level is Bonferroni-corrected across every
interval the file reads. Measured over 15 seeds, both cases passed 15/15 with a
worst-case slack of 0.017 (continuous) and 0.020 (discrete) against interval
half-widths of roughly 0.08.

Sub-tests:
    Case 1:  A continuous fit agrees with the empirical CDF.
    Case 2:  A discrete fit agrees with the empirical CDF.
    Case 3:  Each distribution type produces the structure it names, and the
             structural zeros survive the EM algorithm.
    Case 4:  With randominit False the supplied start is used: two separated
             starts reach the same optimum without being identical, and the
             distribution type no longer affects the result.
    Case 5:  The seed makes a random initialization reproducible, and a
             different seed reaches the same optimum by another route.
    Case 6:  fixediter runs exactly as asked and the log-likelihood never
             decreases as the count grows.
    Case 7:  Zero observations leave an atom at zero of their own size.
    Case 8:  The reported mean, variance, density and CDF match the same
             quantities computed from the fitted parameters by hand.
    Case 9:  AIC and BIC are the same parameter count read two ways.
    Case 10: getdist returns the distribution fit reports on.
    Case 11: plot writes a file, for continuous and for discrete data, leaves
             the empirical CDF out and warns when anything is censored, closes
             the figure it wrote, draws no confidence intervals unless asked,
             and computes them as Clopper-Pearson intervals.
    Case 12: An unknown distribution type yields no fit, and infeasible input
             terminates.
    Case 13: Censoring reaches the fitting class, survives the removal of the
             zero observations with its rows still matched, and is validated.
    Case 14: verbose decides whether anything is written to standard output.

References:
    Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in
    Applied Probability. Springer. https://doi.org/10.1007/978-1-4939-7049-0
'''

import contextlib
import io
import os
import sys
import warnings

import numpy as np
import matplotlib
from scipy.linalg import expm
from scipy.stats import beta
from statsmodels.stats.proportion import proportion_confint

# a backend that needs no display, since case 11 only writes the figure to file
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Load phasedist from the src-folder so the test can be run without installing
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from phasedist.fit import fit
from phasedist.dist import dist


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

TOL = 1e-10       # tolerance for comparisons that are exact in theory
TOLAPPROX = 1e-4  # tolerance for two runs of the EM algorithm agreeing
SEED = 0

NPHASES = 3
NOBS = 300

CONFIDENCE = 0.95   # confidence level checked in case 11
EMTOLERANCE = 1e-4  # convergence tolerance for the accuracy cases
STARTTOLERANCE = 1e-5  # tighter, where two fits have to meet

# The continuous truth: a sum of three exponentials with distinct rates, i.e. a
# generalized Erlang, which the "generlang" family contains exactly.
RATES = np.array([1.0, 1.8, 3.0])

# The discrete truth: a negative binomial of three phases, shifted so the
# support starts at three, which is the discrete counterpart.
PROB = 0.4

# Points at which the fitted and empirical CDFs are compared, spread over the
# body of each distribution.
CPHPOINTS = (0.4, 0.8, 1.4, 2.2, 3.5)
DPHPOINTS = (3, 4, 5, 7, 10)

# One failed interval fails the whole test, so the level belongs to the family
# of intervals and not to each one. Measured false-alarm rate over 15 seeds:
# 0/15 for either case.
ALPHA = 0.01
NINTERVALS = len(CPHPOINTS) + len(DPHPOINTS)
LEVEL = 1.0 - ALPHA / NINTERVALS

# Two phases are enough where the point is the start rather than the fit, and a
# two-phase fit converges fast enough to run twice per case.
SMALLPHASES = 2
SMALLOBS = 200

# Two starts far enough apart that reaching the same optimum means something.
# Both are dense, so the EM preserves no zeros in either and both search the
# same family.
STARTPI1 = np.array([0.7, 0.3])
STARTGEN1 = np.array([[-1.60, 0.90],
                      [0.25, -1.10]])
STARTPI2 = np.array([0.2, 0.8])
STARTGEN2 = np.array([[-2.80, 0.60],
                      [0.80, -2.10]])

STRUCTURESTEPS = 3  # EM steps for case 3, few enough that nothing has collapsed
NZEROS = 30         # zero observations in case 7

# Every public method of fit. Checked against the class below, so that a method
# added later cannot go untested unnoticed.
COVEREDMETHODS = ("getaic", "getbic", "getcumprob", "getdensity", "getdist",
                  "getexitrates", "getinitdist", "getloglik", "getmean",
                  "getphasegen", "getquantile", "getvar", "plot")

publicmethods = tuple(sorted(name for name in dir(fit)
                             if not name.startswith("_")
                             and callable(getattr(fit, name))))

if publicmethods != tuple(sorted(COVEREDMETHODS)):
    sys.exit("Validation test failed at initialization: the public methods of fit are %s, but the test covers %s." % (publicmethods, tuple(sorted(COVEREDMETHODS))))

# The structures of case 3 are written out at NPHASES, so a change of dimension
# has to fail loudly rather than quietly skip the case.
if NPHASES != 3 or SMALLPHASES != 2:
    sys.exit("Validation test failed at initialization: the hand-written structures of case 3 and the starts of case 4 assume three and two phases respectively.")

if RATES.size != NPHASES or np.min(np.abs(np.diff(np.sort(RATES)))) < 0.5:
    sys.exit("Validation test failed at initialization: the true rates must number %d and stay apart, so that the truth is not exchangeable between phases." % NPHASES)

offdiagonalmask = ~np.eye(SMALLPHASES, dtype=bool)

for name, gen in (("first", STARTGEN1), ("second", STARTGEN2)):
    if (np.any(np.diag(gen) >= 0.0) or np.any(gen[offdiagonalmask] <= 0.0)
            or np.any(np.sum(gen, axis=1) > TOL)):
        sys.exit("Validation test failed at initialization: the %s start of case 4 is not a dense sub-intensity matrix." % name)

if abs(np.sum(STARTPI1) - 1.0) > TOL or abs(np.sum(STARTPI2) - 1.0) > TOL:
    sys.exit("Validation test failed at initialization: a start of case 4 has an initial distribution that does not sum to one.")

# the two starts have to be far enough apart for case 4 to mean anything
if (np.max(np.abs(STARTGEN1 - STARTGEN2)) < 0.5
        or np.max(np.abs(STARTPI1 - STARTPI2)) < 0.25):
    sys.exit("Validation test failed at initialization: the two starts of case 4 are too close to tell a used start from an ignored one.")

if NZEROS <= 0 or NZEROS >= NOBS:
    sys.exit("Validation test failed at initialization: case 7 needs some but not all observations to be zero.")

np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# The continuous sample is built from the raw rates rather than through any
# phase-type machinery, so it shares no code with the class under test.
CPHOBS = np.zeros(NOBS)
for rate in RATES:
    CPHOBS += rng.exponential(1.0 / rate, size=NOBS)

DPHOBS = (rng.negative_binomial(NPHASES, PROB, size=NOBS)
          + NPHASES).astype(float)

SMALLOBSERVATIONS = np.zeros(SMALLOBS)
for rate in RATES:
    SMALLOBSERVATIONS += rng.exponential(1.0 / rate, size=SMALLOBS)

if np.any(CPHOBS <= 0.0) or np.any(DPHOBS < 1.0):
    sys.exit("Validation test failed at initialization: the generated samples are not in the support of the distributions they come from.")


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def quietfit(**arguments):
    '''
    Returns a fit, with warnings and printed output suppressed. fixediter sets
    the tolerance to minus infinity, so the fitting classes always report that
    they stopped at itermax, which is expected here rather than a fault.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            return fit(**arguments)


def exitratesof(gen):
    '''
    Returns the exit rates implied by a sub-intensity matrix, i.e. minus its
    row sums.
    '''
    return -np.asarray(gen, dtype=float).sum(axis=1)


def clopperpearson(count, total, level):
    '''
    Returns the Clopper-Pearson interval for a binomial proportion. The ends
    are set directly where the beta distribution has no parameters for them.
    '''
    alpha = 1.0 - level
    lower = 0.0 if count == 0 else beta.ppf(alpha / 2.0, count, total - count + 1)
    upper = 1.0 if count == total else beta.ppf(1.0 - alpha / 2.0, count + 1,
                                                total - count)
    return float(lower), float(upper)


def parameters(model):
    '''
    Returns the fitted initial distribution, generator and exit rates as plain
    arrays. dist holds them as numpy matrices, so the containers differ from
    what the comparisons below need.
    '''
    return (np.asarray(model.getinitdist(), dtype=float).ravel(),
            np.asarray(model.getphasegen(), dtype=float),
            np.asarray(model.getexitrates(), dtype=float).ravel())


def momentsbyhand(initdist, phgen, discrete):
    '''
    Returns the mean and variance computed from the parameters through the
    Green matrix, by algebra the class under test does not share.

    Continuous: U = (-T)^-1, mean = pi U e, E(tau^2) = 2 pi U^2 e.
    Discrete:   U = (I-T)^-1, mean = pi U e, E(tau(tau-1)) = 2 pi T U^2 e.
    '''
    size = phgen.shape[0]
    ones = np.ones(size)

    if discrete:
        green = np.linalg.inv(np.eye(size) - phgen)
        mean = float(initdist @ green @ ones)
        second = float(2.0 * initdist @ phgen @ green @ green @ ones) + mean
    else:
        green = np.linalg.inv(-phgen)
        mean = float(initdist @ green @ ones)
        second = float(2.0 * initdist @ green @ green @ ones)

    return mean, second - mean ** 2


def densitybyhand(initdist, phgen, exitrates, x, discrete):
    '''
    Returns the density (continuous) or probability mass (discrete) at x,
    computed from the parameters directly.
    '''
    if discrete:
        return float(initdist
                     @ np.linalg.matrix_power(phgen, int(x) - 1)
                     @ exitrates)

    return float(initdist @ expm(phgen * float(x)) @ exitrates)


def cumprobbyhand(initdist, phgen, x, discrete):
    '''
    Returns the cumulative probability at x, computed from the parameters
    directly.
    '''
    ones = np.ones(phgen.shape[0])

    if discrete:
        return float(1.0 - initdist
                     @ np.linalg.matrix_power(phgen, int(x)) @ ones)

    return float(1.0 - initdist @ expm(phgen * float(x)) @ ones)


def cdfagrees(model, observations, points, label):
    '''
    Exits if the fitted CDF falls outside the Clopper-Pearson interval for the
    empirical CDF at any of the given points.
    '''
    for x in points:
        count = int(np.sum(observations <= x))
        lower, upper = clopperpearson(count, observations.size, LEVEL)
        fitted = model.getcumprob(x)

        if not lower <= fitted <= upper:
            sys.exit("Validation test failed at %s: at x = %s the fitted CDF is %.6f, outside the %.4f%% Clopper-Pearson interval [%.6f, %.6f] for the empirical CDF." % (label, x, fitted, 100.0 * LEVEL, lower, upper))


# ------------------------------------------------------------------
# CASE 1: A continuous fit agrees with the empirical CDF
# ------------------------------------------------------------------

continuousmodel = quietfit(obs=np.copy(CPHOBS), nphases=NPHASES,
                           dtype="generlang", seed=SEED,
                           tolerance=EMTOLERANCE)

cdfagrees(continuousmodel, CPHOBS, CPHPOINTS, "case 1")

if not continuousmodel.fitaccepted:
    sys.exit("Validation test failed at case 1: the continuous fit reports that its own parameters are infeasible.")


# ------------------------------------------------------------------
# CASE 2: A discrete fit agrees with the empirical CDF
# ------------------------------------------------------------------

discretemodel = quietfit(obs=np.copy(DPHOBS), nphases=NPHASES,
                         dtype="generlang", discrete=True, seed=SEED,
                         tolerance=EMTOLERANCE)

cdfagrees(discretemodel, DPHOBS, DPHPOINTS, "case 2")

if not discretemodel.fitaccepted:
    sys.exit("Validation test failed at case 2: the discrete fit reports that its own parameters are infeasible.")


# ------------------------------------------------------------------
# CASE 3: The distribution types and their structural zeros
# ------------------------------------------------------------------

# Which elements each type declares to be non-zero, at three phases. The
# generator's diagonal is always non-zero in the continuous case, being minus
# the sum of the row and the exit rate.
BAND = np.array([[1, 1, 0],
                 [0, 1, 1],
                 [0, 0, 1]])

STRUCTURES = (
    ("general", np.array([1, 1, 1]), np.ones((NPHASES, NPHASES), dtype=int),
     np.array([1, 1, 1])),
    ("generlang", np.array([1, 0, 0]), BAND, np.array([0, 0, 1])),
    ("hyperexp", np.array([1, 1, 1]), np.eye(NPHASES, dtype=int),
     np.array([1, 1, 1])),
    ("coxian", np.array([1, 0, 0]), BAND, np.array([1, 1, 1])),
    ("gencoxian", np.array([1, 1, 1]), BAND, np.array([1, 1, 1])),
)

if len(STRUCTURES) != 5:
    sys.exit("Validation test failed at case 3: the table of structures does not cover the five named distribution types.")

for name, pipattern, genpattern, exitpattern in STRUCTURES:
    model = quietfit(obs=np.copy(CPHOBS), nphases=NPHASES, dtype=name,
                     seed=SEED, fixediter=STRUCTURESTEPS)

    initdist, phgen, exitrates = parameters(model)

    for label, fitted, pattern in (("initial distribution", initdist, pipattern),
                                   ("generator", phgen, genpattern),
                                   ("exit rates", exitrates, exitpattern)):
        if fitted.shape != pattern.shape:
            sys.exit("Validation test failed at case 3: for the '%s' type the %s has shape %s rather than %s." % (name, label, fitted.shape, pattern.shape))

        # the EM algorithm preserves structural zeros, so they have to be zero
        # exactly and not merely small
        if np.any(fitted[pattern == 0] != 0.0):
            sys.exit("Validation test failed at case 3: for the '%s' type the %s has a non-zero where the structure is zero: %s." % (name, label, fitted))

        if np.any(fitted[pattern == 1] == 0.0):
            sys.exit("Validation test failed at case 3: for the '%s' type the %s has a zero where the structure is non-zero: %s." % (name, label, fitted))


# ------------------------------------------------------------------
# CASE 4: randominit False uses the supplied start
# ------------------------------------------------------------------

# The start has to matter and yet not determine the answer. Two fits that come
# out identical mean the start was discarded; two that disagree beyond
# tolerance mean the EM algorithm is not reaching the same optimum.
def fittedfromstart(initdist, phgen, dtype):
    '''
    Fits the small sample from a given start, with no random initialization.
    '''
    return quietfit(obs=np.copy(SMALLOBSERVATIONS), nphases=SMALLPHASES,
                    dtype=dtype, randominit=False,
                    initdist=np.copy(initdist), initphgen=np.copy(phgen),
                    initexitrates=exitratesof(phgen),
                    tolerance=STARTTOLERANCE)


fromfirst = fittedfromstart(STARTPI1, STARTGEN1, "custom")
fromsecond = fittedfromstart(STARTPI2, STARTGEN2, "custom")

if fromfirst.getloglik() == fromsecond.getloglik():
    sys.exit("Validation test failed at case 4: two different starts give byte-identical log-likelihoods, so the supplied start is being discarded.")

gap = abs(fromfirst.getloglik() - fromsecond.getloglik())

if gap > TOLAPPROX:
    sys.exit("Validation test failed at case 4: two starts reach log-likelihoods %.10f and %.10f, differing by %.3e, which is more than %.1e, so they are not reaching the same optimum." % (fromfirst.getloglik(), fromsecond.getloglik(), gap, TOLAPPROX))

# With no random initialization the start is used as given, so the named type
# is irrelevant on this path and has to leave the result untouched.
frompreset = fittedfromstart(STARTPI1, STARTGEN1, "general")

if frompreset.getloglik() != fromfirst.getloglik():
    sys.exit("Validation test failed at case 4: with randominit False the distribution type changed the fit, giving %.10f against %.10f, so a type is still overwriting the supplied start." % (frompreset.getloglik(), fromfirst.getloglik()))


# ------------------------------------------------------------------
# CASE 5: The seed
# ------------------------------------------------------------------

def fittedwithseed(seed):
    '''
    Fits the small sample from a random start drawn with the given seed.
    '''
    return quietfit(obs=np.copy(SMALLOBSERVATIONS), nphases=SMALLPHASES,
                    seed=seed, tolerance=STARTTOLERANCE)


first = fittedwithseed(SEED + 11)
repeat = fittedwithseed(SEED + 11)
other = fittedwithseed(SEED + 99)

if first.getloglik() != repeat.getloglik():
    sys.exit("Validation test failed at case 5: the same seed gave log-likelihoods %.12f and %.12f, so a random initialization is not reproducible." % (first.getloglik(), repeat.getloglik()))

if first.getloglik() == other.getloglik():
    sys.exit("Validation test failed at case 5: a different seed gave a byte-identical fit, so the seed is not reaching the random initialization.")

gap = abs(first.getloglik() - other.getloglik())

if gap > TOLAPPROX:
    sys.exit("Validation test failed at case 5: two seeds reach log-likelihoods differing by %.3e, which is more than %.1e." % (gap, TOLAPPROX))


# ------------------------------------------------------------------
# CASE 6: fixediter
# ------------------------------------------------------------------

# The EM algorithm never decreases the log-likelihood, so a longer run cannot
# do worse than a shorter one. This also confirms fixediter is what sets the
# length of the run, since the three fits share a seed and a tolerance.
ITERATIONCOUNTS = (1, 3, 10, 30)

loglikelihoods = [quietfit(obs=np.copy(SMALLOBSERVATIONS),
                           nphases=SMALLPHASES, seed=SEED,
                           fixediter=count).getloglik()
                  for count in ITERATIONCOUNTS]

for index in range(1, len(ITERATIONCOUNTS)):
    if loglikelihoods[index] < loglikelihoods[index - 1] - TOL:
        sys.exit("Validation test failed at case 6: %d iterations give a log-likelihood of %.10f against %.10f after %d, so the fit got worse with more iterations." % (ITERATIONCOUNTS[index], loglikelihoods[index], loglikelihoods[index - 1], ITERATIONCOUNTS[index - 1]))

# and the count has to make a difference, or fixediter is being ignored
if loglikelihoods[0] == loglikelihoods[-1]:
    sys.exit("Validation test failed at case 6: %d and %d iterations give the same log-likelihood, so fixediter is not setting the length of the run." % (ITERATIONCOUNTS[0], ITERATIONCOUNTS[-1]))


# ------------------------------------------------------------------
# CASE 7: Zero observations
# ------------------------------------------------------------------

# A zero observation cannot come from a phase-type distribution with no atom at
# zero, so fit removes the zeros, fits the rest, and scales the initial
# distribution down to leave an atom of exactly their proportion.
withzeros = np.copy(SMALLOBSERVATIONS)
withzeros[:NZEROS] = 0.0

zeromodel = quietfit(obs=withzeros, nphases=SMALLPHASES, seed=SEED,
                     tolerance=EMTOLERANCE)

expectedmass = 1.0 - NZEROS / float(SMALLOBS)
initdist = np.asarray(zeromodel.getinitdist(), dtype=float).ravel()

if abs(np.sum(initdist) - expectedmass) > TOL:
    sys.exit("Validation test failed at case 7: with %d of %d observations at zero the initial distribution sums to %.12f rather than %.12f." % (NZEROS, SMALLOBS, np.sum(initdist), expectedmass))

if abs(zeromodel.getcumprob(0.0) - (1.0 - expectedmass)) > TOL:
    sys.exit("Validation test failed at case 7: the fitted distribution puts %.12f at zero rather than the observed proportion %.12f." % (zeromodel.getcumprob(0.0), 1.0 - expectedmass))

# a sample with no zeros must not be given an atom
if abs(np.sum(np.asarray(continuousmodel.getinitdist(), dtype=float)) - 1.0) > TOL:
    sys.exit("Validation test failed at case 7: with no zero observations the initial distribution does not sum to one.")

# The atom makes the model a mixture, and the log-likelihood has to cover every
# observation
nonzero = withzeros[withzeros != 0.0]
byhand = float(np.sum(np.log([zeromodel.getdensity(y) for y in nonzero])))
byhand += NZEROS * np.log(zeromodel.getcumprob(0.0))

if abs(zeromodel.getloglik() - byhand) > TOL:
    sys.exit("Validation test failed at case 7: with %d of %d observations at zero the reported log-likelihood is %.10f against %.10f summed over every observation, so the zeros are left out of it." % (NZEROS, SMALLOBS, zeromodel.getloglik(), byhand))

# the proportion at zero is estimated, so it is one more parameter
if zeromodel.nparam != zeromodel.d.nparam + 1:
    sys.exit("Validation test failed at case 7: with zero observations the parameter count is %d against the %d of the phase-type part, so the atom is not counted." % (zeromodel.nparam, zeromodel.d.nparam))

if continuousmodel.nparam != continuousmodel.d.nparam:
    sys.exit("Validation test failed at case 7: with no zero observations the parameter count is %d against the %d of the phase-type part." % (continuousmodel.nparam, continuousmodel.d.nparam))

# and the criteria are built on the whole sample, not the kept part
for label, model, total in (("the zero-inflated fit", zeromodel, SMALLOBS),
                            ("the continuous fit", continuousmodel, NOBS)):
    if abs(model.getaic() - (-2.0 * model.getloglik() + 2.0 * model.nparam)) > TOL:
        sys.exit("Validation test failed at case 7: for %s the AIC is not -2L+2k." % label)

    if abs(model.getbic() - (-2.0 * model.getloglik()
                             + model.nparam * np.log(total))) > TOL:
        sys.exit("Validation test failed at case 7: for %s the BIC is not -2L+k log n with n the whole sample of %d." % (label, total))


# ------------------------------------------------------------------
# CASE 8: The reported quantities against the fitted parameters
# ------------------------------------------------------------------

for label, model, discrete, points in (
        ("the continuous fit", continuousmodel, False, CPHPOINTS),
        ("the discrete fit", discretemodel, True, DPHPOINTS)):

    initdist, phgen, exitrates = parameters(model)

    mean, variance = momentsbyhand(initdist, phgen, discrete)

    if abs(model.getmean() - mean) > TOL:
        sys.exit("Validation test failed at case 8: for %s the reported mean %.10f differs from the mean of its own parameters, %.10f." % (label, model.getmean(), mean))

    if abs(model.getvar() - variance) > TOL:
        sys.exit("Validation test failed at case 8: for %s the reported variance %.10f differs from the variance of its own parameters, %.10f." % (label, model.getvar(), variance))

    for x in points:
        expected = densitybyhand(initdist, phgen, exitrates, x, discrete)

        if abs(model.getdensity(x) - expected) > TOL:
            sys.exit("Validation test failed at case 8: for %s the reported density at %s is %.10f against %.10f computed from its own parameters." % (label, x, model.getdensity(x), expected))

        expected = cumprobbyhand(initdist, phgen, x, discrete)

        if abs(model.getcumprob(x) - expected) > TOL:
            sys.exit("Validation test failed at case 8: for %s the reported cumulative probability at %s is %.10f against %.10f computed from its own parameters." % (label, x, model.getcumprob(x), expected))

    # the exit rates have to be the ones the generator implies, or the three
    # reported parameters do not describe one distribution
    if not discrete and np.max(np.abs(exitrates - exitratesof(phgen))) > TOL:
        sys.exit("Validation test failed at case 8: for %s the reported exit rates are not minus the row sums of the reported generator." % label)

    # the quantile has to invert the CDF it is reported alongside
    for probability in (0.25, 0.5, 0.75):
        quantile = model.getquantile(probability)

        if discrete:
            # the discrete quantile is the smallest n with F(n) >= p
            if model.getcumprob(quantile) < probability:
                sys.exit("Validation test failed at case 8: for %s the %.2f-quantile is %s, where the CDF is only %.6f." % (label, probability, quantile, model.getcumprob(quantile)))
        elif abs(model.getcumprob(quantile) - probability) > 1e-5:
            sys.exit("Validation test failed at case 8: for %s the CDF at the reported %.2f-quantile is %.8f." % (label, probability, model.getcumprob(quantile)))


# ------------------------------------------------------------------
# CASE 9: AIC and BIC
# ------------------------------------------------------------------

# AIC = -2L + 2k and BIC = -2L + k log n are the same parameter count read two
# ways, so solving the pair for k has to return the whole number of parameters
# and reproduce both criteria.
for label, model, observations in (("the continuous fit", continuousmodel, CPHOBS),
                                   ("the discrete fit", discretemodel, DPHOBS)):
    loglik = model.getloglik()
    nparam = ((model.getbic() - model.getaic())
              / (np.log(observations.size) - 2.0))

    if abs(nparam - round(nparam)) > TOL or round(nparam) < 1:
        sys.exit("Validation test failed at case 9: for %s AIC and BIC imply a parameter count of %.10f, which is not a positive whole number." % (label, nparam))

    if abs(model.getaic() - (-2.0 * loglik + 2.0 * nparam)) > TOL:
        sys.exit("Validation test failed at case 9: for %s the AIC is %.10f rather than -2L+2k = %.10f." % (label, model.getaic(), -2.0 * loglik + 2.0 * nparam))

    if abs(model.getbic() - (-2.0 * loglik
                             + nparam * np.log(observations.size))) > TOL:
        sys.exit("Validation test failed at case 9: for %s the BIC is %.10f rather than -2L+k log n = %.10f." % (label, model.getbic(), -2.0 * loglik + nparam * np.log(observations.size)))


# ------------------------------------------------------------------
# CASE 10: getdist
# ------------------------------------------------------------------

for label, model, points in (("the continuous fit", continuousmodel, CPHPOINTS),
                             ("the discrete fit", discretemodel, DPHPOINTS)):
    returned = model.getdist()

    if not isinstance(returned, dist):
        sys.exit("Validation test failed at case 10: for %s getdist returned a %s rather than a dist." % (label, type(returned).__name__))

    if abs(returned.getmean() - model.getmean()) > TOL:
        sys.exit("Validation test failed at case 10: for %s the returned distribution has mean %.10f against the reported %.10f." % (label, returned.getmean(), model.getmean()))

    for x in points:
        if abs(returned.getcumprob(x) - model.getcumprob(x)) > TOL:
            sys.exit("Validation test failed at case 10: for %s the returned distribution and the fit disagree on the cumulative probability at %s." % (label, x))


# ------------------------------------------------------------------
# CASE 11: plot
# ------------------------------------------------------------------

# The continuous and discrete branches of plot share almost no code, so both
# are exercised, censored and not. The figures are only written to file.
PLOTCENSORING = np.full((SMALLOBS, 2), np.nan)
PLOTCENSORING[:40, 1] = 1.0

DISCRETEPLOTCENSORING = np.full((NOBS, 2), np.nan)
DISCRETEPLOTCENSORING[:80, 1] = 5.0

plotcensored = quietfit(obs=np.copy(SMALLOBSERVATIONS),
                        censoring=np.copy(PLOTCENSORING),
                        nphases=SMALLPHASES, seed=SEED, fixediter=1)
plotcensoreddiscrete = quietfit(obs=np.copy(DPHOBS),
                                censoring=np.copy(DISCRETEPLOTCENSORING),
                                nphases=NPHASES, discrete=True, seed=SEED,
                                fixediter=1)

plotted = []

try:
    for label, model, mustwarn in (("continuous", continuousmodel, False),
                                   ("discrete", discretemodel, False),
                                   ("censored_continuous", plotcensored, True),
                                   ("censored_discrete", plotcensoreddiscrete,
                                    True)):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "test_fit_%s_cdf.png" % label)
        plotted.append(path)

        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            model.plot(filename=path)

        warned = any(issubclass(entry.category, UserWarning) for entry in raised)

        if warned and not mustwarn:
            sys.exit("Validation test failed at case 11: plotting the %s fit warned about censoring, although nothing is censored." % label)

        if mustwarn and not warned:
            sys.exit("Validation test failed at case 11: plotting the %s fit drew an empirical CDF from placeholder values without warning." % label)

        if not os.path.exists(path):
            sys.exit("Validation test failed at case 11: plotting the %s fit wrote no file." % label)

        if os.path.getsize(path) == 0:
            sys.exit("Validation test failed at case 11: plotting the %s fit wrote an empty file." % label)

    # the figure is closed once written, so repeated calls cannot pile up
    openfigures = len(plt.get_fignums())

    for _ in range(3):
        continuousmodel.plot(filename=plotted[0])

    if len(plt.get_fignums()) != openfigures:
        sys.exit("Validation test failed at case 11: three further calls left %d figures open against %d before, so plot does not close the figure it wrote." % (len(plt.get_fignums()), openfigures))

    # the intervals are off unless asked for, so the default figure has to be
    # the one drawn with them off and not the one drawn with them on
    figures = {}

    for label, arguments in (("default", {}), ("off", {"confint": False}),
                             ("on", {"confint": True})):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "test_fit_confint_%s.png" % label)
        plotted.append(path)
        continuousmodel.plot(filename=path, **arguments)

        with open(path, "rb") as stream:
            figures[label] = stream.read()

    if figures["default"] != figures["off"]:
        sys.exit("Validation test failed at case 11: the figure drawn by default differs from the one drawn with confint False, so the confidence intervals are not off by default.")

    if figures["default"] == figures["on"]:
        sys.exit("Validation test failed at case 11: asking for confidence intervals changed nothing in the figure.")

    # The intervals are Clopper-Pearson, checked against the exact method in
    # statsmodels. The two ends are included, the beta distribution having no
    # parameters there and statsmodels returning a nan.
    CPCOUNTS = np.array([0, 1, 37, NOBS - 1, NOBS])

    lower, upper = continuousmodel._fit__clopperpearson(CPCOUNTS, CONFIDENCE)

    for index, count in enumerate(CPCOUNTS):
        reference = proportion_confint(count, NOBS, alpha=1.0 - CONFIDENCE,
                                       method="beta")
        wantlower = 0.0 if count == 0 else reference[0]
        wantupper = 1.0 if count == NOBS else reference[1]

        if abs(lower[index] - wantlower) > TOL or abs(upper[index] - wantupper) > TOL:
            sys.exit("Validation test failed at case 11: for %d of %d observations the interval is [%.10f, %.10f] against the Clopper-Pearson interval [%.10f, %.10f]." % (count, NOBS, lower[index], upper[index], wantlower, wantupper))

    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        sys.exit("Validation test failed at case 11: the Clopper-Pearson interval is not a number at one of the ends.")
finally:
    for path in plotted:
        if os.path.exists(path):
            os.remove(path)


# ------------------------------------------------------------------
# CASE 12: Input that has to be refused
# ------------------------------------------------------------------

# An unknown type cannot be fitted, so no distribution may come out of it. The
# attribute is the one every reported quantity is read from.
unknown = quietfit(obs=np.copy(SMALLOBSERVATIONS), nphases=SMALLPHASES,
                   dtype="nosuchtype", seed=SEED, fixediter=1)

if hasattr(unknown, "dist"):
    sys.exit("Validation test failed at case 12: an unknown distribution type still produced a fitted distribution.")

# Infeasible input terminates rather than returning something unusable.
negative = np.copy(SMALLOBSERVATIONS)
negative[0] = -1.0

try:
    quietfit(obs=negative, nphases=SMALLPHASES, seed=SEED, fixediter=1)
    sys.exit("Validation test failed at case 12: a negative observation was accepted.")
except SystemExit as termination:
    if termination.code != 1:
        raise


# ------------------------------------------------------------------
# CASE 13: Censoring
# ------------------------------------------------------------------

CENSORLIMIT = 1.3   # a fixed limit, so the censoring is not informative
NCENSORED = 60      # rows censored in the identity below
ZEROSTEP = 7        # every seventh observation set to zero
CENSORSTEP = 11     # every eleventh row right-censored

# all four kinds in one array
fourkinds = np.full((SMALLOBS, 2), np.nan)
fourkinds[0:30, 1] = 1.0
fourkinds[30:60, 0] = 2.0
fourkinds[60:90, 0] = 0.8
fourkinds[60:90, 1] = 2.2

lowermissing = np.isnan(fourkinds[:, 0])
uppermissing = np.isnan(fourkinds[:, 1])
kindcounts = (int(np.sum(lowermissing & uppermissing)),
              int(np.sum(lowermissing & ~uppermissing)),
              int(np.sum(~lowermissing & uppermissing)),
              int(np.sum(~lowermissing & ~uppermissing)))

if min(kindcounts) == 0:
    sys.exit("Validation test failed at case 13: the censoring array does not hold all four kinds, the uncensored, right, left and interval counts being %s." % (kindcounts,))

censored = quietfit(obs=np.copy(SMALLOBSERVATIONS), censoring=np.copy(fourkinds),
                    nphases=SMALLPHASES, seed=SEED, tolerance=STARTTOLERANCE)
uncensored = quietfit(obs=np.copy(SMALLOBSERVATIONS), nphases=SMALLPHASES,
                      seed=SEED, tolerance=STARTTOLERANCE)

if censored.getloglik() == uncensored.getloglik():
    sys.exit("Validation test failed at case 13: censoring left the fit unchanged, so the censoring array is not reaching the fitting class.")

if not censored.fitaccepted:
    sys.exit("Validation test failed at case 13: the censored fit reports that its own parameters are infeasible.")

# Left censoring is interval censoring with lower limit zero, exactly. The two
# take different routes through the E-step, so agreement to the last digit says
# the array arrives intact and row for row.
leftonly = np.full((SMALLOBS, 2), np.nan)
leftonly[:NCENSORED, 0] = CENSORLIMIT

asinterval = np.full((SMALLOBS, 2), np.nan)
asinterval[:NCENSORED, 0] = 0.0
asinterval[:NCENSORED, 1] = CENSORLIMIT

fromleft = quietfit(obs=np.copy(SMALLOBSERVATIONS), censoring=leftonly,
                    nphases=SMALLPHASES, seed=SEED, tolerance=STARTTOLERANCE)
frominterval = quietfit(obs=np.copy(SMALLOBSERVATIONS), censoring=asinterval,
                        nphases=SMALLPHASES, seed=SEED,
                        tolerance=STARTTOLERANCE)

# the two routes accumulate round-off differently, so they agree to TOL
# rather than to the last bit
if abs(fromleft.getloglik() - frominterval.getloglik()) > TOL:
    sys.exit("Validation test failed at case 13: left censoring gives %.14f where the same thing as interval censoring from zero gives %.14f, and the two are equal in theory." % (fromleft.getloglik(), frominterval.getloglik()))

# Removing the zero observations has to carry the censoring array with it. The
# zeros are scattered among censored rows, and the fit has to match the one
# obtained by removing them beforehand.
mixedobs = np.copy(SMALLOBSERVATIONS)
mixedobs[::ZEROSTEP] = 0.0

mixedcensoring = np.full((SMALLOBS, 2), np.nan)
mixedcensoring[3::CENSORSTEP, 1] = 1.0
mixedcensoring[::ZEROSTEP, :] = np.nan  # the zeros are left uncensored

iscensored = ~np.all(np.isnan(mixedcensoring), axis=1)
kept = (mixedobs != 0.0) | iscensored

if np.count_nonzero(~kept) == 0 or np.count_nonzero(iscensored) == 0:
    sys.exit("Validation test failed at case 13: the mixed sample has to hold both zero observations and censored rows, but holds %d and %d." % (np.count_nonzero(~kept), np.count_nonzero(iscensored)))

stripped = quietfit(obs=mixedobs[kept].copy(),
                    censoring=mixedcensoring[kept].copy(),
                    nphases=SMALLPHASES, seed=SEED, tolerance=STARTTOLERANCE)
mixed = quietfit(obs=np.copy(mixedobs), censoring=np.copy(mixedcensoring),
                 nphases=SMALLPHASES, seed=SEED, tolerance=STARTTOLERANCE)

# the phase-type part is what the matching affects; the two full
# log-likelihoods differ by the atom, which only the first fit has
if abs(mixed.d.getloglik() - stripped.d.getloglik()) > TOL:
    sys.exit("Validation test failed at case 13: removing the zero observations gives %.14f where removing them beforehand gives %.14f, so the censoring rows are no longer matched to the observations." % (mixed.d.getloglik(), stripped.d.getloglik()))

fraczero = np.count_nonzero(~kept) / float(SMALLOBS)
mixedinitdist = np.asarray(mixed.getinitdist(), dtype=float).ravel()
strippedinitdist = np.asarray(stripped.getinitdist(), dtype=float).ravel()

if np.max(np.abs(mixedinitdist - strippedinitdist * (1.0 - fraczero))) > TOL:
    sys.exit("Validation test failed at case 13: the atom at zero is not the uncensored zeros alone, the initial distribution being %s against %s scaled by %.6f." % (mixedinitdist, strippedinitdist, 1.0 - fraczero))

# A value in obs is a real observation at zero only when the censoring row is
# missing in both columns. Otherwise it is a placeholder and the row stays,
# whether it holds a zero or a nan. The sample below is small enough to read,
# so the rows that survive and the atom they leave are written out rather than
# derived. d is the fitting class fit handed them to, and the continuous path
# does not reorder the observations.
MARKEDOBS = np.array([0.0, 0.0, np.nan, 7.0, 0.0, 9.0])
MARKEDCENSORING = np.array([[np.nan, 1.0],        # right, kept
                            [0.5, 2.0],           # interval, kept
                            [np.nan, 3.0],        # right, kept
                            [np.nan, np.nan],     # uncensored 7, kept
                            [np.nan, np.nan],     # uncensored zero, dropped
                            [np.nan, np.nan]])    # uncensored 9, kept

SURVIVINGOBS = np.array([0.0, 0.0, np.nan, 7.0, 9.0])
SURVIVINGCENSORING = np.array([[np.nan, 1.0],
                               [0.5, 2.0],
                               [np.nan, 3.0],
                               [np.nan, np.nan],
                               [np.nan, np.nan]])
SURVIVINGATOM = 1.0 / 6.0

marked = quietfit(obs=np.copy(MARKEDOBS), censoring=np.copy(MARKEDCENSORING),
                  nphases=SMALLPHASES, seed=SEED, fixediter=1)

handedobs = np.asarray(marked.d.obs, dtype=float)
handedcensoring = np.asarray(marked.d.censoring, dtype=float)

if not np.array_equal(handedobs, SURVIVINGOBS, equal_nan=True):
    sys.exit("Validation test failed at case 13: of the observations %s, those handed to the fitting class were %s rather than %s, so a row was dropped on the strength of its value in obs rather than its censoring row." % (MARKEDOBS, handedobs, SURVIVINGOBS))

if not np.array_equal(handedcensoring, SURVIVINGCENSORING, equal_nan=True):
    sys.exit("Validation test failed at case 13: the censoring rows handed to the fitting class were\n%s\nrather than\n%s" % (handedcensoring, SURVIVINGCENSORING))

if abs(marked.getcumprob(0.0) - SURVIVINGATOM) > TOL:
    sys.exit("Validation test failed at case 13: one of six observations is an uncensored zero, so the atom at zero should be %.10f, but it is %.10f." % (SURVIVINGATOM, marked.getcumprob(0.0)))

# a censored row carries no usable observation, so its place in obs may hold
# anything
placeholders = np.copy(SMALLOBSERVATIONS)
placeholders[:NCENSORED] = 0.0

fromplaceholders = quietfit(obs=placeholders, censoring=leftonly,
                            nphases=SMALLPHASES, seed=SEED,
                            tolerance=STARTTOLERANCE)

if abs(fromplaceholders.getloglik() - fromleft.getloglik()) > TOL:
    sys.exit("Validation test failed at case 13: zeroing the observations of the censored rows changed the fit, so their values are not being ignored.")

# the discrete path takes censoring too
discretecensoring = np.full((NOBS, 2), np.nan)
discretecensoring[:80, 1] = 5.0

discretecensored = quietfit(obs=np.copy(DPHOBS), censoring=discretecensoring,
                            nphases=NPHASES, dtype="generlang", discrete=True,
                            seed=SEED, tolerance=EMTOLERANCE)

if discretecensored.getloglik() == discretemodel.getloglik():
    sys.exit("Validation test failed at case 13: censoring left the discrete fit unchanged.")

if not discretecensored.fitaccepted:
    sys.exit("Validation test failed at case 13: the censored discrete fit reports that its own parameters are infeasible.")

# a censoring array that cannot be matched to the observations is refused
for label, badcensoring in (("one column", np.full((SMALLOBS, 1), np.nan)),
                            ("too few rows", np.full((SMALLOBS - 1, 2), np.nan)),
                            ("too many rows", np.full((SMALLOBS + 1, 2), np.nan)),
                            ("not an array", "nonsense")):
    try:
        quietfit(obs=np.copy(SMALLOBSERVATIONS), censoring=badcensoring,
                 nphases=SMALLPHASES, seed=SEED, fixediter=1)
        sys.exit("Validation test failed at case 13: a censoring array with %s was accepted." % label)
    except SystemExit as termination:
        if termination.code != 1:
            raise


# ------------------------------------------------------------------
# CASE 14: verbose
# ------------------------------------------------------------------

# The fitting classes report every 25 iterations, so the run has to be at least
# that long for there to be anything to find.
VERBOSESTEPS = 25

verbosepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "test_fit_verbose.txt")

try:
    for flag in (True, False):
        with open(verbosepath, "w") as stream:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(stream):
                    fit(obs=np.copy(SMALLOBSERVATIONS), nphases=SMALLPHASES,
                        seed=SEED, fixediter=VERBOSESTEPS, verbose=flag)

        written = os.path.getsize(verbosepath)

        if flag and written == 0:
            sys.exit("Validation test failed at case 14: with verbose on, the fit wrote nothing to standard output.")

        if not flag and written > 0:
            sys.exit("Validation test failed at case 14: with verbose off, the fit still wrote %d bytes to standard output." % written)
finally:
    if os.path.exists(verbosepath):
        os.remove(verbosepath)


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print("fit: All tests passed.")
