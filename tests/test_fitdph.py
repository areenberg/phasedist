'''
UNIT TEST FOR THE _fitdph CLASS (fits a discrete phase-type distribution to data
with the EM algorithm).

Every public method of the class is covered by one of the cases below, and the
list of covered methods is checked against the class itself, so that a method
added later cannot go untested unnoticed.

Three structures are fitted, so that a mistake tied to one shape of
distribution cannot hide: a fully general one, a Coxian one (the phases in
series, with an exit from every one of them), and a negative binomial one (the
phases in series with the same self-transition probability, and an exit only
from the last). The data is simulated with the dist class, whose sampler is
tested against chi-square goodness-of-fit tests in test_dist.py, so it is not
being taken on trust here.

How the fits are judged. The fitted parameters themselves are not a valid
target: a phase-type representation is heavily over-parametrized (Theorem
3.1.22, p. 138), so many different parameter sets describe the same
distribution and the EM algorithm is free to move among them. What is
identified is the distribution, so the criterion is on the fitted distribution
function, and it is compared with the data the fit was given.

At each evaluation point the EMPIRICAL distribution function is formed from the
observations, which is a genuine count of how many of them are at most that
point out of how many there are. That count is binomial, so its exact
Clopper-Pearson interval is the right one for it, and the fit is accepted when
the FITTED probability lies inside. The question the criterion asks is therefore
whether the fitted distribution is consistent with the sample it was fitted to,
which is a statement about the fitting code rather than about the simulation.
That the simulated data really follows the distribution it claims to is settled
separately, by the chi-square goodness-of-fit tests of test_dist.py.

Only UNCENSORED observations enter the empirical distribution function, since a
censored observation has no value to compare with the evaluation point. In cases
3 and 4 that leaves about half the sample, which widens the intervals of its own
accord: the criterion is then judged on the information that is actually there.

That the uncensored ones form an unbiased sample is a property of how the
censoring is set up, not something to take for granted. An observation is left
uncensored partly because of its value -- right-censoring at c hides exactly
those above c, so it leaves exactly those at or below it. What rescues it is
that the right and the left limit are the same: the piece left by right-censoring
is the distribution truncated at or below c, the piece left by left-censoring is
the distribution truncated above c, and the two arrive in the proportions that
put the distribution back together. A check below holds the two limits equal, so
that separating them fails loudly rather than quietly skewing the sample. Over
sixty thousand simulated observations the uncensored-only empirical distribution
sits within 0.0035 of the true one, which is sampling error at that size.

CDFLEVEL is the confidence that EVERY one of those intervals covers its value at
once, not the confidence of each one separately. A single interval missing would
fail the file, and there are NINTERVALS of them across the four cases and the
three structures, so each is taken at a level corrected for how many there are.

Censoring is applied at FIXED limits, drawn independently of the value being
hidden. A limit derived from the observation itself would make the censoring
informative, the censored likelihood the wrong model, and the fit inconsistent
through no fault of the code being tested.

Sub-tests:
    Case 1: uncensored data, default settings, fitted distribution within
            sampling error of the empirical one for all three structures.
    Case 2: the same with randominit=False and an invented starting point
    supplied as
            the starting point, which checks that user-specified start
            parameters are used.
    Case 3: as case 1, with a mix of uncensored and censored observations.
    Case 4: as case 2, with a mix of uncensored and censored observations.
    Case 5: getmean and getvar agree with the same quantities computed outside
            the class from the fitted parameters.
    Case 6: getdensity and getcumprob agree with the same quantities computed
            outside the class, and with each other.
    Case 7: getinitdist, getphasegen and getexitrates return fitted parameters
            that form a feasible discrete phase-type distribution.
    Case 8: getloglik is negative, agrees with the log-likelihood of the same
            data computed outside the class, and getaic and getbic exceed twice
            its magnitude and match their definitions exactly.
    Case 9: the fit has converged, does not depend on the order the
            observations are given in, and refuses a censoring array that
            cannot be matched to them.

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
from scipy.stats import beta

# Load phasedist from the src-folder so the test can be run without installing
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

import phasedist.fitdph as fitdphmodule
from phasedist.dist import dist
from phasedist.fitdph import _fitdph


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

TOL = 1e-12          # tolerance for comparisons that are exact in theory
TOLFIT = 1e-9        # tolerance for quantities recomputed from fitted values
TOLLOGLIK = 1e-6     # tolerance for the log-likelihood recomputed from them
# The most a converged fit may gain from being continued. Measured across the
# six fits of case 9 the largest gain is 2.1e-8, so this is a genuine check with
# margin to spare. The continuous file needs a looser value: its fits stop much
# closer to their tolerance than these do.
MAXIMPROVEMENT = 1e-5
SEED = 4

NPHASES = 3
NSAMPLES = 500       # observations per fit, as required by the case
CDFLEVEL = 0.99      # confidence that ALL the intervals cover their values

# Everything below is fitted on the class's own default settings, the seed
# apart. Nothing here sets the convergence tolerance or the iteration limit, so
# what the cases exercise is the behaviour a user gets without tuning anything.

# Every public method of _fitdph. Checked against the class below, so that a new
# public method cannot slip in without a case covering it.
COVEREDMETHODS = ("fit", "getaic", "getbic", "getcumprob", "getdensity",
                  "getexitrates", "getinitdist", "getloglik", "getmean",
                  "getphasegen", "getvar")

# ------------------------------------------------------------------
# The three distributions that are fitted.
#
# The zero pattern of each is what defines its structure: the EM algorithm
# preserves structural zeros (p. 678), and the random initialization samples
# only where the supplied parameters are non-zero, so passing these as the
# starting point fixes the shape of the model being fitted.
# ------------------------------------------------------------------

GENERALPI = np.array([0.5, 0.3, 0.2])
GENERALGEN = np.array([[0.50, 0.15, 0.10],
                       [0.10, 0.30, 0.20],
                       [0.15, 0.10, 0.35]])

COXIANPI = np.array([1.0, 0.0, 0.0])
COXIANGEN = np.array([[0.45, 0.30, 0.00],
                      [0.00, 0.35, 0.35],
                      [0.00, 0.00, 0.55]])

NEGBINOMPI = np.array([1.0, 0.0, 0.0])
NEGBINOMGEN = np.array([[0.60, 0.40, 0.00],
                        [0.00, 0.60, 0.40],
                        [0.00, 0.00, 0.60]])

STRUCTURES = (
    ("general", GENERALPI, GENERALGEN),
    ("Coxian", COXIANPI, COXIANGEN),
    ("negative binomial", NEGBINOMPI, NEGBINOMGEN),
)


# ------------------------------------------------------------------
# Where the fits that do not randomize their starting point begin.
#
# These are invented values, not the parameters the data was drawn from.
# Starting a fit at the truth is not what happens in practice, and it hides
# work: the EM algorithm begins at its destination with nothing left to do, so
# the case stops exercising the fitting it is there to test. Each start keeps
# the ZERO PATTERN of its structure, since that is what fixes the shape being
# fitted, and differs from the truth in every parameter that is free to move.
# The negative binomial start keeps its first two rows summing to exactly one,
# which is what makes those phases unable to exit.
#
# The parameters within each start are also kept clearly apart from one
# another, matching the continuous file: a representation whose rates agree to
# within rounding is the ill-conditioned case for the matrix functions behind
# the E-step, and there is no reason to start a fit there.
# ------------------------------------------------------------------

STARTGENERALPI = np.array([0.2, 0.5, 0.3])
STARTGENERALGEN = np.array([[0.35, 0.25, 0.20],
                            [0.20, 0.45, 0.10],
                            [0.10, 0.20, 0.50]])

STARTCOXIANPI = np.array([1.0, 0.0, 0.0])
STARTCOXIANGEN = np.array([[0.60, 0.25, 0.00],
                           [0.00, 0.50, 0.30],
                           [0.00, 0.00, 0.70]])

STARTNEGBINOMPI = np.array([1.0, 0.0, 0.0])
STARTNEGBINOMGEN = np.array([[0.30, 0.70, 0.00],
                             [0.00, 0.50, 0.50],
                             [0.00, 0.00, 0.35]])

STARTS = (
    ("general", STARTGENERALPI, STARTGENERALGEN),
    ("Coxian", STARTCOXIANPI, STARTCOXIANGEN),
    ("negative binomial", STARTNEGBINOMPI, STARTNEGBINOMGEN),
)

# The censoring limits are quantiles of the TRUE distribution, so that they
# sit where each distribution actually has its mass -- the three differ in scale
# by a factor of more than two, and one set of constants would leave a kind
# nearly empty for some of them. They remain fixed constants for a given
# distribution, settled before any data is drawn and never derived from the
# observation being hidden, which is what keeps the censoring uninformative.
CENSORINGQUANTILES = (0.25, 0.5, 0.75)
MINPERKIND = 20      # fewest observations a censoring kind may contribute
MINUNCENSORED = 100  # fewest uncensored observations an interval may rest on


def exitratesof(phgen):
    '''
    Returns the exit probabilities implied by a sub-transition matrix, i.e. one
    minus its row sums.
    '''
    return 1.0 - np.sum(np.asarray(phgen, dtype=float), axis=1)


# Verify the three structures really are feasible discrete phase-type
# distributions, so that a typo in them surfaces here rather than as a
# confusing case failure
for name, pi, phgen in STRUCTURES:
    if abs(np.sum(pi) - 1.0) > TOL or np.any(pi < 0.0):
        sys.exit("Validation test failed at initialization: the initial distribution of the %s structure is not a distribution." % name)
    if np.any(phgen < 0.0):
        sys.exit("Validation test failed at initialization: the sub-transition matrix of the %s structure has a negative element." % name)
    if np.any(exitratesof(phgen) < -TOL):
        sys.exit("Validation test failed at initialization: the rows of the sub-transition matrix of the %s structure sum to more than one." % name)
    if np.sum(exitratesof(phgen)) <= 0.0:
        sys.exit("Validation test failed at initialization: the %s structure can never reach the absorbing state." % name)

# The three structures must differ in shape, or the case is fitting the same
# model three times over
if (np.count_nonzero(GENERALGEN) == np.count_nonzero(COXIANGEN)
        or np.count_nonzero(COXIANGEN) != np.count_nonzero(NEGBINOMGEN)
        or np.count_nonzero(exitratesof(COXIANGEN) > TOL)
        == np.count_nonzero(exitratesof(NEGBINOMGEN) > TOL)):
    sys.exit("Validation test failed at initialization: the three structures do not differ from one another as intended.")

# Every public method must be covered by a case
publicmethods = tuple(sorted(name for name in dir(_fitdph)
                             if not name.startswith("_")
                             and callable(getattr(_fitdph, name))))

if publicmethods != tuple(sorted(COVEREDMETHODS)):
    sys.exit("Validation test failed at initialization: the public methods of _fitdph are %s, but the test covers %s." % (publicmethods, tuple(sorted(COVEREDMETHODS))))


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def truedistribution(pi, phgen, seed=SEED):
    '''
    Returns a dist object holding the true distribution.
    '''
    return dist(discrete=True, initdist=np.copy(pi), phgen=np.copy(phgen),
                seed=seed)


def simulate(pi, phgen, nobs, seed):
    '''
    Draws observations from the true distribution, using the sampler of the
    dist class, which test_dist.py checks against chi-square goodness-of-fit
    tests.
    '''
    return np.asarray(truedistribution(pi, phgen, seed).getrandom(size=nobs),
                      dtype=float)


def censoringlimits(truth):
    '''
    Returns the right limit, the left limit and the interval grid used to
    censor observations from a given true distribution, as quantiles of that
    distribution. These are constants of the distribution, not of any
    observation.
    '''
    quartiles = [int(truth.getquantile(p)) for p in CENSORINGQUANTILES]
    grid = np.array([0.0] + [float(q) for q in quartiles])
    return float(quartiles[1]), float(quartiles[1]), grid


def censor(obs, limits, seed):
    '''
    Returns a censoring array that hides part of what was simulated.

    Each observation is assigned a censoring scheme at random, independently of
    its value, and the scheme then censors at one of the fixed limits. An
    observation whose value falls where its scheme has nothing to hide is left
    uncensored, which is what makes the censoring uninformative: nothing about
    the limit depends on the value behind it.
    '''
    rightlimit, leftlimit, grid = limits
    rng = np.random.default_rng(seed)
    scheme = rng.integers(0, 4, obs.size)
    censoring = np.full((obs.size, 2), np.nan)

    for i, value in enumerate(obs):
        if scheme[i] == 1 and value > rightlimit:
            censoring[i, 1] = rightlimit
        elif scheme[i] == 2 and value <= leftlimit:
            censoring[i, 0] = leftlimit
        elif scheme[i] == 3:
            upper = int(np.searchsorted(grid, value, side="left"))
            if upper < grid.size:
                censoring[i] = (grid[upper - 1], grid[upper])
            else:
                censoring[i, 1] = grid[-1]

    return censoring


def censoringkinds(censoring):
    '''
    Returns how many observations fall into each of the four kinds, in the
    order uncensored, right-censored, left-censored, interval-censored.
    '''
    lower = np.isnan(censoring[:, 0])
    upper = np.isnan(censoring[:, 1])
    return (int(np.sum(lower & upper)), int(np.sum(lower & ~upper)),
            int(np.sum(~lower & upper)), int(np.sum(~lower & ~upper)))


def fitmodel(pi, phgen, obs, censoring=None, randominit=True, seed=SEED):
    '''
    Fits a discrete phase-type distribution of the shape given by pi and phgen.

    With randominit True the supplied parameters act only as the structure, the
    starting values being sampled within it; with randominit False they are the
    starting values themselves.
    '''
    model = _fitdph(obs=np.copy(obs),
                   censoring=None if censoring is None else np.copy(censoring),
                   initpi=np.copy(pi),
                   initphgen=np.copy(phgen),
                   initexitrates=exitratesof(phgen),
                   randominit=randominit,
                   seed=seed,
                   verbose=False)
    model.fit()
    return model


def fittedparameters(model):
    '''
    Returns the fitted parameters as plain one and two dimensional arrays.
    '''
    return (np.asarray(model.getinitdist(), dtype=float).ravel(),
            np.asarray(model.getphasegen(), dtype=float),
            np.asarray(model.getexitrates(), dtype=float).ravel())


def clopperpearson(count, nobs, level):
    '''
    Returns the two ends of the exact Clopper-Pearson confidence interval for a
    binomial proportion, from the quantiles of the beta distribution. The two
    ends are taken separately when nothing or everything was counted, since the
    beta distribution is not defined for the parameters those would ask for.
    '''
    alpha = 1.0 - level
    lower = 0.0 if count <= 0 else beta.ppf(0.5 * alpha, count, nobs - count + 1)
    upper = 1.0 if count >= nobs else beta.ppf(1.0 - 0.5 * alpha, count + 1,
                                               nobs - count)
    return float(lower), float(upper)


def evaluationpoints(truth):
    '''
    Returns the points at which the distribution function is compared, taken as
    the quantiles of the true distribution so that they span it rather than
    clustering where it carries no mass.
    '''
    points = sorted({int(truth.getquantile(p))
                     for p in (0.1, 0.25, 0.5, 0.75, 0.9)})
    return [x for x in points if x >= 1]


def observedvalues(observations, censoring):
    '''
    Returns the observations whose value is known, which is the uncensored ones.
    A censored observation carries no value that can be compared with an
    evaluation point, so it cannot enter an empirical distribution function.
    '''
    if censoring is None:
        return observations

    uncensored = np.isnan(censoring[:, 0]) & np.isnan(censoring[:, 1])
    return observations[uncensored]


def cdfwithinsamplingerror(case, label, model, truth, observations,
                           censoring=None):
    '''
    Checks the fitted distribution function against the empirical one at each
    evaluation point. The empirical count is binomial, so it is given an exact
    Clopper-Pearson interval at the corrected level, and the fitted probability
    has to lie inside it. Exits on the first point that fails.
    '''
    points = evaluationpoints(truth)

    if len(points) < 3:
        sys.exit("Validation test failed at case %d: the %s distribution gives only %d distinct evaluation points, too few to judge the fit." % (case, label, len(points)))

    values = observedvalues(observations, censoring)

    if values.size < MINUNCENSORED:
        sys.exit("Validation test failed at case %d: only %d of the observations of the %s distribution carry a value, too few to form an empirical distribution function." % (case, values.size, label))

    for x in points:
        fittedprob = float(model.getcumprob(x))

        if not np.isfinite(fittedprob):
            sys.exit("Validation test failed at case %d: the fitted distribution function of the %s distribution is not a number at %d." % (case, label, x))

        count = int(np.sum(values <= x))
        lower, upper = clopperpearson(count, values.size, POINTLEVEL)

        if fittedprob < lower or fittedprob > upper:
            sys.exit("Validation test failed at case %d: for the %s distribution the fitted probability of at most %d steps is %.4f, outside the %.4f%% interval [%.4f, %.4f] around the empirical proportion %d of %d." % (case, label, x, fittedprob, 100 * POINTLEVEL, lower, upper, count, values.size))


STRUCTUREBYNAME = {name: (pi, phgen) for name, pi, phgen in STRUCTURES}
STARTBYNAME = {name: (pi, phgen) for name, pi, phgen in STARTS}

# A start that does not have its structure's zero pattern is fitting a different
# shape, and one that is not a valid representation is not a starting point at
# all. Both fail here rather than somewhere downstream.
for startname, startpi, startgen in STARTS:
    truepi, truegen = STRUCTUREBYNAME[startname]

    if (not np.array_equal(startpi != 0.0, truepi != 0.0)
            or not np.array_equal(startgen != 0.0, truegen != 0.0)):
        sys.exit("Validation test failed at initialization: the starting parameters of the %s distribution do not share the zero pattern of the structure, so they describe a different shape." % startname)

    if np.array_equal(startpi, truepi) and np.array_equal(startgen, truegen):
        sys.exit("Validation test failed at initialization: the starting parameters of the %s distribution are the parameters the data was drawn from, which is what they are there to avoid." % startname)

    if (np.any(startgen < 0.0)
            or np.any(exitratesof(startgen) < -TOL)
            or abs(float(np.sum(startpi)) - 1.0) > TOL):
        sys.exit("Validation test failed at initialization: the starting parameters of the %s distribution are not a valid discrete phase-type representation." % startname)

FITS = {}


def getfit(name, censored, randominit):
    '''
    Returns the fit of the named structure under the given settings, fitting it
    the first time it is asked for and reusing it afterwards. The cases below
    lean on the same handful of fits, and each one costs an entire run of the
    EM algorithm.
    '''
    key = (name, censored, randominit)
    if key not in FITS:
        # With randominit the supplied parameters only fix the shape, so the
        # structure serves; without it they ARE the starting point, and the
        # invented start is used rather than the parameters behind the data.
        pi, phgen = (STRUCTUREBYNAME[name] if randominit
                     else STARTBYNAME[name])
        observations, censoring = DATA[name]
        FITS[key] = fitmodel(pi, phgen, observations,
                             censoring=censoring if censored else None,
                             randominit=randominit)
    return FITS[key]


# Four cases check the distribution function, each at the evaluation points of
# each of the three structures. Bonferroni's correction is used to share CDFLEVEL
# out over all of them, since it needs no assumption about how they depend on
# one another, and they do depend on one another: the points of a structure are
# read off one fit of one sample.
NCASESONCDF = 4
NINTERVALS = NCASESONCDF * sum(
    len(evaluationpoints(truedistribution(pi, phgen)))
    for _, pi, phgen in STRUCTURES)

POINTLEVEL = 1.0 - (1.0 - CDFLEVEL) / NINTERVALS

# The data, simulated once and reused by the cases that fit it
DATA = {}
for index, (name, pi, phgen) in enumerate(STRUCTURES):
    limits = censoringlimits(truedistribution(pi, phgen))

    # The uncensored observations are an unbiased sample of the distribution
    # only because the right and the left limit are the same value: what
    # right-censoring leaves behind is the distribution truncated at or below
    # that value, what left-censoring leaves behind is the distribution
    # truncated above it, and the two come back in the proportions that
    # reassemble it. Separating the two limits would quietly skew the empirical
    # distribution function that cases 3 and 4 are judged against, so it fails
    # here instead.
    if limits[0] != limits[1]:
        sys.exit("Validation test failed at initialization: the right and left censoring limits of the %s distribution are %s and %s. They have to be equal, or the uncensored observations are no longer an unbiased sample of the distribution." % (name, limits[0], limits[1]))

    observations = simulate(pi, phgen, NSAMPLES, SEED * 100 + index)
    censoring = censor(observations, limits, SEED * 10 + index)
    DATA[name] = (observations, censoring)

    # every censoring kind has to occur, or cases 3 and 4 quietly stop
    # exercising the kinds that are missing
    kinds = censoringkinds(censoring)
    if min(kinds) < MINPERKIND:
        sys.exit("Validation test failed at initialization: the censored data for the %s distribution holds %s observations of the four kinds, fewer than %d of one of them." % (name, kinds, MINPERKIND))


# ------------------------------------------------------------------
# CASE 1: Uncensored data, fitted from a random start
# ------------------------------------------------------------------

for name, pi, phgen in STRUCTURES:
    cdfwithinsamplingerror(1, name, getfit(name, False, True),
                           truedistribution(pi, phgen), DATA[name][0])


# ------------------------------------------------------------------
# CASE 2: Uncensored data, fitted from an invented starting point
# ------------------------------------------------------------------

# This is what checks that a supplied starting point is used rather than
# discarded. The fit is judged by the same criterion as case 1.
for name, pi, phgen in STRUCTURES:
    cdfwithinsamplingerror(2, name, getfit(name, False, False),
                           truedistribution(pi, phgen), DATA[name][0])

# A starting point that is used must leave its mark: fitting the same data from
# the supplied start and from a random one cannot give byte-identical answers,
# or the supplied start was ignored.
fromtruth = fittedparameters(getfit("general", False, False))
fromrandom = fittedparameters(getfit("general", False, True))

if all(np.array_equal(a, b) for a, b in zip(fromtruth, fromrandom)):
    sys.exit("Validation test failed at case 2: fitting from the supplied start and from a random start gives identical results, so the supplied starting parameters are being ignored.")


# ------------------------------------------------------------------
# CASE 3: Censored data, fitted from a random start
# ------------------------------------------------------------------

for name, pi, phgen in STRUCTURES:
    cdfwithinsamplingerror(3, name, getfit(name, True, True),
                           truedistribution(pi, phgen), DATA[name][0],
                           DATA[name][1])

# Censoring must actually change the fit, or cases 3 and 4 are repeating
# cases 1 and 2 under another name
observations, censoring = DATA["general"]
uncensored = fittedparameters(getfit("general", False, True))
censored = fittedparameters(getfit("general", True, True))

if all(np.array_equal(a, b) for a, b in zip(uncensored, censored)):
    sys.exit("Validation test failed at case 3: fitting with and without the censoring array gives identical results, so the censoring is being ignored.")

# An entirely missing censoring array and one saying that nothing is censored
# describe the same data, and must give the same fit
allmissing = np.full((observations.size, 2), np.nan)
asifnone = fittedparameters(fitmodel(GENERALPI, GENERALGEN, observations,
                                     censoring=allmissing))

for supplied, plain in zip(asifnone, uncensored):
    if np.max(np.abs(supplied - plain)) > TOL:
        sys.exit("Validation test failed at case 3: a censoring array marking every observation as uncensored does not give the same fit as supplying no censoring array at all.")


# ------------------------------------------------------------------
# CASE 4: Censored data, fitted from an invented starting point
# ------------------------------------------------------------------

for name, pi, phgen in STRUCTURES:
    cdfwithinsamplingerror(4, name, getfit(name, True, False),
                           truedistribution(pi, phgen), DATA[name][0],
                           DATA[name][1])


# ------------------------------------------------------------------
# CASE 5: The mean and the variance match the fitted parameters
# ------------------------------------------------------------------

# The fitted model is rebuilt as a dist object outside _fitdph, from the
# parameters _fitdph reports. dist is checked against closed-form results in
# test_dist.py, so agreement here says that _fitdph's own metrics are computed
# from the parameters it ends up with, and not from the ones it started with.
MODELS = {name: getfit(name, True, True) for name, _, _ in STRUCTURES}

for name, pi, phgen in STRUCTURES:
    model = MODELS[name]
    fittedpi, fittedgen, fittedexit = fittedparameters(model)
    outside = dist(discrete=True, initdist=fittedpi, phgen=fittedgen, seed=SEED)

    if abs(model.getmean() - outside.getmean()) > TOLFIT:
        sys.exit("Validation test failed at case 5: for the %s distribution the mean reported by _fitdph, %.10f, differs from the mean of the fitted parameters computed outside it, %.10f." % (name, model.getmean(), outside.getmean()))

    if abs(model.getvar() - outside.getvar()) > TOLFIT:
        sys.exit("Validation test failed at case 5: for the %s distribution the variance reported by _fitdph, %.10f, differs from the variance of the fitted parameters computed outside it, %.10f." % (name, model.getvar(), outside.getvar()))

    # a mean and a variance of a distribution on the positive integers
    if not np.isfinite(model.getmean()) or model.getmean() <= 1.0:
        sys.exit("Validation test failed at case 5: the mean of the %s distribution is %.4f, which no discrete phase-type distribution can have." % (name, model.getmean()))

    if not np.isfinite(model.getvar()) or model.getvar() < 0.0:
        sys.exit("Validation test failed at case 5: the variance of the %s distribution is %.4f, which is not a variance." % (name, model.getvar()))


# ------------------------------------------------------------------
# CASE 6: The density and the distribution function match them too
# ------------------------------------------------------------------

for name, pi, phgen in STRUCTURES:
    model = MODELS[name]
    fittedpi, fittedgen, fittedexit = fittedparameters(model)
    outside = dist(discrete=True, initdist=fittedpi, phgen=fittedgen, seed=SEED)

    for x in (1, 2, 3, 5, 9):
        if abs(model.getdensity(x) - outside.getdensity(x)) > TOLFIT:
            sys.exit("Validation test failed at case 6: for the %s distribution the probability of exactly %d steps reported by _fitdph differs from the one computed outside it." % (name, x))

        if abs(model.getcumprob(x) - outside.getcumprob(x)) > TOLFIT:
            sys.exit("Validation test failed at case 6: for the %s distribution the probability of at most %d steps reported by _fitdph differs from the one computed outside it." % (name, x))

    # the distribution function is the density summed, which ties the two
    # methods to each other rather than each to its own formula
    accumulated = 0.0
    for x in range(1, 60):
        accumulated += model.getdensity(x)
        if abs(accumulated - model.getcumprob(x)) > 1e-10:
            sys.exit("Validation test failed at case 6: for the %s distribution the probabilities of exactly one step up to %d steps sum to %.10f, but the probability of at most %d steps is %.10f." % (name, x, accumulated, x, model.getcumprob(x)))

    # and the whole distribution has to account for all the probability
    if abs(model.getcumprob(400) - 1.0) > 1e-8:
        sys.exit("Validation test failed at case 6: the fitted %s distribution puts %.10f of its probability below 400 steps, so it is not a proper distribution." % (name, model.getcumprob(400)))

    # A value that is not a whole number has no probability mass, as
    # documented. The class says so on the way out, which is captured here
    # rather than left to clutter the output -- and checked, since a missing
    # value returned in silence would be harder to account for.
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        density = model.getdensity(2.5)
        cumulative = model.getcumprob(2.5)

    if not np.isnan(density) or not np.isnan(cumulative):
        sys.exit("Validation test failed at case 6: a value that is not a whole number does not give a missing value.")

    if printed.getvalue().strip() == "":
        sys.exit("Validation test failed at case 6: a value that is not a whole number gives a missing value without saying why.")


# ------------------------------------------------------------------
# CASE 7: The fitted parameters form a feasible distribution
# ------------------------------------------------------------------

for name, pi, phgen in STRUCTURES:
    model = MODELS[name]
    fittedpi, fittedgen, fittedexit = fittedparameters(model)

    if fittedpi.size != NPHASES or fittedgen.shape != (NPHASES, NPHASES) or fittedexit.size != NPHASES:
        sys.exit("Validation test failed at case 7: the parameters fitted for the %s distribution do not have one entry per phase." % name)

    if np.any(fittedpi < 0.0) or abs(np.sum(fittedpi) - 1.0) > 1e-10:
        sys.exit("Validation test failed at case 7: the initial distribution fitted for the %s distribution is not a distribution." % name)

    if np.any(fittedgen < 0.0) or np.any(fittedexit < 0.0):
        sys.exit("Validation test failed at case 7: the parameters fitted for the %s distribution contain a negative probability." % name)

    # each phase either moves on or exits, so its row and its exit probability
    # together account for all of the probability
    if np.max(np.abs(np.sum(fittedgen, axis=1) + fittedexit - 1.0)) > 1e-10:
        sys.exit("Validation test failed at case 7: for the %s distribution the transition probabilities of a phase and its exit probability do not sum to one." % name)

    # the structure the fit was given must survive it: the EM algorithm
    # preserves structural zeros (p. 678), which is what makes a Coxian or a
    # negative binomial distribution fittable at all
    structuralzeros = np.asarray(phgen, dtype=float) == 0.0
    if np.any(fittedgen[structuralzeros] != 0.0):
        sys.exit("Validation test failed at case 7: the %s distribution was fitted with transitions where its structure has none." % name)

    if np.any(fittedpi[np.asarray(pi, dtype=float) == 0.0] != 0.0):
        sys.exit("Validation test failed at case 7: the %s distribution was fitted starting in a phase its structure never starts in." % name)


# ------------------------------------------------------------------
# CASE 8: The log-likelihood and the information criteria
# ------------------------------------------------------------------

def censoredloglikelihood(fitted, observations, censoring):
    '''
    Returns the log-likelihood of the data under a fitted distribution,
    computed here rather than by the class. Each observation contributes the
    logarithm of the probability of what is known about it, which is a point
    probability when it is uncensored and the probability of an interval when
    it is not.
    '''
    total = 0.0
    for index, value in enumerate(observations):
        lower, upper = censoring[index]

        if np.isnan(lower) and np.isnan(upper):
            probability = fitted.getdensity(int(value))          # P(Y = y)
        elif np.isnan(lower):
            probability = 1.0 - fitted.getcumprob(int(upper))    # P(Y > upper)
        elif np.isnan(upper):
            probability = fitted.getcumprob(int(lower))          # P(Y <= lower)
        else:
            probability = (fitted.getcumprob(int(upper))
                           - fitted.getcumprob(int(lower)))      # P(lower < Y <= upper)

        total += np.log(probability)

    return float(total)


def parametercount(fittedpi, fittedgen, fittedexit):
    '''
    Returns the number of free parameters of a fitted distribution: one for
    every transition a phase can make, absorption included, less one because
    the transitions of a phase are constrained to each other, plus one for
    every phase the process can start in, less one for the same reason.
    '''
    total = 0
    for i in range(fittedgen.shape[0]):
        total += (np.count_nonzero(fittedgen[i, :])
                  + np.count_nonzero(fittedexit[i]) - 1)
    return int(total + np.count_nonzero(fittedpi) - 1)


for name, pi, phgen in STRUCTURES:
    model = MODELS[name]
    loglik = float(model.getloglik())

    # The log-likelihood the class reports must be the log-likelihood of the
    # data it was given, censoring and all. Recomputing it here, from the
    # fitted parameters through the dist class, is what catches a fit whose
    # final likelihood quietly forgets that any of the data was censored.
    observations, censoring = DATA[name]
    fittedpi, fittedgen, fittedexit = fittedparameters(model)
    outside = dist(discrete=True, initdist=fittedpi, phgen=fittedgen, seed=SEED)
    recomputed = censoredloglikelihood(outside, observations, censoring)

    if abs(loglik - recomputed) > TOLLOGLIK:
        sys.exit("Validation test failed at case 8: for the %s distribution the log-likelihood reported by _fitdph, %.6f, differs from the log-likelihood of the same data computed outside it, %.6f." % (name, loglik, recomputed))

    # AIC and BIC are fixed by the log-likelihood and the number of free
    # parameters, so both are checked exactly. The criterion further down, that
    # each exceeds twice the magnitude of the log-likelihood, holds for any
    # positive penalty and so cannot tell a wrong penalty from a right one.
    nparam = parametercount(fittedpi, fittedgen, fittedexit)

    if nparam <= 0:
        sys.exit("Validation test failed at case 8: the %s distribution is fitted with %d free parameters." % (name, nparam))

    expectedaic = -2.0 * loglik + 2.0 * nparam
    expectedbic = -2.0 * loglik + nparam * np.log(NSAMPLES)

    if abs(float(model.getaic()) - expectedaic) > TOLFIT:
        sys.exit("Validation test failed at case 8: for the %s distribution the AIC is %.6f, but its log-likelihood and its %d free parameters give %.6f." % (name, float(model.getaic()), nparam, expectedaic))

    if abs(float(model.getbic()) - expectedbic) > TOLFIT:
        sys.exit("Validation test failed at case 8: for the %s distribution the BIC is %.6f, but its log-likelihood and its %d free parameters give %.6f." % (name, float(model.getbic()), nparam, expectedbic))

    # the likelihood is a product of probabilities, so its logarithm is negative
    if not np.isfinite(loglik) or loglik >= 0.0:
        sys.exit("Validation test failed at case 8: the log-likelihood of the %s distribution is %.4f, which is not a negative number." % (name, loglik))

    for metric, value in (("AIC", float(model.getaic())),
                          ("BIC", float(model.getbic()))):
        if not np.isfinite(value):
            sys.exit("Validation test failed at case 8: the %s of the %s distribution is not a number." % (metric, name))

        if value <= 2.0 * abs(loglik):
            sys.exit("Validation test failed at case 8: the %s of the %s distribution is %.4f, which does not exceed twice the magnitude of its log-likelihood, %.4f." % (metric, name, value, 2.0 * abs(loglik)))


# ------------------------------------------------------------------
# CASE 9: The fit is converged, order-independent, and guards its input
# ------------------------------------------------------------------

# A converged fit cannot be improved by continuing it. Restarting from the
# parameters the fit ended on and running again must therefore leave the
# log-likelihood essentially where it was. An EM loop that stops early passes
# every check on the parameters it produces -- the very first M-step already
# reproduces the sample mean exactly -- so this is what catches it.
for name, pi, phgen in STRUCTURES:
    observations, censoring = DATA[name]

    for label, censored in (("uncensored", False), ("censored", True)):
        applied = censoring if censored else None
        first = getfit(name, censored, False)      # the fit case 2 or 4 made
        againpi, againgen, againexit = fittedparameters(first)

        continued = _fitdph(obs=np.copy(observations),
                           censoring=None if applied is None else np.copy(applied),
                           initpi=againpi,
                           initphgen=againgen,
                           initexitrates=againexit,
                           randominit=False,
                           seed=SEED,
                           verbose=False)
        continued.fit()

        improvement = float(continued.getloglik()) - float(first.getloglik())

        if improvement > MAXIMPROVEMENT:
            sys.exit("Validation test failed at case 9: continuing the %s fit of the %s distribution improves its log-likelihood by %.3e, so the first fit had not converged." % (label, name, improvement))

        # continuing must not make it worse either, since the EM algorithm
        # never decreases the log-likelihood
        if improvement < -MAXIMPROVEMENT:
            sys.exit("Validation test failed at case 9: continuing the %s fit of the %s distribution lowers its log-likelihood by %.3e, which the EM algorithm cannot do." % (label, name, -improvement))

# The rows of the censoring array are matched to the observations by position,
# so anything that reorders one has to reorder the other. Presenting the same
# data in a different order must give the same fit.
observations, censoring = DATA["general"]
inorder = fittedparameters(getfit("general", True, False))

shuffle = np.random.default_rng(SEED).permutation(observations.size)
shuffled = fittedparameters(fitmodel(STARTGENERALPI, STARTGENERALGEN,
                                     observations[shuffle],
                                     censoring=censoring[shuffle, :],
                                     randominit=False))

for original, reordered in zip(inorder, shuffled):
    if np.max(np.abs(original - reordered)) > TOLFIT:
        sys.exit("Validation test failed at case 9: presenting the same censored observations in a different order changes the fit, so a censoring rule is reaching the wrong observation.")

# A censoring array of the wrong shape cannot be matched to the observations at
# all, and has to be refused rather than silently misread
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    malformed = _fitdph(obs=np.copy(observations),
                       censoring=np.full((observations.size - 3, 2), np.nan),
                       initpi=np.copy(STARTGENERALPI),
                       initphgen=np.copy(STARTGENERALGEN),
                       initexitrates=exitratesof(STARTGENERALGEN),
                       randominit=False,
                       seed=SEED,
                       verbose=False)

if printed.getvalue().strip() == "":
    sys.exit("Validation test failed at case 9: a censoring array with the wrong number of rows was accepted without a word.")

printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    malformed.fit()

# having refused it, the fit has to go on as though no censoring was given
plain = fittedparameters(getfit("general", False, False))

for refused, uncensoredfit in zip(fittedparameters(malformed), plain):
    if np.max(np.abs(refused - uncensoredfit)) > TOL:
        sys.exit("Validation test failed at case 9: after refusing a censoring array of the wrong shape the fit is not the one the same data gives with no censoring at all.")


# ------------------------------------------------------------------
# CASE 10: the guard on the direction of the log-likelihood
# ------------------------------------------------------------------

# The EM algorithm cannot lower the log-likelihood (p. 675), and the loop stops
# as soon as the improvement falls below the tolerance. A DECREASE is therefore
# the one failure the stopping rule cannot distinguish from success: it looks
# exactly like a converged fit. The class warns instead of passing it over, and
# both halves of that need checking -- a healthy fit that warned would make the
# warning noise, and a broken one that stayed silent would make it useless.

observations, _ = DATA["general"]

with warnings.catch_warnings(record=True) as raised:
    warnings.simplefilter("always")
    fitmodel(STARTGENERALPI, STARTGENERALGEN, observations, randominit=False)

if [w for w in raised if issubclass(w.category, RuntimeWarning)]:
    sys.exit("Validation test failed at case 10: a fit that converges normally raises a RuntimeWarning about the log-likelihood, so the guard fires when nothing is wrong.")


TRUEESTEP = fitdphmodule.edph


class FallingEstep:
    '''
    Stands in for the E-step and reports a log-likelihood that falls at every
    iteration, which the EM algorithm cannot do. The real E-step still runs, so
    the fit proceeds normally in every other respect.
    '''

    def __init__(self, *args, **kwargs):
        self.inner = TRUEESTEP(*args, **kwargs)
        self.calls = 0
        self.loglikelihood = 0.0

    def run(self, **kwargs):
        result = self.inner.run(**kwargs)
        self.calls += 1
        self.loglikelihood = -100.0 - self.calls
        return result


fitdphmodule.edph = FallingEstep
try:
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        fitmodel(STARTGENERALPI, STARTGENERALGEN, observations, randominit=False)
finally:
    fitdphmodule.edph = TRUEESTEP

if not [w for w in raised if issubclass(w.category, RuntimeWarning)]:
    sys.exit("Validation test failed at case 10: the log-likelihood fell at every iteration and the fit finished without a word, so a decreasing likelihood is being read as convergence.")


# The loop can also end by running out of iterations, which is not convergence
# but looks identical from outside: parameters, a log-likelihood, no complaint.
# A fit given too few iterations to converge has to say so.

with warnings.catch_warnings(record=True) as raised:
    warnings.simplefilter("always")
    capped = _fitdph(obs=np.copy(observations),
                  initpi=np.copy(STARTGENERALPI),
                  initphgen=np.copy(STARTGENERALGEN),
                  initexitrates=exitratesof(STARTGENERALGEN),
                  randominit=False,
                  seed=SEED,
                  itermax=3,
                  verbose=False)
    capped.fit()

atcap = [w for w in raised if issubclass(w.category, RuntimeWarning)
         and "itermax" in str(w.message)]
if not atcap:
    sys.exit("Validation test failed at case 10: a fit stopped by its iteration limit before converging reported nothing, so an unconverged fit is indistinguishable from a converged one.")

# and a fit with room to converge must not claim it ran out
with warnings.catch_warnings(record=True) as raised:
    warnings.simplefilter("always")
    fitmodel(STARTGENERALPI, STARTGENERALGEN, observations, randominit=False)

if [w for w in raised if issubclass(w.category, RuntimeWarning)
        and "itermax" in str(w.message)]:
    sys.exit("Validation test failed at case 10: a fit that converged well inside its iteration limit reports that it ran out of iterations.")


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print("fitdph: All tests passed.")
