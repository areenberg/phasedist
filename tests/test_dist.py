'''
UNIT TEST FOR THE dist CLASS (a phase-type distribution object).

Every public method of the class is covered by one of the cases below, and the
list of covered methods is checked against the class itself, so that a method
added later cannot go untested unnoticed.

The methods returning distributional quantities are tested against closed-form
results. Both test distributions use a fully general generator whose rows all
sum to the same value, which makes every exit rate equal and collapses the
phase-type distribution to a memoryless one (Theorem 3.1.22, p. 138): the
continuous one is exponential with rate RATE, and the discrete one is geometric
with success probability PROB. The mean, variance, density, distribution
function and quantile function can then be compared with the textbook formulas
for those two distributions rather than with the class's own machinery.

The four methods describing the phases rather than the absorption time
(getphasetime, getphasetimematrix, getexitprob and getexitprobmatrix) are
tested on structures where the answer can be written down by hand: phases in
series, where the process must walk through them in order and can only exit
from the last one, and phases in parallel, where it exits from whichever phase
it started in. The general generators give a third check, since equal exit
rates make the remaining time memoryless and therefore the same from every
phase. Those structures are also simulated directly, which is the only check
here that shares no algebra with the class; the theoretical values are required
to lie inside 99% confidence intervals for the simulated ones, from Student's t
distribution for the phase times and the exact Clopper-Pearson interval for the
exit probabilities. The 99% is the confidence that every one of those intervals
covers its value at once, so each single interval is taken at the corrected level
SIMLEVEL, and the rate at which this case fails on correct code is known rather
than guessed.

getrandom is tested with Pearson chi-square goodness-of-fit tests at the 1%
level, on three structures per distribution kind and on both sampling methods
the class offers. The expected bin probabilities are computed here from the
matrix representation directly, so the test does not lean on getcumprob to
judge getrandom. Note that a test at the 1% level rejects a correct sampler 1%
of the time by construction, so the seed is fixed; SEED was chosen so that
every test passes with a comfortable margin (the largest statistic is below
half its critical value).

Sub-tests:
    Case 1: getinitdist, getphasegen and getexitrates return the parameters
            that went in.
    Case 2: getmean and getvar match the exponential and geometric formulas.
    Case 3: getdensity and getcumprob match those formulas.
    Case 4: getquantile matches those formulas, including its edge cases.
    Case 5: getphasetime, getphasetimematrix, getexitprob and
            getexitprobmatrix match values worked out by hand, agree with each
            other and with getmean, and fall inside confidence intervals for a
            direct simulation of the underlying process.
    Case 6: getrandom passes chi-square goodness-of-fit tests, and its size
            argument behaves as documented.
    Case 7: countParameters reproduces counts worked out by hand.
    Case 8: plot writes a file for both plot types and both kinds of
            distribution.

References:
    Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in
    Applied Probability. Springer. https://doi.org/10.1007/978-1-4939-7049-0
'''

import contextlib
import io
import os
import sys
import numpy as np

# A non-interactive backend, so that the plots of case 8 can be written without
# a display. This has to be selected before dist is imported, since importing
# it loads pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.linalg import expm
from scipy.stats import beta, chi2, t

# Load phasedist from the src-folder so the test can be run without installing
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from phasedist.dist import dist


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

TOL = 1e-12       # tolerance for comparisons that are exact in theory
TOLQUANTILE = 1e-6  # the quantile function is solved numerically and rounded
SEED = 9

RATE = 0.5  # exit rate of the continuous distribution, i.e. exponential(RATE)
PROB = 0.3  # exit probability of the discrete distribution, i.e. geometric(PROB)

ALPHA = 0.01            # significance level of the statistical tests
NSAMPLES = 10000        # samples drawn for a chi-square test
NSAMPLESQUANTILE = 4000  # samples for the slower quantile-based sampler
NBINS = 10              # bins used for a continuous chi-square test
MINBINPROB = 0.05       # smallest probability a discrete bin may carry
MINEXPECTED = 5.0       # smallest expected count a chi-square bin may have

NSIMPHASES = 20000      # runs simulated for the phase statistics of case 5

# Every public method of dist. Checked against the class below, so that a new
# public method cannot slip in without a case covering it.
COVEREDMETHODS = ("countParameters", "getcumprob", "getdensity", "getexitprob",
                  "getexitprobmatrix", "getexitrates", "getinitdist", "getmean",
                  "getphasegen", "getphasetime", "getphasetimematrix",
                  "getquantile", "getrandom", "getvar", "plot")

# The two distributions used for cases 1 to 5 and 7. Both generators are fully
# general, and their rows all sum to the same value, so every exit rate is
# equal.
INITDIST = np.array([0.5, 0.3, 0.2])

CPHGEN = np.array([[-1.2, 0.4, 0.3],
                   [0.2, -0.9, 0.2],
                   [0.5, 0.3, -1.3]])      # rows sum to -RATE

DPHGEN = np.array([[0.40, 0.20, 0.10],
                   [0.10, 0.50, 0.10],
                   [0.20, 0.20, 0.30]])    # rows sum to 1-PROB

# Structures used for the chi-square tests of case 6, in the order their seeds
# are derived from SEED
CPHSTRUCTURES = (
    ("general", INITDIST, CPHGEN),
    ("generalized Erlang", np.array([1.0, 0.0, 0.0]),
     np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, -1.0]])),
    ("hyperexponential", INITDIST,
     np.array([[-1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, -2.0]])),
)

DPHSTRUCTURES = (
    ("general", INITDIST, DPHGEN),
    ("negative binomial", np.array([1.0, 0.0, 0.0]),
     np.array([[0.55, 0.45, 0.0], [0.0, 0.55, 0.45], [0.0, 0.0, 0.55]])),
    ("hyper-geometric", INITDIST,
     np.array([[0.4, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.8]])),
)

# Structures used for case 5, together with the phase times worked out by hand.
# With the phases in series the process walks through them in order and can only
# exit from the last one, so the expected time in phase j starting from phase i
# is the mean holding time of phase j when j is at or after i and zero
# otherwise. With the phases in parallel the process exits from the phase it
# started in, so the matrix is diagonal. The holding times are all different, so
# that a mix-up of rows and columns cannot pass unnoticed.
CPHSERIESGEN = np.array([[-1.0, 1.0, 0.0],
                         [0.0, -2.0, 2.0],
                         [0.0, 0.0, -4.0]])

CPHSERIESTIME = np.array([[1.0, 1.0 / 2.0, 1.0 / 4.0],
                          [0.0, 1.0 / 2.0, 1.0 / 4.0],
                          [0.0, 0.0, 1.0 / 4.0]])

DPHSERIESGEN = np.array([[0.50, 0.50, 0.00],
                         [0.00, 0.75, 0.25],
                         [0.00, 0.00, 0.80]])

DPHSERIESTIME = np.array([[1.0 / 0.50, 1.0 / 0.25, 1.0 / 0.20],
                          [0.0, 1.0 / 0.25, 1.0 / 0.20],
                          [0.0, 0.0, 1.0 / 0.20]])

CPHPARALLELGEN = np.array([[-1.0, 0.0, 0.0],
                           [0.0, -0.5, 0.0],
                           [0.0, 0.0, -2.0]])

CPHPARALLELTIME = np.diag([1.0 / 1.0, 1.0 / 0.5, 1.0 / 2.0])

DPHPARALLELGEN = np.array([[0.4, 0.0, 0.0],
                           [0.0, 0.6, 0.0],
                           [0.0, 0.0, 0.8]])

DPHPARALLELTIME = np.diag([1.0 / 0.6, 1.0 / 0.4, 1.0 / 0.2])

# In series the process always exits from the last phase, whichever phase it
# starts in; in parallel it always exits from the phase it started in.
SERIESEXIT = np.array([[0.0, 0.0, 1.0],
                       [0.0, 0.0, 1.0],
                       [0.0, 0.0, 1.0]])

PARALLELEXIT = np.eye(3)

# Verify the two main generators really do collapse to a memoryless
# distribution, otherwise the closed-form comparisons below are meaningless
if np.any(np.abs(np.sum(CPHGEN, axis=1) + RATE) > TOL):
    sys.exit("Validation test failed at initialization: the rows of the continuous generator do not all sum to minus RATE.")

if np.any(np.abs(np.sum(DPHGEN, axis=1) - (1.0 - PROB)) > TOL):
    sys.exit("Validation test failed at initialization: the rows of the discrete sub-transition matrix do not all sum to one minus PROB.")

if abs(np.sum(INITDIST) - 1.0) > TOL:
    sys.exit("Validation test failed at initialization: the initial distribution does not sum to one.")

# Every public method must be covered by a case
publicmethods = tuple(sorted(name for name in dir(dist)
                             if not name.startswith("_")
                             and callable(getattr(dist, name))))

if publicmethods != tuple(sorted(COVEREDMETHODS)):
    sys.exit("Validation test failed at initialization: the public methods of dist are %s, but the test covers %s." % (publicmethods, tuple(sorted(COVEREDMETHODS))))


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def makedist(discrete, initdist, phgen, seed=SEED):
    '''
    Returns a dist object built from copies of the given parameters.
    '''
    return dist(discrete=discrete,
                initdist=np.copy(initdist),
                phgen=np.copy(phgen),
                seed=seed)


def cphcumprob(x, initdist, phgen):
    '''
    The continuous distribution function, computed from the matrix
    representation rather than through the class being tested.
    '''
    return 1.0 - float(np.sum(np.matmul(initdist, expm(np.asarray(phgen) * x))))


def dphcumprob(k, initdist, phgen):
    '''
    The discrete distribution function, computed from the matrix representation
    rather than through the class being tested.
    '''
    return 1.0 - float(np.sum(np.matmul(
        initdist, np.linalg.matrix_power(np.asarray(phgen), int(k)))))


def cphbinedges(initdist, phgen, nbins):
    '''
    Returns bin edges splitting the continuous distribution into nbins parts of
    equal probability, found by bisecting its distribution function. The last
    edge is infinite.
    '''
    edges = [0.0]
    for i in range(1, nbins):
        target = i / nbins
        low, high = 0.0, 1.0
        while cphcumprob(high, initdist, phgen) < target:
            high *= 2.0
        for _ in range(200):
            middle = 0.5 * (low + high)
            if cphcumprob(middle, initdist, phgen) < target:
                low = middle
            else:
                high = middle
        edges.append(0.5 * (low + high))
    edges.append(np.inf)
    return np.array(edges)


def dphbins(initdist, phgen, minprob=MINBINPROB, kmax=10000):
    '''
    Groups the positive integers into bins each carrying at least minprob, with
    the final bin holding the whole remaining tail. Returns a list of
    (lower, upper, probability) triples where the final upper bound is
    infinite.
    '''
    bins = []
    current = []
    accumulated = 0.0
    previous = 0.0
    k = 1

    while k <= kmax:
        cumulative = dphcumprob(k, initdist, phgen)
        current.append(k)
        accumulated += cumulative - previous
        previous = cumulative

        if accumulated >= minprob and 1.0 - cumulative >= minprob:
            bins.append((current[0], current[-1], accumulated))
            current = []
            accumulated = 0.0
        if 1.0 - cumulative < minprob:
            break
        k += 1

    # whatever is left, together with the tail beyond it, forms the last bin
    lower = current[0] if current else k + 1
    bins.append((lower, np.inf, accumulated + (1.0 - previous)))
    return bins


def chisquarestatistic(observed, expected):
    '''
    Returns Pearson's chi-square statistic.
    '''
    return float(np.sum((observed - expected) ** 2 / expected))


def cphchisquare(initdist, phgen, seed, method, nsamples):
    '''
    Draws samples from a continuous distribution and returns the chi-square
    statistic against equiprobable bins, its degrees of freedom, and the
    smallest expected count.
    '''
    edges = cphbinedges(initdist, phgen, NBINS)

    probabilities = np.array([
        (cphcumprob(edges[i + 1], initdist, phgen)
         if np.isfinite(edges[i + 1]) else 1.0)
        - cphcumprob(edges[i], initdist, phgen)
        for i in range(NBINS)
    ])

    samples = np.asarray(
        makedist(False, initdist, phgen, seed).getrandom(size=nsamples,
                                                        method=method),
        dtype=float)

    observed, _ = np.histogram(samples, bins=np.append(edges[:-1], np.inf))
    expected = nsamples * probabilities

    return (chisquarestatistic(observed, expected), NBINS - 1,
            float(expected.min()))


def dphchisquare(initdist, phgen, seed, nsamples):
    '''
    Draws samples from a discrete distribution and returns the chi-square
    statistic against grouped integer bins, its degrees of freedom, and the
    smallest expected count.
    '''
    bins = dphbins(initdist, phgen)
    probabilities = np.array([b[2] for b in bins])

    samples = np.asarray(
        makedist(True, initdist, phgen, seed).getrandom(size=nsamples,
                                                       method="direct"),
        dtype=float)

    observed = np.zeros(len(bins))
    for j, (lower, upper, _) in enumerate(bins):
        if np.isfinite(upper):
            observed[j] = np.sum((samples >= lower) & (samples <= upper))
        else:
            observed[j] = np.sum(samples >= lower)

    expected = nsamples * probabilities

    return (chisquarestatistic(observed, expected), len(bins) - 1,
            float(expected.min()))


def simulatephases(discrete, initdist, phgen, nsim, seed):
    '''
    Simulates the underlying Markov process directly from the generator. Returns
    the time each run spends in each phase, as an array with one row per run and
    one column per phase, measured in steps when the process is discrete,
    together with the phase each run exits from. The process is built here from
    the generator and driven by a generator of random numbers of its own, so
    this shares neither algebra nor random numbers with the class being tested.

    The outcome of every single run is returned rather than an average, so that
    the confidence intervals below can be computed from the spread across runs.
    '''
    rng = np.random.default_rng(seed)
    phgen = np.asarray(phgen, dtype=float)
    nphases = phgen.shape[0]
    initdist = np.asarray(initdist, dtype=float).ravel()

    # the distribution of the next phase, absorption being the last column
    if discrete:
        rates = None
        step = np.hstack([phgen, (1.0 - np.sum(phgen, axis=1)).reshape(-1, 1)])
    else:
        rates = -np.diag(phgen)
        step = np.zeros((nphases, nphases + 1))
        for i in range(nphases):
            row = np.copy(phgen[i])
            row[i] = 0.0
            step[i, :nphases] = row / rates[i]
            step[i, nphases] = -np.sum(phgen[i]) / rates[i]

    initcumulative = np.cumsum(initdist)
    stepcumulative = np.cumsum(step, axis=1)

    time = np.zeros((nsim, nphases))
    exitphase = np.zeros(nsim, dtype=int)

    for run in range(nsim):
        s = min(int(np.searchsorted(initcumulative, rng.random())), nphases - 1)
        while True:
            time[run, s] += 1.0 if discrete else rng.exponential(1.0 / rates[s])
            nxt = min(int(np.searchsorted(stepcumulative[s], rng.random())),
                      nphases)
            if nxt == nphases:
                exitphase[run] = s
                break
            s = nxt

    return time, exitphase


def meaninterval(sample, level):
    '''
    Returns the two ends of a confidence interval for the mean of each column of
    sample, at the given confidence level, from Student's t distribution. The
    time a run spends in a phase is far from normally distributed, being skewed
    and with an atom at zero, but the interval is for its mean over many runs,
    which the central limit theorem makes very nearly normal.

    A column that never varies gives an interval of zero width.
    '''
    nsim = sample.shape[0]
    mean = np.mean(sample, axis=0)
    standarderror = np.std(sample, axis=0, ddof=1) / np.sqrt(nsim)
    halfwidth = t.ppf(0.5 * (1.0 + level), nsim - 1) * standarderror
    return mean - halfwidth, mean + halfwidth


def clopperpearsoninterval(counts, nsim, level):
    '''
    Returns the two ends of the Clopper-Pearson confidence interval for each of
    the given counts out of nsim runs, at the given confidence level. The
    interval is the exact one for a binomial count, obtained from the quantiles
    of the beta distribution, and is closed at zero when nothing was counted and
    at one when every run was counted. Those two cases are taken separately,
    since the beta distribution is not defined for the parameters they would
    otherwise ask for.
    '''
    alpha = 1.0 - level
    counts = np.asarray(counts)

    lower = np.zeros(counts.size)
    upper = np.ones(counts.size)

    positive = counts > 0
    lower[positive] = beta.ppf(0.5 * alpha, counts[positive],
                               nsim - counts[positive] + 1)

    incomplete = counts < nsim
    upper[incomplete] = beta.ppf(1.0 - 0.5 * alpha, counts[incomplete] + 1,
                                 nsim - counts[incomplete])

    return lower, upper


CPH = makedist(False, INITDIST, CPHGEN)
DPH = makedist(True, INITDIST, DPHGEN)


# ------------------------------------------------------------------
# CASE 1: The parameters that went in are the parameters that come out
# ------------------------------------------------------------------

# The class stores the initial distribution and the generator as numpy
# matrices, so the values are compared rather than the containers.
if not np.array_equal(np.asarray(CPH.getinitdist()).ravel(), INITDIST):
    sys.exit("Validation test failed at case 1: the continuous initial distribution returned differs from the one supplied.")

if not np.array_equal(np.asarray(DPH.getinitdist()).ravel(), INITDIST):
    sys.exit("Validation test failed at case 1: the discrete initial distribution returned differs from the one supplied.")

if not np.array_equal(np.asarray(CPH.getphasegen()), CPHGEN):
    sys.exit("Validation test failed at case 1: the continuous generator returned differs from the one supplied.")

if not np.array_equal(np.asarray(DPH.getphasegen()), DPHGEN):
    sys.exit("Validation test failed at case 1: the discrete sub-transition matrix returned differs from the one supplied.")

# The exit rates are not supplied to the class; they follow from the generator,
# as minus its row sums in the continuous case and as one minus its row sums in
# the discrete case. They are recomputed here rather than taken from the class.
if np.max(np.abs(np.asarray(CPH.getexitrates()).ravel()
                 + np.sum(CPHGEN, axis=1))) > TOL:
    sys.exit("Validation test failed at case 1: the continuous exit rates do not equal minus the row sums of the generator supplied.")

if np.max(np.abs(np.asarray(DPH.getexitrates()).ravel()
                 - (1.0 - np.sum(DPHGEN, axis=1)))) > TOL:
    sys.exit("Validation test failed at case 1: the discrete exit probabilities do not equal one minus the row sums of the matrix supplied.")

# with these generators every exit rate is equal, which is what makes cases 2
# to 4 comparable with the exponential and geometric formulas
if np.max(np.abs(np.asarray(CPH.getexitrates()).ravel() - RATE)) > TOL:
    sys.exit("Validation test failed at case 1: the continuous exit rates are not all equal to RATE.")

if np.max(np.abs(np.asarray(DPH.getexitrates()).ravel() - PROB)) > TOL:
    sys.exit("Validation test failed at case 1: the discrete exit probabilities are not all equal to PROB.")


# ------------------------------------------------------------------
# CASE 2: The mean and the variance match the closed-form results
# ------------------------------------------------------------------

if abs(CPH.getmean() - 1.0 / RATE) > TOL:
    sys.exit("Validation test failed at case 2: the mean does not equal the mean of the exponential distribution.")

if abs(CPH.getvar() - 1.0 / RATE ** 2) > TOL:
    sys.exit("Validation test failed at case 2: the variance does not equal the variance of the exponential distribution.")

if abs(DPH.getmean() - 1.0 / PROB) > TOL:
    sys.exit("Validation test failed at case 2: the mean does not equal the mean of the geometric distribution.")

if abs(DPH.getvar() - (1.0 - PROB) / PROB ** 2) > TOL:
    sys.exit("Validation test failed at case 2: the variance does not equal the variance of the geometric distribution.")


# ------------------------------------------------------------------
# CASE 3: The density and the distribution function match them too
# ------------------------------------------------------------------

CPHPOINTS = np.array([0.25, 0.5, 1.0, 3.0, 7.0])
DPHPOINTS = np.array([1, 2, 3, 5, 9])

for x in CPHPOINTS:
    if abs(CPH.getdensity(x) - RATE * np.exp(-RATE * x)) > TOL:
        sys.exit("Validation test failed at case 3: the density does not equal the density of the exponential distribution.")
    if abs(CPH.getcumprob(x) - (1.0 - np.exp(-RATE * x))) > TOL:
        sys.exit("Validation test failed at case 3: the distribution function does not equal that of the exponential distribution.")

for k in DPHPOINTS:
    if abs(DPH.getdensity(k) - PROB * (1.0 - PROB) ** (k - 1)) > TOL:
        sys.exit("Validation test failed at case 3: the probability mass does not equal that of the geometric distribution.")
    if abs(DPH.getcumprob(k) - (1.0 - (1.0 - PROB) ** k)) > TOL:
        sys.exit("Validation test failed at case 3: the distribution function does not equal that of the geometric distribution.")

# A whole number given as a floating point value has to be accepted as well,
# not just as an integer: getrandom returns floating point values even for a
# discrete distribution, so getdensity(getrandom()) is ordinary usage.
for k in DPHPOINTS:
    if abs(DPH.getdensity(float(k)) - DPH.getdensity(int(k))) > TOL:
        sys.exit("Validation test failed at case 3: the probability mass at a whole number differs depending on whether it is given as an integer or as a floating point value.")
    if abs(DPH.getcumprob(float(k)) - DPH.getcumprob(int(k))) > TOL:
        sys.exit("Validation test failed at case 3: the distribution function at a whole number differs depending on whether it is given as an integer or as a floating point value.")

# the same thing end to end: the mass at a value the class itself sampled
sampledmass = DPH.getdensity(DPH.getrandom(size=1))

if not np.isfinite(sampledmass) or sampledmass <= 0.0 or sampledmass > 1.0:
    sys.exit("Validation test failed at case 3: the probability mass at a value returned by getrandom is not a probability.")

# both methods also accept an array, which must agree with the scalar calls
if np.max(np.abs(CPH.getdensity(CPHPOINTS)
                 - np.array([CPH.getdensity(x) for x in CPHPOINTS]))) > TOL:
    sys.exit("Validation test failed at case 3: the density of an array of points differs from the densities of the single points.")

if np.max(np.abs(CPH.getcumprob(CPHPOINTS)
                 - np.array([CPH.getcumprob(x) for x in CPHPOINTS]))) > TOL:
    sys.exit("Validation test failed at case 3: the distribution function of an array of points differs from that of the single points.")

if np.max(np.abs(DPH.getdensity(DPHPOINTS)
                 - np.array([DPH.getdensity(k) for k in DPHPOINTS]))) > TOL:
    sys.exit("Validation test failed at case 3: the probability mass of an array of points differs from that of the single points.")

if np.max(np.abs(DPH.getcumprob(DPHPOINTS)
                 - np.array([DPH.getcumprob(k) for k in DPHPOINTS]))) > TOL:
    sys.exit("Validation test failed at case 3: the distribution function of an array of points differs from that of the single points.")


# ------------------------------------------------------------------
# CASE 4: The quantile function matches them as well
# ------------------------------------------------------------------

PROBABILITIES = np.array([0.1, 0.25, 0.5, 0.9, 0.99])

# the continuous quantile is found numerically, so it is compared with the
# looser tolerance
for p in PROBABILITIES:
    if abs(CPH.getquantile(p) - (-np.log(1.0 - p) / RATE)) > TOLQUANTILE:
        sys.exit("Validation test failed at case 4: the quantile does not equal the quantile of the exponential distribution.")

# the discrete quantile is the smallest number of steps whose probability
# reaches p, which for the geometric distribution is a ceiling
for p in PROBABILITIES:
    expected = int(np.ceil(np.log(1.0 - p) / np.log(1.0 - PROB)))
    if DPH.getquantile(p) != expected:
        sys.exit("Validation test failed at case 4: the quantile does not equal the quantile of the geometric distribution.")

if np.max(np.abs(CPH.getquantile(PROBABILITIES)
                 - np.array([CPH.getquantile(p) for p in PROBABILITIES]))) > TOL:
    sys.exit("Validation test failed at case 4: the quantile of an array of probabilities differs from the quantiles of the single probabilities.")

if np.max(np.abs(DPH.getquantile(PROBABILITIES)
                 - np.array([DPH.getquantile(p) for p in PROBABILITIES]))) > TOL:
    sys.exit("Validation test failed at case 4: the quantile of an array of probabilities differs from the quantiles of the single probabilities.")

# documented edge cases
for distribution in (CPH, DPH):
    if distribution.getquantile(0.0) != 0.0:
        sys.exit("Validation test failed at case 4: the quantile at probability zero is not zero.")
    if not np.isinf(distribution.getquantile(1.0)):
        sys.exit("Validation test failed at case 4: the quantile at probability one is not infinite.")
    if not np.isnan(distribution.getquantile(-0.1)):
        sys.exit("Validation test failed at case 4: a negative probability does not give a missing value.")
    if not np.isnan(distribution.getquantile(1.1)):
        sys.exit("Validation test failed at case 4: a probability above one does not give a missing value.")


# ------------------------------------------------------------------
# CASE 5: The phase times and the exit-phase probabilities are the ones
#         worked out by hand
# ------------------------------------------------------------------

# Each entry is a label, whether the distribution is discrete, its parameters,
# and the two matrices worked out by hand above.
PHASECASES = (
    ("continuous phases in series", False, np.array([1.0, 0.0, 0.0]),
     CPHSERIESGEN, CPHSERIESTIME, SERIESEXIT),
    ("discrete phases in series", True, np.array([1.0, 0.0, 0.0]),
     DPHSERIESGEN, DPHSERIESTIME, SERIESEXIT),
    ("continuous phases in parallel", False, INITDIST,
     CPHPARALLELGEN, CPHPARALLELTIME, PARALLELEXIT),
    ("discrete phases in parallel", True, INITDIST,
     DPHPARALLELGEN, DPHPARALLELTIME, PARALLELEXIT),
)

for label, discrete, initdist, phgen, timematrix, exitmatrix in PHASECASES:

    distribution = makedist(discrete, initdist, phgen)

    if np.max(np.abs(np.asarray(distribution.getphasetimematrix())
                     - timematrix)) > TOL:
        sys.exit("Validation test failed at case 5: the expected phase times of %s differ from the ones worked out by hand." % label)

    if np.max(np.abs(np.asarray(distribution.getexitprobmatrix())
                     - exitmatrix)) > TOL:
        sys.exit("Validation test failed at case 5: the exit-phase probabilities of %s differ from the ones worked out by hand." % label)

    # the vectors are the rows of those matrices weighted by the initial
    # distribution
    if np.max(np.abs(np.asarray(distribution.getphasetime()).ravel()
                     - np.matmul(initdist, timematrix))) > TOL:
        sys.exit("Validation test failed at case 5: the expected time per phase of %s differs from the one worked out by hand." % label)

    if np.max(np.abs(np.asarray(distribution.getexitprob()).ravel()
                     - np.matmul(initdist, exitmatrix))) > TOL:
        sys.exit("Validation test failed at case 5: the probability of exiting from each phase of %s differs from the one worked out by hand." % label)

if len(PHASECASES) != 4:
    sys.exit("Validation test failed at case 5: the hand-computed structures no longer cover both kinds of distribution in series and in parallel.")

# The general generators give a third hand-computable case. Their exit rates are
# all equal, so the time still to come is memoryless and does not depend on the
# phase the process is in: the expected time until it exits is then the mean of
# the distribution whichever phase it starts in, which is to say every row of
# the phase-time matrix sums to that mean. The probability of exiting from a
# phase is the time spent there times the common exit rate.
for label, distribution, rate in (("the continuous", CPH, RATE),
                                  ("the discrete", DPH, PROB)):

    timematrix = np.asarray(distribution.getphasetimematrix())

    if np.max(np.abs(np.sum(timematrix, axis=1) - distribution.getmean())) > TOL:
        sys.exit("Validation test failed at case 5: the rows of the phase-time matrix of %s general distribution do not all sum to its mean, although its exit rates are equal." % label)

    if np.max(np.abs(np.asarray(distribution.getexitprobmatrix())
                     - rate * timematrix)) > TOL:
        sys.exit("Validation test failed at case 5: the exit-phase probabilities of %s general distribution are not its phase times scaled by the common exit rate." % label)

# Properties every phase-type distribution must have, checked on every structure
# this file defines
PHASEINVARIANTS = (
    [(False, initdist, phgen) for _, initdist, phgen in CPHSTRUCTURES]
    + [(True, initdist, phgen) for _, initdist, phgen in DPHSTRUCTURES]
    + [(discrete, initdist, phgen)
       for _, discrete, initdist, phgen, _, _ in PHASECASES]
)

for discrete, initdist, phgen in PHASEINVARIANTS:

    distribution = makedist(discrete, initdist, phgen)
    timematrix = np.asarray(distribution.getphasetimematrix())
    exitmatrix = np.asarray(distribution.getexitprobmatrix())
    timevector = np.asarray(distribution.getphasetime()).ravel()
    exitvector = np.asarray(distribution.getexitprob()).ravel()
    nphases = np.asarray(phgen).shape[0]

    if (timematrix.shape != (nphases, nphases)
            or exitmatrix.shape != (nphases, nphases)
            or timevector.size != nphases or exitvector.size != nphases):
        sys.exit("Validation test failed at case 5: the phase statistics do not have one entry per phase.")

    if np.min(timematrix) < 0.0 or np.min(exitmatrix) < 0.0:
        sys.exit("Validation test failed at case 5: a phase statistic is negative.")

    # each row of the exit matrix is a distribution over the phase the process
    # exits from, since it has to exit from one of them
    if np.max(np.abs(np.sum(exitmatrix, axis=1) - 1.0)) > TOL:
        sys.exit("Validation test failed at case 5: the rows of the exit-phase matrix do not sum to one.")

    if abs(np.sum(exitvector) - 1.0) > TOL:
        sys.exit("Validation test failed at case 5: the probabilities of exiting from each phase do not sum to one.")

    # the time spent in the phases adds up to the time until absorption
    if abs(np.sum(timevector) - distribution.getmean()) > TOL:
        sys.exit("Validation test failed at case 5: the expected times per phase do not sum to the mean of the distribution.")

    # the process spends at least one holding time in the phase it starts in,
    # which in the discrete case is one step
    if discrete:
        if np.min(np.diag(timematrix)) < 1.0 - TOL:
            sys.exit("Validation test failed at case 5: the process spends less than one step in the phase it starts in.")
    else:
        holding = 1.0 / -np.diag(np.asarray(phgen, dtype=float))
        if np.min(np.diag(timematrix) - holding) < -TOL:
            sys.exit("Validation test failed at case 5: the process spends less than one mean holding time in the phase it starts in.")

    # the vectors are the matrices weighted by the initial distribution
    flat = np.asarray(initdist, dtype=float).ravel()

    if np.max(np.abs(timevector - np.matmul(flat, timematrix))) > TOL:
        sys.exit("Validation test failed at case 5: the expected time per phase is not the phase-time matrix weighted by the initial distribution.")

    if np.max(np.abs(exitvector - np.matmul(flat, exitmatrix))) > TOL:
        sys.exit("Validation test failed at case 5: the probability of exiting from each phase is not the exit-phase matrix weighted by the initial distribution.")

# Finally the same quantities from a direct simulation of the underlying
# process, which is the only check here that does not go through the matrix
# algebra the class uses.
#
# A simulated mean never equals the value it estimates, so rather than allowing
# a fixed difference, each theoretical value is required to lie inside a
# confidence interval for the simulated one. The two quantities call for
# different intervals: a phase time is a mean of times, so its interval comes
# from Student's t distribution, while an exit probability is a count out of the
# runs, so its interval is the exact Clopper-Pearson one for a binomial count.
# The advantage over a fixed tolerance is that the rate at which this case fails
# on correct code is then known rather than guessed, and it is set below.
SIMULATED = (
    ("the continuous general distribution", False, INITDIST, CPHGEN),
    ("the discrete general distribution", True, INITDIST, DPHGEN),
    ("continuous phases in series", False, np.array([1.0, 0.0, 0.0]),
     CPHSERIESGEN),
    ("discrete phases in series", True, np.array([1.0, 0.0, 0.0]),
     DPHSERIESGEN),
    ("continuous phases in parallel", False, INITDIST, CPHPARALLELGEN),
    ("discrete phases in parallel", True, INITDIST, DPHPARALLELGEN),
)

# One interval is checked per phase for each of the two quantities, and a single
# interval missing its value would fail the case. The confidence level is
# therefore applied to all of them together rather than to each one: every
# interval is taken at a level corrected for how many there are, so that the case
# passes correct code with probability 1-ALPHA in total. Taking each of the
# thirty-six at 99% instead would fail correct code far too often, on 21.7% of
# three hundred trial runs, against 0.0% of them for the corrected level used
# here. Bonferroni's correction is used because it needs no assumption about how
# the intervals depend on one another, and they do depend on one another, every
# interval of a structure being read off the same runs.
NINTERVALS = sum(2 * np.asarray(phgen).shape[0]
                 for _, _, _, phgen in SIMULATED)

SIMLEVEL = 1.0 - ALPHA / NINTERVALS

for index, (label, discrete, initdist, phgen) in enumerate(SIMULATED):

    distribution = makedist(discrete, initdist, phgen)
    simtime, simexitphase = simulatephases(discrete, initdist, phgen,
                                           NSIMPHASES, SEED * 10 + index)

    phasetime = np.asarray(distribution.getphasetime()).ravel()
    exitprob = np.asarray(distribution.getexitprob()).ravel()
    nphases = phasetime.size

    # the expected time spent in each phase
    lower, upper = meaninterval(simtime, SIMLEVEL)

    for i in range(nphases):
        if upper[i] - lower[i] < TOL:
            # a phase whose time never varies from run to run, which is to say
            # one the process either always or never spends the same time in,
            # leaves no room for simulation error
            if abs(phasetime[i] - lower[i]) > TOL:
                sys.exit("Validation test failed at case 5: the expected time in phase %d of %s is %.6f, but every simulated run spent %.6f there." % (i + 1, label, phasetime[i], lower[i]))
        elif phasetime[i] < lower[i] or phasetime[i] > upper[i]:
            sys.exit("Validation test failed at case 5: the expected time in phase %d of %s is %.6f, outside the %.4f%% confidence interval [%.6f, %.6f] for the mean of %d simulated runs." % (i + 1, label, phasetime[i], 100.0 * SIMLEVEL, lower[i], upper[i], NSIMPHASES))

    # the probability of exiting from each phase
    counts = np.array([int(np.sum(simexitphase == i)) for i in range(nphases)])

    # every run has to exit from exactly one phase, or the counts below are not
    # binomial ones and the intervals do not apply
    if np.sum(counts) != NSIMPHASES:
        sys.exit("Validation test failed at case 5: the %d simulated runs of %s exit from a phase %d times in total." % (NSIMPHASES, label, np.sum(counts)))

    lower, upper = clopperpearsoninterval(counts, NSIMPHASES, SIMLEVEL)

    for i in range(nphases):
        if exitprob[i] < lower[i] or exitprob[i] > upper[i]:
            sys.exit("Validation test failed at case 5: the probability of exiting from phase %d of %s is %.6f, outside the %.4f%% Clopper-Pearson interval [%.6f, %.6f] for the %d of %d simulated runs that did so." % (i + 1, label, exitprob[i], 100.0 * SIMLEVEL, lower[i], upper[i], counts[i], NSIMPHASES))


# ------------------------------------------------------------------
# CASE 6: The samples follow the distribution they are drawn from
# ------------------------------------------------------------------

# the size argument, as documented
if np.ndim(CPH.getrandom(size=1)) != 0:
    sys.exit("Validation test failed at case 6: a single continuous sample is not a scalar.")

if np.ndim(DPH.getrandom(size=1)) != 0:
    sys.exit("Validation test failed at case 6: a single discrete sample is not a scalar.")

if not np.isnan(CPH.getrandom(size=0)):
    sys.exit("Validation test failed at case 6: a sample of size zero is not a missing value.")

for size in (2, 7):
    if np.asarray(CPH.getrandom(size=size)).size != size:
        sys.exit("Validation test failed at case 6: the number of continuous samples returned differs from the size requested.")
    if np.asarray(DPH.getrandom(size=size)).size != size:
        sys.exit("Validation test failed at case 6: the number of discrete samples returned differs from the size requested.")

# discrete samples must be whole numbers of at least one
samples = np.asarray(DPH.getrandom(size=200), dtype=float)
if np.any(samples < 1.0) or np.any(samples % 1.0 != 0.0):
    sys.exit("Validation test failed at case 6: the discrete samples are not whole numbers of at least one.")

# the goodness-of-fit tests themselves. The seed of each test is derived from
# SEED so that the whole case is reproducible.
CHISQUARETESTS = []

for index, (name, initdist, phgen) in enumerate(CPHSTRUCTURES):
    CHISQUARETESTS.append(("continuous " + name + ", direct sampling",
                           cphchisquare(initdist, phgen, SEED * 100 + index,
                                        "direct", NSAMPLES)))

CHISQUARETESTS.append(("continuous general, quantile sampling",
                       cphchisquare(CPHSTRUCTURES[0][1], CPHSTRUCTURES[0][2],
                                    SEED * 100 + 3, "quantile",
                                    NSAMPLESQUANTILE)))

for index, (name, initdist, phgen) in enumerate(DPHSTRUCTURES):
    CHISQUARETESTS.append(("discrete " + name + ", direct sampling",
                           dphchisquare(initdist, phgen,
                                        SEED * 100 + 4 + index, NSAMPLES)))

for name, (statistic, degrees, minexpected) in CHISQUARETESTS:

    # a chi-square test is only valid when no bin is nearly empty, and it only
    # says something when there is more than one degree of freedom
    if minexpected < MINEXPECTED:
        sys.exit("Validation test failed at case 6: the bins used for %s have an expected count below %.0f." % (name, MINEXPECTED))

    if degrees < 2:
        sys.exit("Validation test failed at case 6: the bins used for %s leave fewer than two degrees of freedom." % name)

    if statistic > chi2.ppf(1.0 - ALPHA, degrees):
        sys.exit("Validation test failed at case 6: the samples for %s do not follow the distribution they are drawn from (chi-square %.2f on %d degrees of freedom, above the %.0f%% critical value %.2f)." % (name, statistic, degrees, 100 * (1.0 - ALPHA), chi2.ppf(1.0 - ALPHA, degrees)))

if len(CHISQUARETESTS) != len(CPHSTRUCTURES) + len(DPHSTRUCTURES) + 1:
    sys.exit("Validation test failed at case 6: not every structure was submitted to a chi-square test.")


# ------------------------------------------------------------------
# CASE 7: The parameter count is the one worked out by hand
# ------------------------------------------------------------------

# A phase contributes one parameter for every non-zero transition it can make,
# counting absorption, less one because the row is constrained. The initial
# distribution contributes one for every non-zero entry, less one. For a fully
# general three-phase distribution that is 3*(3+1-1) + (3-1) = 11; for phases
# in series with absorption only from the last one it is 3*1 + 0 = 3.
PARAMETERCOUNTS = (
    (False, INITDIST, CPHGEN, 11),
    (False, np.array([1.0, 0.0, 0.0]),
     np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, -1.0]]), 3),
    (True, INITDIST, DPHGEN, 11),
    (True, np.array([1.0, 0.0, 0.0]),
     np.array([[0.55, 0.45, 0.0], [0.0, 0.55, 0.45], [0.0, 0.0, 0.55]]), 3),
)

for discrete, initdist, phgen, expected in PARAMETERCOUNTS:
    if makedist(discrete, initdist, phgen).countParameters() != expected:
        sys.exit("Validation test failed at case 7: the number of parameters counted is not the number worked out by hand.")


# ------------------------------------------------------------------
# CASE 8: A plot is written for both plot types and both kinds
# ------------------------------------------------------------------

PLOTDIRECTORY = os.path.dirname(os.path.abspath(__file__))

for label, distribution in (("cph", CPH), ("dph", DPH)):
    for plottype in ("pdf", "cdf"):

        filename = os.path.join(PLOTDIRECTORY,
                                "_testplot_%s_%s.png" % (label, plottype))

        # never overwrite something that is already there
        if os.path.exists(filename):
            sys.exit("Validation test failed at case 8: the file %s already exists, so the test will not write to it." % filename)

        try:
            # anything printed while plotting means a point was evaluated where
            # the distribution is not defined
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                distribution.plot(type=plottype, filename=filename)

            if printed.getvalue() != "":
                sys.exit("Validation test failed at case 8: plotting the %s of the %s distribution printed %r." % (plottype, label, printed.getvalue().strip()))

            if not os.path.exists(filename):
                sys.exit("Validation test failed at case 8: plotting the %s of the %s distribution did not write a file." % (plottype, label))

            if os.path.getsize(filename) == 0:
                sys.exit("Validation test failed at case 8: plotting the %s of the %s distribution wrote an empty file." % (plottype, label))

            # the figure must actually hold values, not a curve of missing ones
            figure = plt.gcf()
            plotted = np.concatenate([line.get_ydata()
                                      for axes in figure.axes
                                      for line in axes.lines])
            plt.close(figure)

            if plotted.size == 0 or not np.all(np.isfinite(plotted)):
                sys.exit("Validation test failed at case 8: the %s plotted for the %s distribution contains no finite values." % (plottype, label))
        finally:
            if os.path.exists(filename):
                os.remove(filename)

        if os.path.exists(filename):
            sys.exit("Validation test failed at case 8: the plot written for the %s of the %s distribution could not be removed again." % (plottype, label))


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print("dist: All tests passed.")
