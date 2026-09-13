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
    Case 5: getrandom passes chi-square goodness-of-fit tests, and its size
            argument behaves as documented.
    Case 6: countParameters reproduces counts worked out by hand.
    Case 7: plot writes a file for both plot types and both kinds of
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

# A non-interactive backend, so that the plots of case 7 can be written without
# a display. This has to be selected before dist is imported, since importing
# it loads pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.linalg import expm
from scipy.stats import chi2

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

ALPHA = 0.01            # significance level of the chi-square tests
NSAMPLES = 10000        # samples drawn for a chi-square test
NSAMPLESQUANTILE = 4000  # samples for the slower quantile-based sampler
NBINS = 10              # bins used for a continuous chi-square test
MINBINPROB = 0.05       # smallest probability a discrete bin may carry
MINEXPECTED = 5.0       # smallest expected count a chi-square bin may have

# Every public method of dist. Checked against the class below, so that a new
# public method cannot slip in without a case covering it.
COVEREDMETHODS = ("countParameters", "getcumprob", "getdensity", "getexitrates",
                  "getinitdist", "getmean", "getphasegen", "getquantile",
                  "getrandom", "getvar", "plot")

# The two distributions used for cases 1 to 4 and 6. Both generators are fully
# general, and their rows all sum to the same value, so every exit rate is
# equal.
INITDIST = np.array([0.5, 0.3, 0.2])

CPHGEN = np.array([[-1.2, 0.4, 0.3],
                   [0.2, -0.9, 0.2],
                   [0.5, 0.3, -1.3]])      # rows sum to -RATE

DPHGEN = np.array([[0.40, 0.20, 0.10],
                   [0.10, 0.50, 0.10],
                   [0.20, 0.20, 0.30]])    # rows sum to 1-PROB

# Structures used for the chi-square tests of case 5, in the order their seeds
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
# CASE 5: The samples follow the distribution they are drawn from
# ------------------------------------------------------------------

# the size argument, as documented
if np.ndim(CPH.getrandom(size=1)) != 0:
    sys.exit("Validation test failed at case 5: a single continuous sample is not a scalar.")

if np.ndim(DPH.getrandom(size=1)) != 0:
    sys.exit("Validation test failed at case 5: a single discrete sample is not a scalar.")

if not np.isnan(CPH.getrandom(size=0)):
    sys.exit("Validation test failed at case 5: a sample of size zero is not a missing value.")

for size in (2, 7):
    if np.asarray(CPH.getrandom(size=size)).size != size:
        sys.exit("Validation test failed at case 5: the number of continuous samples returned differs from the size requested.")
    if np.asarray(DPH.getrandom(size=size)).size != size:
        sys.exit("Validation test failed at case 5: the number of discrete samples returned differs from the size requested.")

# discrete samples must be whole numbers of at least one
samples = np.asarray(DPH.getrandom(size=200), dtype=float)
if np.any(samples < 1.0) or np.any(samples % 1.0 != 0.0):
    sys.exit("Validation test failed at case 5: the discrete samples are not whole numbers of at least one.")

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
        sys.exit("Validation test failed at case 5: the bins used for %s have an expected count below %.0f." % (name, MINEXPECTED))

    if degrees < 2:
        sys.exit("Validation test failed at case 5: the bins used for %s leave fewer than two degrees of freedom." % name)

    if statistic > chi2.ppf(1.0 - ALPHA, degrees):
        sys.exit("Validation test failed at case 5: the samples for %s do not follow the distribution they are drawn from (chi-square %.2f on %d degrees of freedom, above the %.0f%% critical value %.2f)." % (name, statistic, degrees, 100 * (1.0 - ALPHA), chi2.ppf(1.0 - ALPHA, degrees)))

if len(CHISQUARETESTS) != len(CPHSTRUCTURES) + len(DPHSTRUCTURES) + 1:
    sys.exit("Validation test failed at case 5: not every structure was submitted to a chi-square test.")


# ------------------------------------------------------------------
# CASE 6: The parameter count is the one worked out by hand
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
        sys.exit("Validation test failed at case 6: the number of parameters counted is not the number worked out by hand.")


# ------------------------------------------------------------------
# CASE 7: A plot is written for both plot types and both kinds
# ------------------------------------------------------------------

PLOTDIRECTORY = os.path.dirname(os.path.abspath(__file__))

for label, distribution in (("cph", CPH), ("dph", DPH)):
    for plottype in ("pdf", "cdf"):

        filename = os.path.join(PLOTDIRECTORY,
                                "_testplot_%s_%s.png" % (label, plottype))

        # never overwrite something that is already there
        if os.path.exists(filename):
            sys.exit("Validation test failed at case 7: the file %s already exists, so the test will not write to it." % filename)

        try:
            # anything printed while plotting means a point was evaluated where
            # the distribution is not defined
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                distribution.plot(type=plottype, filename=filename)

            if printed.getvalue() != "":
                sys.exit("Validation test failed at case 7: plotting the %s of the %s distribution printed %r." % (plottype, label, printed.getvalue().strip()))

            if not os.path.exists(filename):
                sys.exit("Validation test failed at case 7: plotting the %s of the %s distribution did not write a file." % (plottype, label))

            if os.path.getsize(filename) == 0:
                sys.exit("Validation test failed at case 7: plotting the %s of the %s distribution wrote an empty file." % (plottype, label))

            # the figure must actually hold values, not a curve of missing ones
            figure = plt.gcf()
            plotted = np.concatenate([line.get_ydata()
                                      for axes in figure.axes
                                      for line in axes.lines])
            plt.close(figure)

            if plotted.size == 0 or not np.all(np.isfinite(plotted)):
                sys.exit("Validation test failed at case 7: the %s plotted for the %s distribution contains no finite values." % (plottype, label))
        finally:
            if os.path.exists(filename):
                os.remove(filename)

        if os.path.exists(filename):
            sys.exit("Validation test failed at case 7: the plot written for the %s of the %s distribution could not be removed again." % (plottype, label))


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print("dist: All tests passed.")
