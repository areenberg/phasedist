'''
UNIT TEST FOR THE fitcph2dist CLASS (fits a continuous phase-type distribution
to another DISTRIBUTION, rather than to data, with the EM algorithm).

Every public method of the class is covered by one of the cases below, and the
list of covered methods is checked against the class itself, so that a method
added later cannot go untested unnoticed.

What makes this file different from test_fitcph.py. There are no observations
here. The target is a distribution given in closed form, so the quantity the
fit is compared against is EXACT and carries no sampling error. Clopper-Pearson
intervals, t intervals and chi-square tests therefore have nothing to do: they
exist to account for sampling error, and there is none. The only randomness is
the random starting point, so the criteria below were instead calibrated across
seeds, and the measured worst case is recorded next to each constant.

How the fits are judged. The fitted parameters themselves are not a valid
target: a phase-type representation is heavily over-parametrized (Bladt and
Nielsen 2017, Theorem 3.1.22, p. 138), so many parameter sets describe the same
distribution and the EM algorithm is free to move among them. What is
identified is the distribution, so every criterion below is on the fitted
distribution function, compared with the target on a grid of its own quantiles.
The mean absolute difference over that grid is used rather than the maximum at
any one point, since a criterion required to hold at every point separately is
markedly less stable than one on the average.

The three properties the class exists to have are cases 3, 4 and 5:

  case 3  more EM iterations bring the fit closer to the target
  case 4  more phases bring the fit closer to the target
  case 5  a target that IS a phase-type distribution is recovered almost
          exactly, the remaining error being the discretization of the target
          rather than the fit

Case 4 is asserted over one to four phases. At five phases the EM algorithm
reaches the four-phase optimum with a redundant phase often enough that the
step is not reliable from a single start -- it failed on 5 of 15 seeds -- so
the five-phase step is asserted on the best of REPLICATESTARTS random starts
instead, which is both what a user would do and what makes the comparison a
statement about the model rather than about one starting point.

Case 2 asserts the one exact invariant available: the EM algorithm cannot
decrease the log-likelihood. A wrong M-step usually breaks it, so it is a sharp
diagnostic and it holds for every seed and every structure.

Case 12 feeds fixed inputs straight into the E-step and compares against
formulas worked out by hand, which no criterion on a converged fit can do: a
component can be wrong and still converge to something plausible. For one phase
the E-step collapses to closed forms that do not depend on the rate at all, and
for two phases with distinct rates the result is compared against numerical
quadrature of the defining integral, which shares no algebra with the Van Loan
block identity the class uses.

The 'percentiles' target is judged differently from the others, in case 6. It
approximates the distribution function by interpolating between the points it
is given, so what it can reach depends on how dense they are, and comparing it
with the target alone would charge it for the model's own approximation error
as well. It is therefore compared with the SAME MODEL fitted to the
distribution itself, which is the floor available at that number of phases.
Refining the grid must close the gap to that floor, and does: forty points
reach within 1.04 of it. The coarse grid of five points up to the ninetieth
percentile is three to four times worse, and most of that is the mass above the
largest supplied point, which no cell carries.

Runtime is about 33 seconds.

Each cell of the grid carries its own conditional mean, so the mean of the
discretized target is exact rather than accurate to the order of a squared
step. Case 6 asserts that directly and case 5 holds the fitted mean of a
phase-type target to 1e-12, where the midpoints this file was first written
against reached only 3.0e-8.

The EM algorithm is accelerated (SQUAREM) by default, so an iteration of it
extrapolates along the direction the plain algorithm is converging in and
backtracks towards the plain step when that overshoots. The 'accelerate'
argument turns it off. Case 2 holds the contract that makes it safe -- an
accelerated iteration never arrives below the two plain ones it is built from --
case 9 holds the consequence, that a fit stopping on its tolerance really has
little left to gain, and case 13 holds both halves of what the argument means.

Case 13 is two-sided on purpose. Given a tolerance tight enough for both routes
to have arrived they must AGREE, since they maximise the same thing; given the
same number of iterations instead, the accelerated one must be AHEAD. The first
half alone is a check that passes most easily when the flag does nothing, and
the mutation sweep confirmed it: wiring the plain route to the accelerated one
survived until the second half was added.

Several random starts are screened by default ('restarts'), because a single
one settles in a worse optimum often enough to matter -- 10 of 18 at six
phases. Case 14 holds what that is worth. A local optimum cannot be recognised
from one run, since nothing about a stationary point says whether a better one
exists, so there is no cheaper test of it than running more than one start.

The file was mutation-tested against 46 deliberate faults in fitcph2dist and
rndcph -- transposed blocks, dropped factors, a wrong index on a denominator,
the structural zeros leaking, the tail cell removed, the cell means replaced by
midpoints or by right endpoints, the partial expectation differenced the wrong
way, the percentile cells read off one endpoint twice, Aitken's max turned into
a min, a one in ten million rescaling of the exit rates, the validation of a
structure name and of a flag removed, the plot leaking its figures, drawing no
target curve or drawing the wrong one, and the two broadcasts that the shared
random-start class once had. All 30 are caught, and every case above catches at
least one of them on its own. With case 2 disabled the whole
set is still caught, so the log-likelihood invariant and the two convergence
cases have independent bite rather than one standing in for the others.

Two of the faults are caught by exactly one case each, which is why those cases
are not redundant: the rescaling of the exit rates is some four hundred times
below the discretization error of the method, so only the hand-computed M-step
of case 12 resolves it, and dropping the normalization of the initial
distribution shows up only on the percentile target of case 6, that being the
one target whose cells do not already carry the whole mass.
'''

import io
import os
import subprocess
import sys
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.stats import chi2, gamma as gammadist, lognorm, norm, weibull_min

# Load phasedist from the src-folder so the test can be run without installing
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phasedist.dist import dist
from phasedist.fitcph2dist import fitcph2dist


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

TOL = 1e-12          # tolerance for comparisons that are exact in theory
TOLFIT = 1e-9        # tolerance for quantities recomputed from fitted values
TOLQUAD = 1e-10      # tolerance against numerical quadrature
# The discretized target carries the mean of the real one exactly, so this is a
# rounding tolerance and not an approximation one. Measured across the six
# targets at 10, 50 and 200 steps the largest relative error is 8.9e-16.
TOLDISCRETE = 1e-13

SEED = 3
NGRID = 25           # evaluation points for every distribution comparison

# Case 3. Fixed iteration counts, and the least the fit must improve between
# the first and the last of them. Measured over 30 seeds the error decreases at
# every checkpoint on all of them, and the worst first-to-last ratio is 3.23.
ITERCHECK = (1, 3, 10, 30, 100)
MINITERGAIN = 2.5

# Case 4. Measured over 20 seeds, one to four phases decreases at every step on
# all of them, with a worst consecutive ratio of 1.84 and a worst one-to-four
# ratio of 7.04.
PHASECHECK = (1, 2, 3, 4)
MINPHASERATIO = 1.3
MINPHASEGAIN = 4.0
# The five-phase step, over the best of this many starts. Measured over 8
# independent blocks of starts the five-phase fit wins every time, by a factor
# of at least 2.30.
REPLICATESTARTS = 3

# Case 5. A phase-type target, fitted at a fine grid. Measured over ten fits
# across five seeds and both structures: the distribution function differs by at
# most 5.0e-5, the mean by 2.7e-15 and the variance by 4.2e-4, with the spread
# across seeds under one percent, so what is left is the discretization and not
# the starting point.
#
# The mean is held to 1e-12 because the grid carries each cell at its own
# conditional mean, which makes the mean of the discretized target exact rather
# than accurate to the order of a squared step. Under the midpoints this file
# was first written against, the same fits recovered the mean only to 3.0e-8.
EXACTSTEPS = 200
EXACTTRUNCATION = 0.9999
EXACTTOLERANCE = 1e-9
TOLEXACTCDF = 1e-4
TOLEXACTMEAN = 1e-12
TOLEXACTVAR = 1e-3

# Case 6. The percentile target against the same model fitted to the
# distribution itself, which is the floor it can reach. Measured over six seeds
# the fine grid lands within 1.04 of that floor and improves on the coarse one
# by a factor of 3.69, both to two decimal places on every seed.
COARSEPERCENTILES = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
FINEPERCENTILES = np.append(np.linspace(0.01, 0.99, 39), 0.999)
MAXPERCENTILEEXCESS = 1.3
MINPERCENTILEGAIN = 2.0

# Case 9. The most a fit stopped at its tolerance may still gain from being
# continued. The generalized Erlang is used because it is the structure whose
# fits settle furthest, which leaves the clearest gap between a fit that really
# has converged and one that stopped short. Measured over seven seeds at three,
# four and five phases the largest real gain is 2.95e-6, against 4.02e-3 when
# Aitken's max becomes a min, 1.37e-4 when the acceleration stops checking that
# it improved anything, and 1.80e-4 when it extrapolates the wrong way.
# Case 13. The two ways of running the algorithm, at a tolerance tight enough
# that both have arrived. Measured over three structures, two phase counts and
# two seeds the largest disagreement is 1.3e-8 in the log-likelihood and 3.4e-8
# in the distance to the target, so this leaves near two orders of magnitude.
AGREETOLERANCE = 1e-9
TOLAGREE = 1e-6
# and, at a fixed budget of iterations rather than a tolerance, how much the
# accelerated route must be ahead by. Measured over four structures, two phase
# counts, three seeds and two budgets the smallest real gain is 7.07e-5, while
# a flag that did nothing would give exactly zero.
# Case 14. Restarts, on the structure with the most pronounced basins.
# Measured over four independent blocks of three seeds the mean gain in
# log-likelihood from screening three starts is 1.3e-2 to 1.1e-1, so this sits
# an order of magnitude below the smallest of them.
RESTARTSTRUCTURE = "gencoxian"
RESTARTPHASES = 4
RESTARTCOUNT = 3
RESTARTSEEDS = 3
MINRESTARTGAIN = 1e-3

ACCELERATEBUDGET = 20
MINACCELERATION = 1e-5

RESUMEPHASES = (3, 4, 5)
RESUMEITERATIONS = 2000
MAXIMPROVEMENT = 2e-5

PLOTFILE = "test_fitcph2dist_plot.png"


# ------------------------------------------------------------------
# The target distributions.
#
# LOGNORMAL. Squared coefficient of variation 1/4, so an order-n phase-type
# distribution cannot match its variance until n reaches 4 (its variance is at
# least mean^2/n, with equality for the Erlang). That leaves something for every
# added phase to buy, which is what cases 3 and 4 need.
#
# PHASETYPE. A generalized Erlang with DISTINCT rates 2 and 1, so the two phases
# are not exchangeable and an index swapped between them changes the answer. A
# common rate would make the occupancies equal and hide exactly that.
# ------------------------------------------------------------------

LOGNORMMEAN = 2.0
LOGNORMVAR = 1.0
LOGNORMMU = np.log(LOGNORMMEAN ** 2 / np.sqrt(LOGNORMMEAN ** 2 + LOGNORMVAR))
LOGNORMSIGMA = np.sqrt(np.log(1.0 + LOGNORMVAR / LOGNORMMEAN ** 2))

PHASEPI = np.array([1.0, 0.0])
PHASEGEN = np.array([[-2.0, 2.0], [0.0, -1.0]])

GAMMASHAPE = 2.0
GAMMASCALE = 1.5
WEIBULLSHAPE = 1.5
WEIBULLSCALE = 2.0
CHISQDF = 3
NORMMU = 1.0         # close enough to zero that the truncation at zero bites,
NORMSIGMA = 1.0      # so the truncated mean differs from mu and is a real check

PERCENTILEP = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

# Case 8 fits a hyperexponential, which cannot have a squared coefficient of
# variation below one, so it needs a target above that bound before distinct
# rates are the right answer rather than merely a possible one. Measured over
# seeds the rates come out spread by 0.477, so this is two orders of magnitude
# below what happens and far above the 8e-5 an unconverged collapse leaves.
WIDEVAR = 8.0
WIDESPREAD = 1e-2

# The five structures the class offers, and custom
STRUCTURES = ("general", "generlang", "hyperexp", "coxian", "gencoxian")


# An order-n phase-type distribution has variance at least mean^2/n, with
# equality for the Erlang. The target must therefore be out of reach at every
# phase count below the last of PHASECHECK and reachable at the last, or the
# sequence of case 4 has nothing to buy and proves nothing.
if not (LOGNORMVAR >= LOGNORMMEAN ** 2 / PHASECHECK[-1] - TOL
        and LOGNORMVAR < LOGNORMMEAN ** 2 / (PHASECHECK[-1] - 1)):
    sys.exit("Validation test failed at initialization: the lognormal variance %.6f is not first reachable at %d phases (the bound there is %.6f and at %d phases it is %.6f), so the phase sequence of case 4 proves nothing." % (LOGNORMVAR, PHASECHECK[-1], LOGNORMMEAN ** 2 / PHASECHECK[-1], PHASECHECK[-1] - 1, LOGNORMMEAN ** 2 / (PHASECHECK[-1] - 1)))

# The wide target must really need a mixture, or the hyperexponential check of
# case 8 passes whatever the starting point did
if WIDEVAR <= LOGNORMMEAN ** 2:
    sys.exit("Validation test failed at initialization: the wide target has squared coefficient of variation %.4f, which a hyperexponential can reach with a single exponential, so case 8 would prove nothing." % (WIDEVAR / LOGNORMMEAN ** 2))

# The phase-type target must be a valid representation with distinct rates
if (abs(float(np.sum(PHASEPI)) - 1.0) > TOL
        or float(np.min(PHASEPI)) < 0.0
        or float(np.max(PHASEGEN - np.diag(np.diag(PHASEGEN)))) < 0.0
        or float(np.max(np.sum(PHASEGEN, axis=1))) > TOL
        or abs(PHASEGEN[0, 0] - PHASEGEN[1, 1]) < TOL):
    sys.exit("Validation test failed at initialization: the phase-type target is not a valid representation with distinct rates.")

# Every public method must be covered by a case
COVEREDMETHODS = ("chisq", "fit", "gamma", "getcumprob", "getdensity",
                  "getdist", "getexitrates", "getinitdist", "getmean",
                  "getphasegen", "getquantile", "getvar", "lognorm",
                  "norm", "percentiles", "phasedist", "plot", "weibull")

publicmethods = tuple(sorted(name for name in dir(fitcph2dist)
                             if not name.startswith("_")
                             and callable(getattr(fitcph2dist, name))))

if publicmethods != tuple(sorted(COVEREDMETHODS)):
    sys.exit("Validation test failed at initialization: the public methods of fitcph2dist are %s, but the test covers %s." % (publicmethods, tuple(sorted(COVEREDMETHODS))))


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def quiet(function, *arguments, **keywords):
    """Runs a function with its printing and its warnings suppressed."""
    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return function(*arguments, **keywords)
    finally:
        sys.stdout.close()
        sys.stdout = stdout


def lognormfit(seed=SEED, nphases=4, **keywords):
    """Fits the lognormal target and returns the fitted object."""
    def build():
        model = fitcph2dist(nphases=nphases, seed=seed, **keywords)
        model.lognorm(mu=LOGNORMMU, sigma=LOGNORMSIGMA)
        model.fit()
        return model
    return quiet(build)


LOGNORMGRID = lognorm.ppf(np.linspace(0.01, 0.95, NGRID), LOGNORMSIGMA,
                          scale=np.exp(LOGNORMMU))
LOGNORMTRUE = lognorm.cdf(LOGNORMGRID, LOGNORMSIGMA, scale=np.exp(LOGNORMMU))


def cdfdistance(model, grid, true):
    """Mean absolute difference between the fitted and the target CDF."""
    fitted = np.array([model.getcumprob(x) for x in grid])
    return float(np.mean(np.abs(fitted - true)))


def phasetypecdf(pi, phgen, x):
    """CDF of a phase-type distribution, formed directly from its definition."""
    return 1.0 - float(pi @ expm(np.asarray(phgen, dtype=float) * x)
                       @ np.ones(np.asarray(phgen).shape[0]))


def phasetypedensity(pi, phgen, x):
    """Density of a phase-type distribution, formed from its definition."""
    phgen = np.asarray(phgen, dtype=float)
    exitrates = -phgen @ np.ones(phgen.shape[0])
    return float(pi @ expm(phgen * x) @ exitrates)


def parametersof(model):
    """The fitted parameters as plain one- and two-dimensional arrays."""
    return (np.asarray(model.getinitdist(), dtype=float).ravel(),
            np.asarray(model.getphasegen(), dtype=float),
            np.asarray(model.getexitrates(), dtype=float).ravel())


def isfeasible(pi, phgen, exitrates):
    """Whether the parameters form a valid phase-type representation."""
    offdiagonal = phgen - np.diag(np.diag(phgen))
    return (abs(float(np.sum(pi)) - 1.0) <= TOLFIT
            and float(np.min(pi)) >= -TOL
            and float(np.min(offdiagonal)) >= -TOL
            and float(np.max(np.diag(phgen))) < 0.0
            and float(np.max(np.sum(phgen, axis=1))) <= TOLFIT
            and float(np.min(exitrates)) >= -TOL)


def randomstart(seed, nphases=3, dtype="general"):
    """The random starting point, read by running no iterations at all."""
    def build():
        model = fitcph2dist(nphases=nphases, dtype=dtype, seed=seed, itermax=0)
        model.lognorm(mu=LOGNORMMU, sigma=LOGNORMSIGMA)
        model.fit()
        return model
    return parametersof(quiet(build))


def estepof(nphases, pi, phgen, exitrates, y, hy, maximize=False):
    """Runs the E-step, and on request the M-step, on a fixed grid."""
    def build():
        model = fitcph2dist(nphases=nphases, seed=1, itermax=0)
        model.phasedist(initdist=np.array([1.0] + [0.0] * (nphases - 1)),
                        phgen=np.array(phgen, dtype=float))
        model.fit()
        return model
    model = quiet(build)
    model.pi = np.array(pi, dtype=float)
    model.phgen = np.array(phgen, dtype=float)
    model.exitrates = np.array(exitrates, dtype=float)
    model.y = np.array(y, dtype=float)
    model.hy = np.array(hy, dtype=float)
    model.ncells = model.y.size
    # the steps are private, but a component observed only through the answer a
    # converged fit arrives at can be wrong and still look plausible
    getattr(model, "_fitcph2dist__estep")()
    if maximize:
        getattr(model, "_fitcph2dist__mstep")()
    return model


def runsnippet(expression):
    """Runs an expression in a fresh interpreter, returning status and output."""
    source = ("import sys, numpy as np\n"
              "sys.path.insert(0, %r)\n"
              "from phasedist.fitcph2dist import fitcph2dist\n"
              "%s\n"
              % (os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "src"), expression))
    result = subprocess.run([sys.executable, "-c", source],
                            capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


try:

    # ------------------------------------------------------------------
    # CASE 1
    # The getters against the phase-type formulas, formed independently with
    # scipy's matrix exponential rather than with the class's uniformization.
    # ------------------------------------------------------------------

    model = lognormfit()
    pi, phgen, exitrates = parametersof(model)

    if not isfeasible(pi, phgen, exitrates):
        sys.exit("Validation test failed at case 1: the fitted parameters are not a valid phase-type representation.")

    if np.max(np.abs(exitrates + phgen @ np.ones(phgen.shape[0]))) > TOLFIT:
        sys.exit("Validation test failed at case 1: the exit rates are not minus the row sums of the fitted generator.")

    for x in LOGNORMGRID[::6]:
        if abs(model.getcumprob(x) - phasetypecdf(pi, phgen, x)) > TOLFIT:
            sys.exit("Validation test failed at case 1: getcumprob at %.6f disagrees with 1 - pi exp(Tx) e." % x)
        if abs(model.getdensity(x) - phasetypedensity(pi, phgen, x)) > TOLFIT:
            sys.exit("Validation test failed at case 1: getdensity at %.6f disagrees with pi exp(Tx) t." % x)

    green = np.linalg.inv(-phgen)
    truemean = float(pi @ green @ np.ones(phgen.shape[0]))
    truevar = (2.0 * float(pi @ green @ green @ np.ones(phgen.shape[0]))
               - truemean ** 2)

    if abs(model.getmean() - truemean) > TOLFIT:
        sys.exit("Validation test failed at case 1: getmean disagrees with pi (-T)^-1 e.")
    if abs(model.getvar() - truevar) > TOLFIT:
        sys.exit("Validation test failed at case 1: getvar disagrees with the second moment of the fitted representation.")

    for probability in (0.25, 0.5, 0.9):
        quantile = model.getquantile(probability)
        if abs(model.getcumprob(quantile) - probability) > 1e-6:
            sys.exit("Validation test failed at case 1: getquantile(%.2f) is not the point where the fitted CDF reaches it." % probability)

    fitteddist = model.getdist()
    if not isinstance(fitteddist, dist):
        sys.exit("Validation test failed at case 1: getdist does not return a dist object.")
    if abs(fitteddist.getmean() - model.getmean()) > TOL:
        sys.exit("Validation test failed at case 1: the returned dist object does not carry the fitted distribution.")

    # ------------------------------------------------------------------
    # CASE 2
    # The EM algorithm cannot decrease the log-likelihood. A non-finite
    # tolerance asks for a fixed number of iterations, so the sequence is read
    # by running the fit again for each length.
    # ------------------------------------------------------------------

    for structure in STRUCTURES:
        sequence = []
        for steps in range(1, 26):
            sequence.append(lognormfit(nphases=3, dtype=structure,
                                       tolerance=-np.inf,
                                       itermax=steps).loglikelihood)
        sequence = np.array(sequence)

        if not np.all(np.isfinite(sequence)):
            sys.exit("Validation test failed at case 2: the log-likelihood is not finite throughout the fit of the %s structure." % structure)
        if np.min(np.diff(sequence)) < -TOLFIT:
            sys.exit("Validation test failed at case 2: the log-likelihood decreased by %.3e during the fit of the %s structure, which the EM algorithm cannot do." % (-np.min(np.diff(sequence)), structure))
        if sequence[-1] - sequence[0] <= TOLFIT:
            sys.exit("Validation test failed at case 2: the log-likelihood did not rise at all over 25 iterations of the %s structure, so monotonicity is holding vacuously." % structure)

    # The accelerated iteration extrapolates along the direction the EM
    # algorithm is converging in, and backtracks towards the plain algorithm
    # when that overshoots. Its contract is therefore not merely that the
    # log-likelihood rises, but that one accelerated iteration never arrives
    # anywhere WORSE than the two plain iterations it is built from, since at
    # the end of its backtracking it is exactly those two. That is what makes
    # the acceleration safe to have on by default, and no criterion on a
    # converged fit tests it: a damaged extrapolation still converges, just to
    # somewhere else.
    for structure in STRUCTURES:
        for started in (0, 3):
            model = lognormfit(nphases=4, dtype=structure, tolerance=-np.inf,
                               itermax=started)
            start = getattr(model, "_fitcph2dist__pack")()
            emstep = getattr(model, "_fitcph2dist__emstep")
            loglikat = getattr(model, "_fitcph2dist__loglikat")
            squaremstep = getattr(model, "_fitcph2dist__squaremstep")

            emstep()
            emstep()
            plain = loglikat(getattr(model, "_fitcph2dist__pack")())

            loglikat(start)
            squaremstep()
            accelerated = loglikat(getattr(model, "_fitcph2dist__pack")())

            if accelerated < plain - TOLFIT:
                sys.exit("Validation test failed at case 2: an accelerated iteration of the %s structure reached %.12f, below the %.12f the two plain iterations it replaces reach, so the backtracking is not holding it to them." % (structure, accelerated, plain))

    # ------------------------------------------------------------------
    # CASE 3
    # More iterations bring the fitted distribution closer to the target.
    # ------------------------------------------------------------------

    errors = np.array([cdfdistance(lognormfit(tolerance=-np.inf, itermax=steps),
                                   LOGNORMGRID, LOGNORMTRUE)
                       for steps in ITERCHECK])

    if np.max(np.diff(errors)) >= 0.0:
        worst = int(np.argmax(np.diff(errors)))
        sys.exit("Validation test failed at case 3: the distance to the target did not fall between %d and %d iterations (%.6f to %.6f)." % (ITERCHECK[worst], ITERCHECK[worst + 1], errors[worst], errors[worst + 1]))

    if errors[0] / errors[-1] < MINITERGAIN:
        sys.exit("Validation test failed at case 3: %d iterations improved on %d by a factor of only %.2f, against the %.2f expected." % (ITERCHECK[-1], ITERCHECK[0], errors[0] / errors[-1], MINITERGAIN))

    # ------------------------------------------------------------------
    # CASE 4
    # More phases bring the fitted distribution closer to the target.
    # ------------------------------------------------------------------

    phaseerrors = np.array([cdfdistance(lognormfit(nphases=nphases),
                                        LOGNORMGRID, LOGNORMTRUE)
                            for nphases in PHASECHECK])

    ratios = phaseerrors[:-1] / phaseerrors[1:]

    if np.min(ratios) < MINPHASERATIO:
        worst = int(np.argmin(ratios))
        sys.exit("Validation test failed at case 4: going from %d to %d phases improved the fit by a factor of only %.2f, against the %.2f expected." % (PHASECHECK[worst], PHASECHECK[worst + 1], ratios[worst], MINPHASERATIO))

    if phaseerrors[0] / phaseerrors[-1] < MINPHASEGAIN:
        sys.exit("Validation test failed at case 4: %d phases improved on %d by a factor of only %.2f, against the %.2f expected." % (PHASECHECK[-1], PHASECHECK[0], phaseerrors[0] / phaseerrors[-1], MINPHASEGAIN))

    # The next phase pays off too, but only once the starting point is not the
    # thing being measured: a single start reaches the four-phase optimum with a
    # redundant fifth phase often enough to make it unreliable.
    bestfour = min(cdfdistance(lognormfit(seed=SEED + offset, nphases=4),
                               LOGNORMGRID, LOGNORMTRUE)
                   for offset in range(REPLICATESTARTS))
    bestfive = min(cdfdistance(lognormfit(seed=SEED + offset, nphases=5),
                               LOGNORMGRID, LOGNORMTRUE)
                   for offset in range(REPLICATESTARTS))

    if bestfour / bestfive < MINPHASERATIO:
        sys.exit("Validation test failed at case 4: over %d starts, five phases improved on four by a factor of only %.2f, against the %.2f expected." % (REPLICATESTARTS, bestfour / bestfive, MINPHASERATIO))

    # ------------------------------------------------------------------
    # CASE 5
    # A target that is itself a phase-type distribution is recovered almost
    # exactly. What is left is the discretization of the target, not the fit.
    # ------------------------------------------------------------------

    target = dist(discrete=False, initdist=np.copy(PHASEPI),
                  phgen=np.copy(PHASEGEN))
    phasegrid = np.array([target.getquantile(p)
                          for p in np.linspace(0.01, 0.95, NGRID)])
    phasetrue = np.array([phasetypecdf(PHASEPI, PHASEGEN, x)
                          for x in phasegrid])

    # the dist object and the formula must agree before anything is built on it
    if np.max(np.abs(np.array([target.getcumprob(x) for x in phasegrid])
                     - phasetrue)) > TOLFIT:
        sys.exit("Validation test failed at case 5: the dist object and 1 - pi exp(Tx) e disagree on the target itself.")

    for structure in ("general", "generlang"):
        def buildexact():
            model = fitcph2dist(nphases=2, dtype=structure, seed=SEED,
                                tolerance=EXACTTOLERANCE, steps=EXACTSTEPS,
                                truncation=EXACTTRUNCATION)
            model.phasedist(initdist=np.copy(PHASEPI), phgen=np.copy(PHASEGEN))
            model.fit()
            return model
        model = quiet(buildexact)

        distance = cdfdistance(model, phasegrid, phasetrue)
        if distance > TOLEXACTCDF:
            sys.exit("Validation test failed at case 5: the %s fit to a phase-type target differs from it by %.3e, against the %.3e allowed." % (structure, distance, TOLEXACTCDF))
        if abs(model.getmean() - target.getmean()) > TOLEXACTMEAN:
            sys.exit("Validation test failed at case 5: the %s fit to a phase-type target recovers the mean only to %.3e." % (structure, abs(model.getmean() - target.getmean())))
        if abs(model.getvar() - target.getvar()) > TOLEXACTVAR:
            sys.exit("Validation test failed at case 5: the %s fit to a phase-type target recovers the variance only to %.3e." % (structure, abs(model.getvar() - target.getvar())))

    # ------------------------------------------------------------------
    # CASE 6
    # Every target the class accepts, and the two ways of naming each.
    # ------------------------------------------------------------------

    # lognormal, by the parameters of the underlying normal and by its own
    bymoments = lognormfit()
    def bymu():
        model = fitcph2dist(nphases=4, seed=SEED)
        model.lognorm(mean=LOGNORMMEAN, var=LOGNORMVAR)
        model.fit()
        return model
    if abs(bymoments.getmean() - quiet(bymu).getmean()) > TOL:
        sys.exit("Validation test failed at case 6: naming the lognormal by mu and sigma and by its mean and variance give different fits.")

    # the gamma, by scale and by rate
    def bygamma(**keywords):
        model = fitcph2dist(nphases=4, seed=SEED)
        model.gamma(shape=GAMMASHAPE, **keywords)
        model.fit()
        return model
    if abs(quiet(bygamma, scale=GAMMASCALE).getmean()
           - quiet(bygamma, rate=1.0 / GAMMASCALE).getmean()) > TOL:
        sys.exit("Validation test failed at case 6: naming the gamma by its scale and by its rate give different fits.")

    # every target reaches the mean of the distribution it is pointed at. The
    # normal is truncated at zero, so its mean is the truncated one
    alpha = -NORMMU / NORMSIGMA
    truncatedmean = NORMMU + NORMSIGMA * norm.pdf(alpha) / norm.sf(alpha)

    TARGETS = (
        ("lognorm", lambda m: m.lognorm(mean=LOGNORMMEAN, var=LOGNORMVAR),
         LOGNORMMEAN),
        ("norm", lambda m: m.norm(mu=NORMMU, sigma=NORMSIGMA), truncatedmean),
        ("gamma", lambda m: m.gamma(shape=GAMMASHAPE, scale=GAMMASCALE),
         GAMMASHAPE * GAMMASCALE),
        ("weibull", lambda m: m.weibull(shape=WEIBULLSHAPE, scale=WEIBULLSCALE),
         WEIBULLSCALE * float(weibull_min.mean(WEIBULLSHAPE))),
        ("chisq", lambda m: m.chisq(df=CHISQDF), float(CHISQDF)),
        ("phasedist", lambda m: m.phasedist(initdist=np.copy(PHASEPI),
                                            phgen=np.copy(PHASEGEN)),
         float(target.getmean())),
    )

    for name, setup, expectedmean in TARGETS:
        def buildtarget():
            model = fitcph2dist(nphases=5, seed=SEED, steps=100,
                                truncation=0.999)
            setup(model)
            model.fit()
            return model
        model = quiet(buildtarget)
        if abs(model.getmean() - expectedmean) / expectedmean > 1e-3:
            sys.exit("Validation test failed at case 6: the %s target was fitted with mean %.6f, against the %.6f of the distribution itself." % (name, model.getmean(), expectedmean))
        if not isfeasible(*parametersof(model)):
            sys.exit("Validation test failed at case 6: the fit to the %s target is not a valid phase-type representation." % name)

        # Each cell carries its own conditional mean and the tail cell carries
        # the mean above the truncation, so the discretized target the EM
        # algorithm actually sees has the mean of the real one exactly, at any
        # grid. A midpoint would leave an error of the order of a squared step.
        discretizedmean = float(np.sum(model.hy * model.y))
        if abs(discretizedmean - expectedmean) / expectedmean > TOLDISCRETE:
            sys.exit("Validation test failed at case 6: the discretized %s target has mean %.12f, against the %.12f of the distribution it is built from." % (name, discretizedmean, expectedmean))
        if abs(float(np.sum(model.hy)) - 1.0) > TOLDISCRETE:
            sys.exit("Validation test failed at case 6: the cells of the %s target carry %.12f rather than the whole mass." % (name, float(np.sum(model.hy))))

    # The percentile target approximates the CDF by interpolating between the
    # supplied points, so refining them must bring it to the same place the
    # parametric target reaches. Both are fitted at the same number of phases,
    # so the model's own approximation error is common to the two and what is
    # left is the resolution of the percentile grid.
    def percentilefit(cumprobs):
        def build():
            model = fitcph2dist(nphases=PHASECHECK[-1], seed=SEED, steps=200)
            model.percentiles(cumprobs=np.copy(cumprobs),
                              x=lognorm.ppf(cumprobs, LOGNORMSIGMA,
                                            scale=np.exp(LOGNORMMU)))
            model.fit()
            return model
        return quiet(build)

    def parametricfit():
        model = fitcph2dist(nphases=PHASECHECK[-1], seed=SEED, steps=200,
                            truncation=0.999)
        model.lognorm(mu=LOGNORMMU, sigma=LOGNORMSIGMA)
        model.fit()
        return model

    coarse = percentilefit(COARSEPERCENTILES)
    fine = percentilefit(FINEPERCENTILES)

    floor = cdfdistance(quiet(parametricfit), LOGNORMGRID, LOGNORMTRUE)
    coarseerror = cdfdistance(coarse, LOGNORMGRID, LOGNORMTRUE)
    fineerror = cdfdistance(fine, LOGNORMGRID, LOGNORMTRUE)

    if fineerror / floor > MAXPERCENTILEEXCESS:
        sys.exit("Validation test failed at case 6: with %d percentiles the fit is %.2f times as far from the target as the same model fitted to the distribution itself, against the %.2f allowed." % (FINEPERCENTILES.size, fineerror / floor, MAXPERCENTILEEXCESS))

    if coarseerror / fineerror < MINPERCENTILEGAIN:
        sys.exit("Validation test failed at case 6: refining the percentiles from %d points to %d improved the fit by a factor of only %.2f, against the %.2f expected." % (COARSEPERCENTILES.size, FINEPERCENTILES.size, coarseerror / fineerror, MINPERCENTILEGAIN))

    fittedp = np.array([fine.getcumprob(x)
                        for x in lognorm.ppf(FINEPERCENTILES, LOGNORMSIGMA,
                                             scale=np.exp(LOGNORMMU))])
    if np.min(np.diff(fittedp)) <= 0.0:
        sys.exit("Validation test failed at case 6: the fitted percentile distribution function is not increasing over the supplied points.")

    # The percentile grid is the one target whose cells do NOT carry the whole
    # mass, since there is nothing above the largest supplied point to put in a
    # tail cell. It is therefore the only target on which the M-step's
    # normalization of the initial distribution does anything: everywhere else
    # the counts already sum to one and dropping it would pass unnoticed.
    if float(np.sum(coarse.hy)) >= 1.0 - TOL:
        sys.exit("Validation test failed at case 6: the percentile cells carry the whole mass, so this case no longer exercises the normalization of the initial distribution.")
    for name, model in (("coarse", coarse), ("fine", fine)):
        if not isfeasible(*parametersof(model)):
            sys.exit("Validation test failed at case 6: the %s percentile fit is not a valid phase-type representation, its initial distribution summing to %.12f." % (name, float(np.sum(parametersof(model)[0]))))

    # ------------------------------------------------------------------
    # CASE 7
    # Each structure keeps its own shape. The EM algorithm preserves structural
    # zeros, which is how a sub-family is fitted at all.
    # ------------------------------------------------------------------

    zerocounts = {}
    for structure in STRUCTURES:
        model = lognormfit(nphases=4, dtype=structure)
        pi, phgen, exitrates = parametersof(model)

        if not isfeasible(pi, phgen, exitrates):
            sys.exit("Validation test failed at case 7: the %s fit is not a valid phase-type representation." % structure)

        offdiagonal = phgen - np.diag(np.diag(phgen))
        zerocounts[structure] = (int(np.sum(offdiagonal == 0.0)) - phgen.shape[0],
                                 int(np.sum(pi == 0.0)))

    if zerocounts["general"][0] != 0:
        sys.exit("Validation test failed at case 7: the general structure acquired a structural zero off the diagonal.")
    if zerocounts["hyperexp"][0] != 4 * 4 - 4:
        sys.exit("Validation test failed at case 7: the hyperexponential structure does not have an empty off-diagonal, so it is not a mixture of exponentials.")
    for structure in ("generlang", "coxian", "gencoxian"):
        if zerocounts[structure][0] != 4 * 4 - 4 - 3:
            sys.exit("Validation test failed at case 7: the %s structure does not have exactly the three transitions of a series of four phases." % structure)
    for structure in ("generlang", "coxian"):
        if zerocounts[structure][1] != 3:
            sys.exit("Validation test failed at case 7: the %s structure does not start from a single phase." % structure)

    # the structures must actually differ, or the case above is vacuous
    if len(set(zerocounts.values())) < 3:
        sys.exit("Validation test failed at case 7: the five structures do not differ from one another as intended.")

    # ------------------------------------------------------------------
    # CASE 8
    # The starting point: reproducible from the seed, different between seeds,
    # and different BETWEEN PHASES within one start. The last of these is the
    # substance of the case: drawing one value and broadcasting it over the
    # phases leaves every phase identical, which is a stationary point of the EM
    # algorithm for the hyperexponential structure, where nothing else breaks
    # the symmetry, and the fit can then never be anything but an exponential.
    # ------------------------------------------------------------------

    firstpi, firstgen, firstexit = randomstart(SEED)
    againpi, againgen, againexit = randomstart(SEED)
    otherpi, othergen, otherexit = randomstart(SEED + 1)

    if not (np.array_equal(firstpi, againpi)
            and np.array_equal(firstgen, againgen)
            and np.array_equal(firstexit, againexit)):
        sys.exit("Validation test failed at case 8: the same seed did not reproduce the same starting point.")

    if (np.array_equal(firstpi, otherpi)
            or np.array_equal(firstexit, otherexit)):
        sys.exit("Validation test failed at case 8: two different seeds gave the same starting point.")

    if float(np.ptp(firstpi)) <= TOL:
        sys.exit("Validation test failed at case 8: every phase starts with the same initial probability, so the starting point is not random across phases.")
    if float(np.ptp(firstexit)) <= TOL:
        sys.exit("Validation test failed at case 8: every phase starts with the same exit rate, so the starting point is not random across phases.")

    # And the consequence, on the structure that has nothing else to break the
    # symmetry: a hyperexponential fit must reach distinct rates. The target
    # has to be one that REQUIRES them, which means a squared coefficient of
    # variation above one, since a mixture of exponentials cannot go below it
    # and the best hyperexponential for a target under that bound is the single
    # exponential. Fitted to the lognormal used elsewhere in this file, whose
    # squared coefficient of variation is a quarter, a collapsed fit is the
    # right answer and this check would pass whatever the starting point did.
    def widefit():
        model = fitcph2dist(nphases=3, dtype="hyperexp", seed=SEED)
        model.lognorm(mean=LOGNORMMEAN, var=WIDEVAR)
        model.fit()
        return model
    hyperexprates = parametersof(quiet(widefit))[2]

    if float(np.ptp(hyperexprates)) <= WIDESPREAD:
        sys.exit("Validation test failed at case 8: fitted to a target that needs a mixture, the hyperexponential reached rates spread by only %.3e, so it has collapsed to a single exponential." % float(np.ptp(hyperexprates)))

    # a supplied starting point is used rather than a random one
    startpi = np.array([0.6, 0.4])
    startgen = np.array([[-1.5, 0.8], [0.3, -1.2]])
    startexit = -startgen.sum(axis=1)

    def supplied(itermax):
        model = fitcph2dist(nphases=2, dtype="custom", randominit=False,
                            initdist=np.copy(startpi),
                            initphgen=np.copy(startgen),
                            initexitrates=np.copy(startexit),
                            seed=SEED, itermax=itermax)
        model.lognorm(mu=LOGNORMMU, sigma=LOGNORMSIGMA)
        model.fit()
        return model

    atstart = parametersof(quiet(supplied, 0))
    if np.max(np.abs(atstart[0] - startpi)) > TOL:
        sys.exit("Validation test failed at case 8: the supplied initial distribution was not used as the starting point.")
    if np.max(np.abs(atstart[1] - startgen)) > TOL:
        sys.exit("Validation test failed at case 8: the supplied generator was not used as the starting point.")

    if cdfdistance(quiet(supplied, 100), LOGNORMGRID, LOGNORMTRUE) >= \
            cdfdistance(quiet(supplied, 1), LOGNORMGRID, LOGNORMTRUE):
        sys.exit("Validation test failed at case 8: fitting from the supplied starting point did not improve on it.")

    # ------------------------------------------------------------------
    # CASE 9
    # Stopping. Reaching itermax with a tolerance still unmet is reported, and
    # a non-finite tolerance means a fixed number of iterations was asked for,
    # so reaching itermax is then the intent and is not reported.
    # ------------------------------------------------------------------

    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            capped = fitcph2dist(nphases=3, seed=SEED, tolerance=1e-12,
                                 itermax=3)
            capped.lognorm(mu=LOGNORMMU, sigma=LOGNORMSIGMA)
            capped.fit()
        finally:
            sys.stdout.close()
            sys.stdout = stdout
        if not any(issubclass(entry.category, RuntimeWarning)
                   and "itermax" in str(entry.message) for entry in raised):
            sys.exit("Validation test failed at case 9: stopping at itermax with the tolerance unmet was not reported.")

    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        lognormfit(nphases=3, tolerance=-np.inf, itermax=3)
        if any(issubclass(entry.category, RuntimeWarning)
               and "itermax" in str(entry.message) for entry in raised):
            sys.exit("Validation test failed at case 9: a fixed number of iterations was reported as a failure to converge.")

    # a tolerance that is met stops before itermax and is not reported
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        converged = lognormfit(nphases=3, tolerance=1e-4)
        if any(issubclass(entry.category, RuntimeWarning)
               and "itermax" in str(entry.message) for entry in raised):
            sys.exit("Validation test failed at case 9: a converged fit was reported as having stopped at itermax.")
    if converged.eps > 1e-4:
        sys.exit("Validation test failed at case 9: the fit stopped with the convergence criterion above the tolerance it was given.")

    # What the tolerance claims is the distance from here to the limit the EM
    # algorithm is converging to, not the size of the last step, which for a
    # phase-type representation understates it by orders of magnitude: the
    # likelihood has a long flat ridge, because many parameter sets describe the
    # same distribution, so the fit creeps along it. The claim is therefore
    # checked by continuing a stopped fit from where it left off and seeing what
    # is actually left to gain.
    for phases in RESUMEPHASES:
        stopped = lognormfit(nphases=phases, dtype="generlang")

        def continuefrom():
            model = fitcph2dist(
                nphases=phases, dtype="custom", randominit=False,
                initdist=np.asarray(stopped.getinitdist(), dtype=float).ravel(),
                initphgen=np.asarray(stopped.getphasegen(), dtype=float),
                initexitrates=np.asarray(stopped.getexitrates(),
                                         dtype=float).ravel(),
                tolerance=-np.inf, itermax=RESUMEITERATIONS, seed=SEED)
            model.lognorm(mu=LOGNORMMU, sigma=LOGNORMSIGMA)
            model.fit()
            return model
        gain = quiet(continuefrom).loglikelihood - stopped.loglikelihood

        if gain < -TOLFIT:
            sys.exit("Validation test failed at case 9: continuing the %d-phase fit lowered the log-likelihood by %.3e, which the EM algorithm cannot do." % (phases, -gain))
        if gain > MAXIMPROVEMENT:
            sys.exit("Validation test failed at case 9: the %d-phase fit stopped claiming less than its tolerance was left to gain, but %d further iterations gained %.3e, against the %.3e allowed." % (phases, RESUMEITERATIONS, gain, MAXIMPROVEMENT))

    # ------------------------------------------------------------------
    # CASE 13
    # The two ways of running the EM algorithm are two routes to the same
    # answer, so given a tolerance tight enough for both to have arrived they
    # must agree. This is the sharpest statement available about the
    # acceleration: it is allowed to take a different path, and nothing else.
    # ------------------------------------------------------------------

    for structure in ("general", "generlang"):
        for phases in (3, 4):
            def bothways(accelerate):
                model = fitcph2dist(nphases=phases, dtype=structure, seed=SEED,
                                    tolerance=AGREETOLERANCE,
                                    accelerate=accelerate)
                model.lognorm(mu=LOGNORMMU, sigma=LOGNORMSIGMA)
                model.fit()
                return model

            plain = quiet(bothways, False)
            accelerated = quiet(bothways, True)

            gap = abs(accelerated.loglikelihood - plain.loglikelihood)
            if gap > TOLAGREE:
                sys.exit("Validation test failed at case 13: on the %s structure at %d phases the accelerated fit reached a log-likelihood %.3e from the plain one, against the %.3e allowed." % (structure, phases, gap, TOLAGREE))

            gap = abs(cdfdistance(accelerated, LOGNORMGRID, LOGNORMTRUE)
                      - cdfdistance(plain, LOGNORMGRID, LOGNORMTRUE))
            if gap > TOLAGREE:
                sys.exit("Validation test failed at case 13: on the %s structure at %d phases the two fits are %.3e apart in their distance to the target, against the %.3e allowed." % (structure, phases, gap, TOLAGREE))

            if not isfeasible(*parametersof(plain)):
                sys.exit("Validation test failed at case 13: the unaccelerated fit of the %s structure is not a valid phase-type representation." % structure)

            # Agreeing is only half of it, and on its own it is a check that
            # passes most easily when the flag does nothing at all. The other
            # half is that the two really are different routes: given the same
            # number of iterations rather than the same tolerance, the
            # accelerated one has to be further along.
            def budgeted(accelerate):
                model = fitcph2dist(nphases=phases, dtype=structure, seed=SEED,
                                    tolerance=-np.inf, itermax=ACCELERATEBUDGET,
                                    accelerate=accelerate)
                model.lognorm(mu=LOGNORMMU, sigma=LOGNORMSIGMA)
                model.fit()
                return model.loglikelihood

            gained = quiet(budgeted, True) - quiet(budgeted, False)
            if gained < MINACCELERATION:
                sys.exit("Validation test failed at case 13: over %d iterations of the %s structure at %d phases the accelerated route gained only %.3e on the plain one, against the %.3e expected, so the two are not different routes." % (ACCELERATEBUDGET, structure, phases, gained, MINACCELERATION))

    # ------------------------------------------------------------------
    # CASE 14
    # Restarts. Several random starts are run to a loose tolerance and the one
    # reaching the highest log-likelihood is taken the rest of the way, which
    # is worth having because a single start settles in a worse optimum often
    # enough to matter. A local optimum cannot be recognised from one run --
    # nothing about a stationary point says whether a better one exists -- so
    # there is no cheaper test than trying more than one start.
    #
    # The gain is a statement about starts in aggregate, not about any one of
    # them: the screen ranks at a loose tolerance and the trajectories can
    # still cross before the fit ends, so an individual seed is occasionally a
    # little worse. The structure below is the one with the most pronounced
    # basins, which is where there is something for restarts to find.
    # ------------------------------------------------------------------

    def restarted(restarts, seed):
        def build():
            model = fitcph2dist(nphases=RESTARTPHASES, dtype=RESTARTSTRUCTURE,
                                seed=seed, restarts=restarts)
            model.lognorm(mu=LOGNORMMU, sigma=LOGNORMSIGMA)
            model.fit()
            return model
        return quiet(build)

    single = np.array([restarted(1, SEED + k).loglikelihood
                       for k in range(RESTARTSEEDS)])
    screened = np.array([restarted(RESTARTCOUNT, SEED + k).loglikelihood
                         for k in range(RESTARTSEEDS)])

    if np.mean(screened) - np.mean(single) < MINRESTARTGAIN:
        sys.exit("Validation test failed at case 14: over %d starts, screening %d of them gained %.3e in mean log-likelihood, against the %.3e expected, so the restarts are not finding a better optimum." % (RESTARTSEEDS, RESTARTCOUNT, np.mean(screened) - np.mean(single), MINRESTARTGAIN))

    for model in (restarted(1, SEED), restarted(RESTARTCOUNT, SEED)):
        if not isfeasible(*parametersof(model)):
            sys.exit("Validation test failed at case 14: a restarted fit is not a valid phase-type representation.")

    # A supplied start is the one the caller asked for, so there is nothing to
    # screen and the restarts must make no difference whatever
    def suppliedrestarts(restarts):
        model = fitcph2dist(nphases=2, dtype="custom", randominit=False,
                            initdist=np.copy(startpi),
                            initphgen=np.copy(startgen),
                            initexitrates=np.copy(startexit),
                            seed=SEED, restarts=restarts)
        model.lognorm(mu=LOGNORMMU, sigma=LOGNORMSIGMA)
        model.fit()
        return model.loglikelihood

    if quiet(suppliedrestarts, 1) != quiet(suppliedrestarts, RESTARTCOUNT):
        sys.exit("Validation test failed at case 14: the restarts changed a fit from a supplied starting point, which has nothing to screen.")

    # and a tolerance that is not finite asks for a fixed number of iterations,
    # which screening first would quietly spend before any of them are counted
    def fixedrestarts(restarts):
        return lognormfit(nphases=3, restarts=restarts, tolerance=-np.inf,
                          itermax=6).loglikelihood

    if fixedrestarts(1) != fixedrestarts(RESTARTCOUNT):
        sys.exit("Validation test failed at case 14: the restarts changed a fit asked for as a fixed number of iterations.")

    # ------------------------------------------------------------------
    # CASE 10
    # The comparison plot is written.
    # ------------------------------------------------------------------

    # before the fit there is nothing to draw, which must be said rather than
    # left as a plot carrying the target alone
    def plotunfitted():
        model = fitcph2dist(nphases=3, seed=SEED)
        model.lognorm(mu=LOGNORMMU, sigma=LOGNORMSIGMA)
        model.plot(filename=PLOTFILE)
    stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        plotunfitted()
        reported = sys.stdout.getvalue()
    finally:
        sys.stdout = stdout

    if "Error" not in reported:
        sys.exit("Validation test failed at case 10: plotting before fitting was not reported.")
    if os.path.exists(PLOTFILE):
        sys.exit("Validation test failed at case 10: plotting before fitting wrote a file anyway.")

    # every target must draw a target curve, the truncated normal and the
    # percentile step function included
    opened = len(plt.get_fignums())
    for name, setup, _ in TARGETS:
        def plotone():
            model = fitcph2dist(nphases=3, seed=SEED)
            setup(model)
            model.fit()
            model.plot(filename=PLOTFILE)
        quiet(plotone)
        if not os.path.exists(PLOTFILE) or os.path.getsize(PLOTFILE) == 0:
            sys.exit("Validation test failed at case 10: plot did not write a non-empty file for the %s target." % name)
        os.remove(PLOTFILE)

    def plotpercentile():
        model = fitcph2dist(nphases=3, seed=SEED)
        model.percentiles(cumprobs=np.copy(FINEPERCENTILES),
                          x=lognorm.ppf(FINEPERCENTILES, LOGNORMSIGMA,
                                        scale=np.exp(LOGNORMMU)))
        model.fit()
        model.plot(filename=PLOTFILE)
    quiet(plotpercentile)

    if not os.path.exists(PLOTFILE) or os.path.getsize(PLOTFILE) == 0:
        sys.exit("Validation test failed at case 10: plot did not write a non-empty file for the percentile target.")

    # the figures must be closed again, or fitting in a loop exhausts matplotlib
    if len(plt.get_fignums()) != opened:
        sys.exit("Validation test failed at case 10: plot left %d figures open." % (len(plt.get_fignums()) - opened))

    # The two targets the plot has to form a density for itself, rather than
    # reading it off scipy, are the truncated normal and the percentiles. A
    # curve that is simply absent still writes a perfectly good file, so these
    # are checked as densities rather than through the picture.
    def buildnorm():
        model = fitcph2dist(nphases=3, seed=SEED)
        model.norm(mu=NORMMU, sigma=NORMSIGMA)
        return model
    normtarget = quiet(buildnorm)
    normdensity = getattr(normtarget, "_fitcph2dist__normtruncdensity")

    mass = quad(normdensity, 0.0, NORMMU + 40.0 * NORMSIGMA, limit=400)[0]
    if abs(mass - 1.0) > 1e-8:
        sys.exit("Validation test failed at case 10: the truncated normal the plot draws integrates to %.10f rather than one." % mass)
    if normdensity(-1.0) != 0.0:
        sys.exit("Validation test failed at case 10: the truncated normal the plot draws is not zero below the truncation at zero.")

    def buildper():
        model = fitcph2dist(nphases=3, seed=SEED)
        model.percentiles(cumprobs=np.copy(PERCENTILEP),
                          x=lognorm.ppf(PERCENTILEP, LOGNORMSIGMA,
                                        scale=np.exp(LOGNORMMU)))
        return model
    pertarget = quiet(buildper)
    perdensity = getattr(pertarget, "_fitcph2dist__perdensity")
    pertop = float(lognorm.ppf(PERCENTILEP[-1], LOGNORMSIGMA,
                               scale=np.exp(LOGNORMMU)))

    # the percentiles carry no mass above the largest of them, so the density
    # the plot draws must integrate to exactly that probability and no further
    mass = quad(np.vectorize(perdensity), 0.0, pertop, limit=400)[0]
    if abs(mass - PERCENTILEP[-1]) > 1e-6:
        sys.exit("Validation test failed at case 10: the percentile density the plot draws integrates to %.8f rather than the %.8f its largest cumulative probability states." % (mass, PERCENTILEP[-1]))
    if perdensity(pertop * 1.5) != 0.0 or perdensity(-1.0) != 0.0:
        sys.exit("Validation test failed at case 10: the percentile density the plot draws is not zero outside the range the percentiles cover.")

    # Having the density is not the same as drawing it, and a figure missing a
    # whole curve still writes a perfectly good file. The figure is therefore
    # read back, which means holding off the close that plot does for itself.
    closefigure = plt.close
    plt.close = lambda *arguments: None
    try:
        for name, setup in (("normal", lambda m: m.norm(mu=NORMMU,
                                                        sigma=NORMSIGMA)),
                            ("percentile",
                             lambda m: m.percentiles(
                                 cumprobs=np.copy(PERCENTILEP),
                                 x=lognorm.ppf(PERCENTILEP, LOGNORMSIGMA,
                                               scale=np.exp(LOGNORMMU))))):
            def drawone():
                model = fitcph2dist(nphases=3, seed=SEED)
                setup(model)
                model.fit()
                model.plot(filename=PLOTFILE)
            quiet(drawone)

            drawn = plt.gcf().axes[0].lines
            if len(drawn) != 2:
                sys.exit("Validation test failed at case 10: the %s plot drew %d curves rather than the target and the approximation." % (name, len(drawn)))

            labels = [line.get_label() for line in drawn]
            if "True density" not in labels:
                sys.exit("Validation test failed at case 10: the %s plot has no curve labelled as the target." % name)

            target = np.asarray(drawn[labels.index("True density")].get_ydata())
            if not np.any(np.abs(target) > 0.0):
                sys.exit("Validation test failed at case 10: the %s plot drew its target curve flat at zero, so the target is not shown." % name)
            if not np.all(np.isfinite(target)):
                sys.exit("Validation test failed at case 10: the %s plot drew a target curve that is not finite throughout." % name)

            closefigure("all")
    finally:
        plt.close = closefigure
        plt.close("all")

    # ------------------------------------------------------------------
    # CASE 11
    # Infeasible input is refused rather than fitted.
    # ------------------------------------------------------------------

    REFUSED = (
        ("a number of phases below one", 'fitcph2dist(nphases=0)'),
        ("an unknown structure", 'fitcph2dist(nphases=2, dtype="nosuch")'),
        ("a structure that is not a string", 'fitcph2dist(nphases=2, dtype=7)'),
        ("a tolerance that is not a number", 'fitcph2dist(nphases=2, tolerance="x")'),
        ("a verbose flag that is not boolean", 'fitcph2dist(nphases=2, verbose=1)'),
        ("an accelerate flag that is not boolean", 'fitcph2dist(nphases=2, accelerate=1)'),
        ("a restart count below one", 'fitcph2dist(nphases=2, restarts=0)'),
        ("a restart count that is not an integer", 'fitcph2dist(nphases=2, restarts=2.5)'),
        ("a randominit flag that is not boolean", 'fitcph2dist(nphases=2, randominit=1)'),
        ("a structure with a phase that can never be left",
         'm = fitcph2dist(nphases=2, dtype="custom", randominit=True,'
         ' initdist=np.array([1.0, 1.0]), initphgen=np.zeros((2, 2)),'
         ' initexitrates=np.array([1.0, 0.0]));'
         ' m.lognorm(mean=2.0, var=1.0)'),
    )

    for description, expression in REFUSED:
        status, output = runsnippet(expression)
        if status == 0:
            sys.exit("Validation test failed at case 11: %s was accepted rather than refused." % description)
        if "Error" not in output:
            sys.exit("Validation test failed at case 11: %s was refused without saying why." % description)

    # every structure the class does offer must still be accepted, or the check
    # above would pass by refusing everything
    for structure in STRUCTURES:
        status, output = runsnippet('m = fitcph2dist(nphases=3, dtype="%s",'
                                    ' seed=1); m.lognorm(mean=2.0, var=1.0);'
                                    ' m.fit()' % structure)
        if status != 0:
            sys.exit("Validation test failed at case 11: the %s structure was refused." % structure)

    # ------------------------------------------------------------------
    # CASE 12
    # The E-step against formulas worked out by hand. A fit run to convergence
    # observes the E-step only through what it arrives at, which leaves room for
    # a component that is wrong and still converges to something plausible.
    #
    # With ONE phase the E-step collapses. Writing w_k for hy_k divided by the
    # density at y_k, the density is lambda exp(-lambda y) and the integral over
    # the time of the jump is lambda y exp(-lambda y), so
    #
    #     b   = sum_k w_k lambda exp(-lambda y_k)  = sum_k hy_k
    #     n   = sum_k w_k lambda exp(-lambda y_k)  = sum_k hy_k
    #     z   = sum_k w_k lambda y_k exp(-lambda y_k) = sum_k hy_k y_k
    #
    # none of which depends on lambda, so the expected values are exact and the
    # same for every rate. The M-step then returns sum(hy)/sum(hy y), the
    # maximum likelihood rate of an exponential for weighted observations.
    # ------------------------------------------------------------------

    ONEY = np.array([1.0, 3.0])
    ONEHY = np.array([0.5, 0.5])
    expectedb = float(np.sum(ONEHY))
    expectedz = float(np.sum(ONEHY * ONEY))

    for rate in (0.25, 0.5, 2.0):
        model = estepof(1, [1.0], [[-rate]], [rate], ONEY, ONEHY)
        if (abs(model.bi[0] - expectedb) > TOL
                or abs(model.ni[0] - expectedb) > TOL
                or abs(model.zi[0] - expectedz) > TOL
                or abs(model.nij[0, 0]) > TOL):
            sys.exit("Validation test failed at case 12: the one-phase E-step at rate %.2f gave b=%.12f n=%.12f z=%.12f, against the %.12f, %.12f and %.12f these must be for any rate." % (rate, model.bi[0], model.ni[0], model.zi[0], expectedb, expectedb, expectedz))

    # The M-step on those counts is exact: the rate is sum(hy)/sum(hy y), which
    # the grid above makes 1/2, and the single diagonal entry is minus it. This
    # is the one check in the file with no discretization between it and the
    # answer, so it resolves a shift in the parameters far below anything a
    # comparison of distributions can see: the method's own discretization error
    # is 3.8e-5, while this separates 1e-12.
    expectedrate = expectedb / expectedz

    if abs(expectedrate - 0.5) > TOL:
        sys.exit("Validation test failed at case 12: the one-phase grid no longer gives an exactly representable rate, so the check below is no longer exact.")

    for rate in (0.25, 2.0):
        model = estepof(1, [1.0], [[-rate]], [rate], ONEY, ONEHY, maximize=True)
        if abs(model.pi[0] - 1.0) > TOL:
            sys.exit("Validation test failed at case 12: the one-phase M-step left the initial probability at %.12f rather than one." % model.pi[0])
        if abs(model.exitrates[0] - expectedrate) > TOL:
            sys.exit("Validation test failed at case 12: the one-phase M-step at rate %.2f returned the rate %.12f, against the %.12f that sum(hy)/sum(hy y) must give for any rate." % (rate, model.exitrates[0], expectedrate))
        if abs(model.phgen[0, 0] + expectedrate) > TOL:
            sys.exit("Validation test failed at case 12: the one-phase M-step returned the generator entry %.12f, which is not minus its exit rate." % model.phgen[0, 0])

    # Two phases with DISTINCT rates, against numerical quadrature of the
    # defining integral. The quadrature shares no algebra with the Van Loan
    # block identity the class evaluates, so agreement is a statement about the
    # identity and not about the implementation agreeing with itself.
    TWOPI = np.array([0.75, 0.25])
    TWOGEN = np.array([[-2.0, 0.5], [1.0, -3.0]])
    TWOEXIT = -TWOGEN.sum(axis=1)
    TWOY = np.array([0.5, 1.5, 2.5])
    TWOHY = np.array([0.2, 0.5, 0.3])

    if abs(TWOGEN[0, 0] - TWOGEN[1, 1]) < TOL or abs(TWOPI[0] - TWOPI[1]) < TOL:
        sys.exit("Validation test failed at case 12: the two-phase inputs are exchangeable, so an index swapped between the phases would not change the answer.")

    model = estepof(2, TWOPI, TWOGEN, TWOEXIT, TWOY, TWOHY)

    density = np.array([float(TWOPI @ expm(TWOGEN * y) @ TWOEXIT)
                        for y in TWOY])
    weights = TWOHY / density

    expectedbi = TWOPI * np.array(
        [sum(weights[k] * (expm(TWOGEN * TWOY[k]) @ TWOEXIT)[i]
             for k in range(TWOY.size)) for i in range(2)])
    expectedni = TWOEXIT * np.array(
        [sum(weights[k] * (TWOPI @ expm(TWOGEN * TWOY[k]))[i]
             for k in range(TWOY.size)) for i in range(2)])

    integral = np.zeros((2, 2))
    for k in range(TWOY.size):
        for i in range(2):
            for j in range(2):
                def integrand(u, i=i, j=j, k=k):
                    return ((TWOPI @ expm(TWOGEN * u))[i]
                            * (expm(TWOGEN * (TWOY[k] - u)) @ TWOEXIT)[j])
                integral[i, j] += weights[k] * quad(integrand, 0.0, TWOY[k],
                                                    limit=200)[0]

    expectedzi = np.diag(integral).copy()
    expectednij = TWOGEN * integral
    np.fill_diagonal(expectednij, 0.0)

    for name, got, expected in (("bi", model.bi, expectedbi),
                                ("ni", model.ni, expectedni),
                                ("zi", model.zi, expectedzi),
                                ("nij", model.nij, expectednij)):
        if np.max(np.abs(got - expected)) > TOLQUAD:
            sys.exit("Validation test failed at case 12: the two-phase E-step gave %s differing from quadrature of the defining integral by %.3e." % (name, np.max(np.abs(got - expected))))

    if abs(model.loglikelihood
           - float(np.sum(TWOHY * np.log(density)))) > TOLQUAD:
        sys.exit("Validation test failed at case 12: the log-likelihood is not the sum of hy times the log of the fitted density over the cells.")

    # and the M-step on those same counts, with the phases carrying different
    # values throughout, so a index taken from the wrong phase changes the answer
    maximized = estepof(2, TWOPI, TWOGEN, TWOEXIT, TWOY, TWOHY, maximize=True)

    expectedpi = expectedbi / np.sum(expectedbi)
    expectedrates = expectedni / expectedzi
    expectedgen = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            if j != i:
                expectedgen[i, j] = expectednij[i, j] / expectedzi[i]
        expectedgen[i, i] = -(np.sum(expectedgen[i]) + expectedrates[i])

    if float(np.min(np.abs(np.diff(expectedrates)))) < TOLQUAD:
        sys.exit("Validation test failed at case 12: the two-phase M-step returns the same rate for both phases, so an index taken from the wrong one would not change the answer.")

    if np.max(np.abs(maximized.pi - expectedpi)) > TOLQUAD:
        sys.exit("Validation test failed at case 12: the two-phase M-step returned the initial distribution %s, against the %s the counts give." % (maximized.pi, expectedpi))
    if np.max(np.abs(maximized.exitrates - expectedrates)) > TOLQUAD:
        sys.exit("Validation test failed at case 12: the two-phase M-step returned the exit rates %s, against the %s that ni/zi gives." % (maximized.exitrates, expectedrates))
    if np.max(np.abs(maximized.phgen - expectedgen)) > TOLQUAD:
        sys.exit("Validation test failed at case 12: the two-phase M-step returned a generator differing from nij/zi by %.3e." % np.max(np.abs(maximized.phgen - expectedgen)))

finally:
    if os.path.exists(PLOTFILE):
        os.remove(PLOTFILE)


print("fitcph2dist: All tests passed.")
