'''
UNIT TEST FOR THE mcph CLASS (M-step of the EM algorithm for continuous-time
phase-type distributions), run in sequence with the ecph class (E-step).

The test is built around an Erlang distribution. The sum of NPHASES
exponentially distributed variables with the same rate is the time until
absorption of a continuous phase-type distribution with NPHASES phases in
series, where every phase has the same exit rate, so that

    T[i,i] = -RATE,  T[i,i+1] = RATE,  t[p-1] = RATE,  pi = (1,0,...,0),

and the resulting CPH distribution is exactly Erlang with NPHASES phases and
rate RATE. This gives the test a known ground truth to compare against.

The fitted parameters themselves are deliberately not used as the acceptance
criterion. A CPH representation is heavily overparametrized (Theorem 3.1.22,
p. 138), so several different generators describe the same distribution and the
EM algorithm may drift along that set without the fit getting any worse. The
distribution it implies is what is identified, so the test compares the
log-likelihood and the density at a few points instead.

Sub-tests:
    Case 1: Uncensored data only. Averaged over replicate sample sets, the
            log-likelihood and the density are closer to the truth after 100
            steps than after 2 steps.
    Case 2: The same, for a mix of uncensored, interval-, right- and
            left-censored observations.
    Case 3: The fitted parameters are a feasible CPH representation.

References:
    Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in
    Applied Probability. Springer. https://doi.org/10.1007/978-1-4939-7049-0
'''

import os
import sys
import numpy as np
from scipy.linalg import expm

# Load phasedist from the src-folder so the test can be run without installing
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from phasedist.ecph import ecph
from phasedist.mcph import mcph


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

TOL = 1e-9  # tolerance for comparisons that are exact in theory
SEED = 0

NPHASES = 3
RATE = 0.5  # rate of each of the exponential variables in the Erlang sum

NOBS = 300        # size of each sample set
NREPLICATES = 5   # number of replicate sample sets averaged over
STEPSFEW = 2      # a few EM steps
STEPSMANY = 100   # many EM steps

# Points at which the fitted density is compared to the true one
POINTS = (2.0, 6.0, 12.0)

# Censoring limits for case 2. They are fixed constants rather than functions
# of the observation, so that the censoring carries no information about the
# value it hides. Limits derived from the observation itself would make the
# censoring informative, the censored likelihood would then be the wrong model
# for the data, and the fit would converge to a biased distribution.
RIGHTLIMIT = 6.0
LEFTLIMIT = 6.0
BINWIDTH = 2.0

# True parameters: NPHASES phases in series, each left at rate RATE,
# absorption only from the last phase.
INITDIST = np.zeros(NPHASES)
INITDIST[0] = 1.0
PHGEN = np.zeros((NPHASES, NPHASES))
EXITRATES = np.zeros(NPHASES)
for i in range(NPHASES):
    PHGEN[i, i] = -RATE
    if i < NPHASES - 1:
        PHGEN[i, i + 1] = RATE
EXITRATES[NPHASES - 1] = RATE

# Verify that the true parameters are feasible before they are used
if abs(np.sum(INITDIST) - 1.0) > TOL or np.any(INITDIST < 0.0):
    sys.exit("Validation test failed at initialization: the initial distribution is infeasible.")

if np.any(PHGEN[~np.eye(NPHASES, dtype=bool)] < 0.0) or np.any(EXITRATES < 0.0):
    sys.exit("Validation test failed at initialization: negative off-diagonal transition rates or exit rates.")

if np.any(np.diag(PHGEN) >= 0.0):
    sys.exit("Validation test failed at initialization: the phase-type generator has non-negative diagonal values.")

if np.any(np.abs(np.sum(PHGEN, axis=1) + EXITRATES) > TOL):
    sys.exit("Validation test failed at initialization: the rows of the phase-type generator and the exit rates do not sum to zero.")


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def density(y, initdist, phgen, exitrates):
    '''
    Returns the CPH density at y.
    '''
    return float(np.matmul(initdist, np.matmul(expm(phgen * y), exitrates)))


def randomstart():
    '''
    Draws a random, feasible set of CPH parameters to start the EM algorithm
    from. Every transition rate is strictly positive, so no rate is held at
    zero by the structural zeros of the EM algorithm (p. 678).
    '''
    initdist = np.random.uniform(0.1, 1.0, size=NPHASES)
    initdist = initdist / np.sum(initdist)

    phgen = np.random.uniform(0.1, 1.0, size=(NPHASES, NPHASES))
    np.fill_diagonal(phgen, 0.0)
    exitrates = np.random.uniform(0.1, 1.0, size=NPHASES)
    np.fill_diagonal(phgen, -(np.sum(phgen, axis=1) + exitrates))

    return initdist, phgen, exitrates


def censoringlimits(y, kind):
    '''
    Returns the pair of censoring limits for an observation y of the given
    kind, using limits that do not depend on y.
    '''
    if kind == 1:
        lower = BINWIDTH * np.floor(y / BINWIDTH)
        return lower, lower + BINWIDTH
    if kind == 2 and y > RIGHTLIMIT:
        return np.nan, RIGHTLIMIT
    if kind == 3 and y <= LEFTLIMIT:
        return LEFTLIMIT, np.nan
    return np.nan, np.nan


def mixedcensoring(obs):
    '''
    Returns a censoring array mixing uncensored, interval-, right- and
    left-censored observations.
    '''
    censoring = np.full((obs.size, 2), np.nan)
    for k in range(obs.size):
        censoring[k, 0], censoring[k, 1] = censoringlimits(obs[k], k % 4)
    return censoring


def emsteps(obs, initdist, phgen, exitrates, nsteps, censoring=None):
    '''
    Runs nsteps sequences of the E-step (ecph) followed by the M-step (mcph),
    and returns the fitted parameters together with the log-likelihood
    evaluated at them.
    '''
    est = ecph(nphases=NPHASES)
    mst = mcph(nphases=NPHASES, nobs=obs.size)

    for _ in range(nsteps):
        bi, zi, ni, nij = est.run(obs=obs,
                                  initdist=initdist,
                                  phgen=phgen,
                                  exitrates=exitrates,
                                  censoring=censoring)
        initdist, phgen, exitrates = mst.run(bi=bi, zi=zi, ni=ni, nij=nij)

    est.run(obs=obs,
            initdist=initdist,
            phgen=phgen,
            exitrates=exitrates,
            censoring=censoring)

    return initdist, phgen, exitrates, float(est.loglikelihood)


def densityerror(initdist, phgen, exitrates):
    '''
    Returns the mean absolute deviation between the fitted density and the true
    density over the evaluation points.
    '''
    return float(np.mean([
        abs(density(y, initdist, phgen, exitrates)
            - density(y, INITDIST, PHGEN, EXITRATES))
        for y in POINTS
    ]))


def isfeasible(initdist, phgen, exitrates):
    '''
    Returns True if the parameters are a feasible CPH representation.
    '''
    if not (np.all(np.isfinite(initdist))
            and np.all(np.isfinite(phgen))
            and np.all(np.isfinite(exitrates))):
        return False
    if np.any(initdist < 0.0) or abs(np.sum(initdist) - 1.0) > TOL:
        return False
    if np.any(phgen[~np.eye(NPHASES, dtype=bool)] < 0.0):
        return False
    if np.any(np.diag(phgen) >= 0.0):
        return False
    if np.any(exitrates < 0.0):
        return False
    if np.any(np.abs(np.sum(phgen, axis=1) + exitrates) > TOL):
        return False
    return True


def averagefit(censored):
    '''
    Fits every replicate sample set with a few and with many EM steps, and
    returns the mean log-likelihood and the mean density error for each of the
    two step counts, together with all fitted parameters.
    '''
    loglikfew = loglikmany = 0.0
    errorfew = errormany = 0.0
    fitted = []

    for rep in range(NREPLICATES):
        obs = SAMPLES[rep]
        initdist, phgen, exitrates = STARTS[rep]
        censoring = mixedcensoring(obs) if censored else None

        few = emsteps(obs, initdist, phgen, exitrates, STEPSFEW, censoring)
        many = emsteps(obs, initdist, phgen, exitrates, STEPSMANY, censoring)

        loglikfew += few[3]
        loglikmany += many[3]
        errorfew += densityerror(few[0], few[1], few[2])
        errormany += densityerror(many[0], many[1], many[2])
        fitted.append(few[:3])
        fitted.append(many[:3])

    return (loglikfew / NREPLICATES, loglikmany / NREPLICATES,
            errorfew / NREPLICATES, errormany / NREPLICATES, fitted)


# Replicate sample sets of Erlang observations, each simulated by adding
# NPHASES exponentially distributed samples, and a random set of starting
# parameters for each of them.
np.random.seed(SEED)
SAMPLES = [np.sum([np.random.exponential(1.0 / RATE, size=NOBS)
                   for _ in range(NPHASES)], axis=0)
           for _ in range(NREPLICATES)]
STARTS = [randomstart() for _ in range(NREPLICATES)]


# ------------------------------------------------------------------
# CASE 1: Uncensored data, more EM steps give a better fit
# ------------------------------------------------------------------

# A single sample set only favours the longer run with high probability, not
# with certainty, so the comparison is made on the average over NREPLICATES
# replicate sample sets.
loglikfew, loglikmany, errorfew, errormany, fitteduncensored = averagefit(False)

if not loglikmany > loglikfew:
    sys.exit("Validation test failed at case 1: the log-likelihood is not higher after %d steps than after %d steps." % (STEPSMANY, STEPSFEW))

if not errormany < errorfew:
    sys.exit("Validation test failed at case 1: the fitted density is not closer to the true density after %d steps than after %d steps." % (STEPSMANY, STEPSFEW))


# ------------------------------------------------------------------
# CASE 2: Mixed censored data, more EM steps give a better fit
# ------------------------------------------------------------------

loglikfew, loglikmany, errorfew, errormany, fittedcensored = averagefit(True)

if not loglikmany > loglikfew:
    sys.exit("Validation test failed at case 2: the log-likelihood is not higher after %d steps than after %d steps for censored data." % (STEPSMANY, STEPSFEW))

if not errormany < errorfew:
    sys.exit("Validation test failed at case 2: the fitted density is not closer to the true density after %d steps than after %d steps for censored data." % (STEPSMANY, STEPSFEW))

# All four kinds of observation must actually occur in the censored data,
# otherwise the case would silently test less than it claims to
censoring = mixedcensoring(SAMPLES[0])
lower = censoring[:, 0]
upper = censoring[:, 1]
counts = (np.sum(np.isnan(lower) & np.isnan(upper)),
          np.sum(~np.isnan(lower) & ~np.isnan(upper)),
          np.sum(np.isnan(lower) & ~np.isnan(upper)),
          np.sum(~np.isnan(lower) & np.isnan(upper)))

if min(counts) == 0:
    sys.exit("Validation test failed at case 2: the censored data set does not contain all four kinds of observation.")


# ------------------------------------------------------------------
# CASE 3: The fitted parameters are a feasible CPH representation
# ------------------------------------------------------------------

for initdist, phgen, exitrates in fitteduncensored:
    if not isfeasible(initdist, phgen, exitrates):
        sys.exit("Validation test failed at case 3: fitting uncensored data returned parameters that are not a feasible continuous phase-type distribution.")

for initdist, phgen, exitrates in fittedcensored:
    if not isfeasible(initdist, phgen, exitrates):
        sys.exit("Validation test failed at case 3: fitting censored data returned parameters that are not a feasible continuous phase-type distribution.")


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print("mcph: All tests passed.")
