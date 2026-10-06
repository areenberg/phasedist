'''
UNIT TEST FOR THE ecph CLASS (E-step of the EM algorithm for continuous-time
phase-type distributions).

The test is built around a CPH distribution whose exit rate is the same in
every phase. Such a CPH is exactly an exponential distribution: if t_i = lambda
for all i, then the rows of T sum to -lambda, hence T*e = -lambda*e,
exp(Ty)*e = exp(-lambda*y)*e and

    f(y) = pi exp(Ty) t = lambda exp(-lambda*y) pi e = lambda exp(-lambda*y),

which is the exponential density. This gives the test a known ground truth to
compare the estimates against.

Sub-tests:
    Case 1: E-step (ecph) followed by the M-step (mcph) on uncensored data
            recovers the true parameters better from 300 observations than
            from 30 observations.
    Case 2: Left censoring equals interval censoring with lower limit 0.
    Case 3: Right censoring equals interval censoring with a large upper limit.
    Case 4: Interval censoring approaches the uncensored case when the two
            limits are very close to the observation.
    Case 5: The input checks refuse infeasible input, accept feasible input,
            and are not fooled by the caching of the sample checks.

References:
    Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in
    Applied Probability. Springer. https://doi.org/10.1007/978-1-4939-7049-0
'''

import os
import sys
import numpy as np

# Load phasedist from the src-folder so the test can be run without installing
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from phasedist.ecph import ecph
from phasedist.mcph import mcph


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

TOL = 1e-10       # tolerance for comparisons that are exact in theory
TOLAPPROX = 1e-6  # tolerance for case 4, which holds in the limit only
SEED = 0

NPHASES = 3
RATE = 0.5  # parameter of the exponential distribution

NSMALL = 30       # size of the small sample set
NLARGE = 300      # size of the large sample set
NREPLICATES = 20  # number of replicate sample sets averaged over in case 1

# Feasible CPH parameters. Every row of the phase-type generator sums to -RATE,
# and every exit rate equals RATE, so the absorption time is exponentially
# distributed with parameter RATE.
INITDIST = np.array([0.5, 0.3, 0.2])
PHGEN = np.array([[-1.2, 0.4, 0.3],
                  [0.2, -0.9, 0.2],
                  [0.5, 0.3, -1.3]])
EXITRATES = np.array([RATE, RATE, RATE])

# Verify that the parameters are feasible before they are used
if abs(np.sum(INITDIST) - 1.0) > TOL or np.any(INITDIST < 0.0):
    sys.exit("Validation test failed at initialization: the initial distribution is infeasible.")

if np.any(PHGEN[~np.eye(NPHASES, dtype=bool)] < 0.0) or np.any(EXITRATES < 0.0):
    sys.exit("Validation test failed at initialization: negative off-diagonal transition rates or exit rates.")

if np.any(np.diag(PHGEN) >= 0.0):
    sys.exit("Validation test failed at initialization: the phase-type generator has non-negative diagonal values.")

if np.any(np.abs(np.sum(PHGEN, axis=1) + EXITRATES) > TOL):
    sys.exit("Validation test failed at initialization: the rows of the phase-type generator and the exit rates do not sum to zero.")

# Sample sets of uncensored observations from the same exponential
# distribution. Case 1 averages over NREPLICATES pairs of sample sets, so that
# the comparison of the small and the large sample set does not depend on the
# particular seed. Cases 2-4 use the first small sample set.
np.random.seed(SEED)
SAMPLESSMALL = [np.random.exponential(1.0 / RATE, size=NSMALL) for _ in range(NREPLICATES)]
SAMPLESLARGE = [np.random.exponential(1.0 / RATE, size=NLARGE) for _ in range(NREPLICATES)]

OBSSMALL = SAMPLESSMALL[0]
OBSLARGE = SAMPLESLARGE[0]


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def runestep(obs, censoring=None):
    '''
    Runs the E-step and returns copies of the sufficient statistics and the
    log-likelihood.
    '''
    est = ecph(nphases=NPHASES)
    bi, zi, ni, nij = est.run(obs=obs,
                              initdist=INITDIST,
                              phgen=PHGEN,
                              exitrates=EXITRATES,
                              censoring=censoring)
    return bi.copy(), zi.copy(), ni.copy(), nij.copy(), float(est.loglikelihood)


def runemstep(obs):
    '''
    Runs the E-step followed by the M-step on uncensored observations and
    returns the updated parameters.
    '''
    bi, zi, ni, nij = runestep(obs)[:4]
    mst = mcph(nphases=NPHASES, nobs=obs.size)
    return mst.run(bi=bi, zi=zi, ni=ni, nij=nij)


def statsdiff(first, second):
    '''
    Returns the largest absolute difference between the sufficient statistics
    of two E-step results, disregarding the log-likelihood.
    '''
    return max(np.max(np.abs(first[0] - second[0])),
               np.max(np.abs(first[1] - second[1])),
               np.max(np.abs(first[2] - second[2])),
               np.max(np.abs(first[3] - second[3])))


def maxdiff(first, second):
    '''
    Returns the largest absolute difference between two E-step results,
    including the log-likelihood.
    '''
    return max(statsdiff(first, second), abs(first[4] - second[4]))


def l1dist(estimate, truth):
    '''
    Returns the sum of absolute deviations between an estimate and the truth.
    '''
    return float(np.sum(np.abs(np.asarray(estimate) - np.asarray(truth))))


# ------------------------------------------------------------------
# CASE 1: More observations give estimates closer to the true parameters
# ------------------------------------------------------------------

# The updated parameters are random, so a single pair of sample sets only
# favours the large sample set with high probability, not with certainty. The
# distances are therefore averaged over NREPLICATES replicate pairs of sample
# sets, which makes the comparison hold essentially always.
phgensmall = 0.0
phgenlarge = 0.0
exitsmall = 0.0
exitlarge = 0.0
initdistdev = 0.0

for rep in range(NREPLICATES):

    initdistupd, phgenupd, exitratesupd = runemstep(SAMPLESSMALL[rep])
    phgensmall += l1dist(phgenupd, PHGEN)
    exitsmall += l1dist(exitratesupd, EXITRATES)
    initdistdev = max(initdistdev, l1dist(initdistupd, INITDIST))

    initdistupd, phgenupd, exitratesupd = runemstep(SAMPLESLARGE[rep])
    phgenlarge += l1dist(phgenupd, PHGEN)
    exitlarge += l1dist(exitratesupd, EXITRATES)
    initdistdev = max(initdistdev, l1dist(initdistupd, INITDIST))

phgensmall /= NREPLICATES
phgenlarge /= NREPLICATES
exitsmall /= NREPLICATES
exitlarge /= NREPLICATES

# The phase-type generator and the exit rates must on average be closer to the
# true parameters when the large sample set is used
if not phgenlarge < phgensmall:
    sys.exit("Validation test failed at case 1: the estimated phase-type generator is not closer to the true generator for the large sample set.")

if not exitlarge < exitsmall:
    sys.exit("Validation test failed at case 1: the estimated exit rates are not closer to the true rates for the large sample set.")

# The initial distribution is recovered exactly from every sample set,
# irrespective of its size, so it is checked against the true parameter rather
# than compared between the two sample sizes. Because all exit rates are equal
# we have exp(Ty)*t = RATE*exp(-RATE*y)*e, and hence
#
#     E(B_i|Y=y) = pi_i (exp(Ty) t)_i / (pi exp(Ty) t) = pi_i
#
# for every y. The expected number of processes initiating in state i is
# therefore pi_i for every observation, and the M-step returns pi regardless of
# the observations.
if initdistdev > TOL:
    sys.exit("Validation test failed at case 1: the initial distribution is not recovered exactly.")


# ------------------------------------------------------------------
# CASE 2: Left censoring equals interval censoring with lower limit zero
# ------------------------------------------------------------------

limits = OBSSMALL

# Left-censored observations, i.e. it is known that Y <= limit
censleft = np.column_stack([limits, np.full(NSMALL, np.nan)])

# The same observations as interval-censored on (0,limit]
censinterval = np.column_stack([np.zeros(NSMALL), limits])

if maxdiff(runestep(OBSSMALL, censleft), runestep(OBSSMALL, censinterval)) > TOL:
    sys.exit("Validation test failed at case 2: left censoring does not equal interval censoring with lower limit zero.")


# ------------------------------------------------------------------
# CASE 3: Right censoring equals interval censoring with a large upper limit
# ------------------------------------------------------------------

UPPERLIMIT = 500.0  # large enough that P(Y>UPPERLIMIT) is numerically zero

# Right-censored observations, i.e. it is known that Y > limit
censright = np.column_stack([np.full(NSMALL, np.nan), limits])

# The same observations as interval-censored on (limit,UPPERLIMIT]
censinterval = np.column_stack([limits, np.full(NSMALL, UPPERLIMIT)])

resright = runestep(OBSSMALL, censright)
resinterval = runestep(OBSSMALL, censinterval)

# Both describe the event Y > limit, so the likelihood contribution and the
# expected number of processes initiating in state i must be identical. The
# statistics z_i, n_i and n_ij are deliberately not compared: a right-censored
# process is only observed on [0,limit], where no absorption has been seen, so
# its time spent in each state and its jump counts cover that window alone,
# whereas an interval-censored process is known to have been absorbed before
# UPPERLIMIT, so its statistics cover the entire lifetime [0,Y]. Both are
# correct for their own observation window, but they are not the same quantity.
if abs(resright[4] - resinterval[4]) > TOL:
    sys.exit("Validation test failed at case 3: right censoring and interval censoring with a large upper limit do not give the same log-likelihood.")

if np.max(np.abs(resright[0] - resinterval[0])) > TOL:
    sys.exit("Validation test failed at case 3: right censoring and interval censoring with a large upper limit do not give the same initiation counts.")

# The time spent in each state is instead checked against the limits directly.
# A right-censored process is known to be in some transient state throughout
# [0,limit], so the time it spends across all transient states in that window
# is exactly the limit itself, whatever the parameters are.
if abs(np.sum(resright[1]) - np.sum(limits)) > TOL:
    sys.exit("Validation test failed at case 3: the time spent in the transient states does not equal the censoring limits for right-censored observations.")

# The jump counts are checked by recombining the two censoring types at one
# shared limit. Weighting the left-censored contribution by P(Y<=limit) and the
# right-censored contribution by P(Y>limit) removes the conditioning, the
# K-matrix terms cancel, and what remains is the elementary identity that the
# expected number of jumps from i to j over the observation window equals the
# transition rate times the expected time spent in state i over that window.
# P(Y>limit) is known in closed form because the CPH equals the exponential
# distribution.
SHAREDLIMIT = 1.7
survival = np.exp(-RATE * SHAREDLIMIT)

resleftshared = runestep(
    OBSSMALL, np.column_stack([np.full(NSMALL, SHAREDLIMIT), np.full(NSMALL, np.nan)])
)
resrightshared = runestep(
    OBSSMALL, np.column_stack([np.full(NSMALL, np.nan), np.full(NSMALL, SHAREDLIMIT)])
)

zicombined = resleftshared[1] * (1.0 - survival) + resrightshared[1] * survival
nijcombined = resleftshared[3] * (1.0 - survival) + resrightshared[3] * survival
offdiagonal = ~np.eye(NPHASES, dtype=bool)

if np.max(np.abs(nijcombined - PHGEN * zicombined[:, None])[offdiagonal]) > TOL:
    sys.exit("Validation test failed at case 3: the recombined jump counts do not equal the transition rates times the recombined time spent in each state.")


# ------------------------------------------------------------------
# CASE 4: Interval censoring on (y-eps,y+eps] approaches the uncensored
#         observation y
# ------------------------------------------------------------------

# In continuous time the event Y=y has probability zero, so unlike the discrete
# case this is a limit rather than an exact identity: the interval is shrunk
# around the observation instead of bracketing it exactly. EPSILON balances the
# truncation error, which is of order EPSILON^2 because the interval is centred
# on the observation, against the floating-point cancellation in
# P(Y>y-eps)-P(Y>y+eps), which grows as EPSILON becomes small.
EPSILON = 1e-5

censinterval = np.column_stack([OBSSMALL - EPSILON, OBSSMALL + EPSILON])

resuncensored = runestep(OBSSMALL)
resinterval = runestep(OBSSMALL, censinterval)

if statsdiff(resuncensored, resinterval) > TOLAPPROX:
    sys.exit("Validation test failed at case 4: interval censoring on (y-eps,y+eps] does not approach the uncensored observation y.")

# The log-likelihoods are not directly comparable. An uncensored observation
# contributes the density f(y), whereas an interval-censored observation
# contributes the probability P(Y in (y-eps,y+eps]) = 2*eps*f(y) + O(eps^3), so
# the two log-likelihoods differ by NSMALL*log(2*eps) by construction. The
# offset is removed before the comparison.
offset = NSMALL * np.log(2.0 * EPSILON)

if abs(resuncensored[4] - (resinterval[4] - offset)) > TOLAPPROX:
    sys.exit("Validation test failed at case 4: interval censoring on (y-eps,y+eps] does not approach the uncensored log-likelihood.")


# ------------------------------------------------------------------
# CASE 5: The input checks
# ------------------------------------------------------------------

def refuses(**overrides):
    '''
    Returns True if the E-step refuses the feasible arguments with the given
    replacements applied.
    '''
    arguments = {"obs": OBSSMALL, "initdist": INITDIST, "phgen": PHGEN,
                 "exitrates": EXITRATES, "censoring": None}
    arguments.update(overrides)

    try:
        ecph(nphases=NPHASES).run(**arguments)
    except ValueError:
        return True

    return False


def refusalmessage(**overrides):
    '''
    Returns the message the E-step refuses the arguments with, or None.
    '''
    arguments = {"obs": OBSSMALL, "initdist": INITDIST, "phgen": PHGEN,
                 "exitrates": EXITRATES, "censoring": None}
    arguments.update(overrides)

    try:
        ecph(nphases=NPHASES).run(**arguments)
    except ValueError as error:
        return str(error)

    return None


# An empty first row: the diagonal is zero while the row sum and the
# off-diagonal elements stay feasible, which isolates the condition on the
# diagonal from the other two
EMPTYROWGEN = np.copy(PHGEN)
EMPTYROWGEN[0, :] = 0.0

# only the sign of the off-diagonal is at fault here
NEGATIVERATEGEN = np.copy(PHGEN)
NEGATIVERATEGEN[0, 1] = -PHGEN[0, 1]

# and only the row sum here
POSITIVEROWGEN = np.copy(PHGEN)
POSITIVEROWGEN[0, 1] = 2.0

# feasible except for one negative limit
NEGATIVELIMIT = np.full((NSMALL, 2), np.nan)
NEGATIVELIMIT[0, 1] = -1.0

INFEASIBLE = (
    ("no observations", {"obs": np.array([])}),
    ("two-dimensional observations", {"obs": np.copy(OBSSMALL).reshape(-1, 1)}),
    ("an observation at zero", {"obs": np.append(OBSSMALL, 0.0)}),
    ("a negative observation", {"obs": np.append(OBSSMALL, -1.0)}),
    ("an infinite observation", {"obs": np.append(OBSSMALL, np.inf)}),
    ("a missing observation", {"obs": np.append(OBSSMALL, np.nan)}),
    ("a negative initial probability", {"initdist": np.array([1.2, -0.2, 0.0])}),
    ("an initial distribution summing above one", {"initdist": np.array([0.5, 0.3, 0.3])}),
    ("a missing initial probability", {"initdist": np.array([0.5, 0.3, np.nan])}),
    ("a generator with a zero diagonal element", {"phgen": EMPTYROWGEN}),
    ("a generator with a negative transition rate", {"phgen": NEGATIVERATEGEN}),
    ("a generator with a row summing above zero", {"phgen": POSITIVEROWGEN}),
    ("a negative exit rate", {"exitrates": np.array([RATE, -RATE, RATE])}),
    ("an infinite exit rate", {"exitrates": np.array([RATE, np.inf, RATE])}),
    ("censoring with one column", {"censoring": np.full((NSMALL, 1), np.nan)}),
    ("censoring with too few rows", {"censoring": np.full((NSMALL - 1, 2), np.nan)}),
    ("a negative censoring limit", {"censoring": NEGATIVELIMIT}),
)

for description, override in INFEASIBLE:
    if not refuses(**override):
        sys.exit("Validation test failed at case 5: the E-step accepted %s." % description)

# A mismatched dimension is caught further down in any case, by NumPy, when
# the arrays meet in a matrix product. The check earns its place by naming the
# cause, so the message is what is asserted here.
MISMATCHED = (
    ("an initial distribution of the wrong length", {"initdist": INITDIST[:2]}),
    ("a generator of the wrong shape", {"phgen": PHGEN[:2, :2]}),
    ("exit rates of the wrong length", {"exitrates": EXITRATES[:2]}),
)

for description, override in MISMATCHED:
    message = refusalmessage(**override)

    if message is None or "dimension %d" % NPHASES not in message:
        sys.exit("Validation test failed at case 5: given %s the E-step reported '%s', which does not name the dimension it expected." % (description, message))

# the checks are only worth anything if feasible input still passes
if refuses():
    sys.exit("Validation test failed at case 5: the E-step refused feasible uncensored input.")

if refuses(censoring=np.full((NSMALL, 2), np.nan)):
    sys.exit("Validation test failed at case 5: the E-step refused feasible censored input.")

# The number of phases is checked when the class is built
for badphases in (0, -1, 2.5, "3"):
    try:
        ecph(nphases=badphases)
        sys.exit("Validation test failed at case 5: the E-step was built with %s phases." % repr(badphases))
    except ValueError:
        pass

# The sample checks are made once per array, so a later call carrying a
# different and infeasible sample has to be checked afresh
model = ecph(nphases=NPHASES)
model.run(obs=OBSSMALL, initdist=INITDIST, phgen=PHGEN, exitrates=EXITRATES)

try:
    model.run(obs=np.append(OBSSMALL, -1.0), initdist=INITDIST, phgen=PHGEN,
              exitrates=EXITRATES)
    sys.exit("Validation test failed at case 5: a second call with a negative observation was accepted, so the caching of the checks over the sample is not keyed on the array.")
except ValueError:
    pass

# and the same sample offered again has to still be accepted
model.run(obs=OBSSMALL, initdist=INITDIST, phgen=PHGEN, exitrates=EXITRATES)

# a parameter, by contrast, is checked at every call
try:
    model.run(obs=OBSSMALL, initdist=np.array([0.5, 0.3, 0.3]), phgen=PHGEN,
              exitrates=EXITRATES)
    sys.exit("Validation test failed at case 5: an infeasible initial distribution was accepted on a later call, so the parameters are not checked at every call.")
except ValueError:
    pass


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print("ecph: All tests passed.")
