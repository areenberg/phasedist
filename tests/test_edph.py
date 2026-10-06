'''
UNIT TEST FOR THE edph CLASS (E-step of the EM algorithm for discrete-time
phase-type distributions).

The test is built around a DPH distribution whose exit probability is the same
in every phase. Such a DPH is exactly a geometric distribution: if t_i = p for
all i, then the row sums of T are all 1-p, hence T*e = (1-p)*e and

    P(Y=y) = pi T^(y-1) t = p (1-p)^(y-1),

which is the geometric probability mass function. This gives the test a known
ground truth to compare the estimates against.

Sub-tests:
    Case 1: E-step (edph) followed by the M-step (mdph) on uncensored data
            recovers the true parameters better from 300 observations than
            from 30 observations.
    Case 2: Left censoring equals interval censoring with lower limit 0.
    Case 3: Right censoring equals interval censoring with a large upper limit.
    Case 4: Interval censoring equals the uncensored case when the interval
            brackets exactly one integer, i.e. (y-1,y] = {Y=y}.
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

from phasedist.edph import edph
from phasedist.mdph import mdph


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

TOL = 1e-10  # tolerance for comparisons that are exact in theory
SEED = 0

NPHASES = 3
PROB = 0.3  # parameter of the geometric distribution

NSMALL = 30       # size of the small sample set
NLARGE = 300      # size of the large sample set
NREPLICATES = 20  # number of replicate sample sets averaged over in case 1

# Feasible DPH parameters. Every row of the sub-transition matrix sums to
# 1-PROB, and every exit probability equals PROB, so the absorption time is
# geometrically distributed with parameter PROB.
INITDIST = np.array([0.5, 0.3, 0.2])
PHGEN = np.array([[0.40, 0.20, 0.10],
                  [0.10, 0.50, 0.10],
                  [0.20, 0.20, 0.30]])
EXITRATES = np.array([PROB, PROB, PROB])

# Verify that the parameters are feasible before they are used
if abs(np.sum(INITDIST) - 1.0) > TOL or np.any(INITDIST < 0.0):
    sys.exit("Validation test failed at initialization: the initial distribution is infeasible.")

if np.any(PHGEN < 0.0) or np.any(EXITRATES < 0.0):
    sys.exit("Validation test failed at initialization: negative transition or exit probabilities.")

if np.any(np.abs(np.sum(PHGEN, axis=1) + EXITRATES - 1.0) > TOL):
    sys.exit("Validation test failed at initialization: the rows of the sub-transition matrix and the exit probabilities do not sum to one.")

# Sample sets of uncensored observations from the same geometric distribution.
# Case 1 averages over NREPLICATES pairs of sample sets, so that the comparison
# of the small and the large sample set does not depend on the particular seed.
# Cases 2-4 use the first small sample set.
np.random.seed(SEED)
SAMPLESSMALL = [np.random.geometric(PROB, size=NSMALL) for _ in range(NREPLICATES)]
SAMPLESLARGE = [np.random.geometric(PROB, size=NLARGE) for _ in range(NREPLICATES)]

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
    est = edph(nphases=NPHASES)
    bi, ni, nij = est.run(obs=obs,
                          initdist=INITDIST,
                          phgen=PHGEN,
                          exitrates=EXITRATES,
                          censoring=censoring)
    return bi.copy(), ni.copy(), nij.copy(), float(est.loglikelihood)


def runemstep(obs):
    '''
    Runs the E-step followed by the M-step on uncensored observations and
    returns the updated parameters.
    '''
    bi, ni, nij = runestep(obs)[:3]
    mst = mdph(nphases=NPHASES, nobs=obs.size)
    return mst.run(bi=bi, ni=ni, nij=nij)


def maxdiff(first, second):
    '''
    Returns the largest absolute difference between two E-step results.
    '''
    return max(np.max(np.abs(first[0] - second[0])),
               np.max(np.abs(first[1] - second[1])),
               np.max(np.abs(first[2] - second[2])),
               abs(first[3] - second[3]))


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

# The sub-transition matrix and the exit probabilities must on average be
# closer to the true parameters when the large sample set is used
if not phgenlarge < phgensmall:
    sys.exit("Validation test failed at case 1: the estimated sub-transition matrix is not closer to the true matrix for the large sample set.")

if not exitlarge < exitsmall:
    sys.exit("Validation test failed at case 1: the estimated exit probabilities are not closer to the true probabilities for the large sample set.")

# The initial distribution is recovered exactly from every sample set,
# irrespective of its size, so it is checked against the true parameter rather
# than compared between the two sample sizes. Because all exit probabilities
# are equal we have T*e = (1-PROB)*e, and hence
#
#     E(B_i|Y=y) = pi_i (T^(y-1) t)_i / (pi T^(y-1) t) = pi_i
#
# for every y. The expected number of processes initiating in state i is
# therefore pi_i for every observation, and the M-step returns pi regardless of
# the observations.
if initdistdev > TOL:
    sys.exit("Validation test failed at case 1: the initial distribution is not recovered exactly.")


# ------------------------------------------------------------------
# CASE 2: Left censoring equals interval censoring with lower limit zero
# ------------------------------------------------------------------

limits = OBSSMALL.astype(float)

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
# statistics n_i and n_ij are deliberately not compared: a right-censored
# process is only observed on [0,limit], where no absorption has been seen, so
# its counts cover that window alone, whereas an interval-censored process is
# known to have been absorbed before UPPERLIMIT, so its counts cover the entire
# lifetime [0,Y]. Both are correct for their own observation window, but they
# are not the same quantity.
if abs(resright[3] - resinterval[3]) > TOL:
    sys.exit("Validation test failed at case 3: right censoring and interval censoring with a large upper limit do not give the same log-likelihood.")

if np.max(np.abs(resright[0] - resinterval[0])) > TOL:
    sys.exit("Validation test failed at case 3: right censoring and interval censoring with a large upper limit do not give the same initiation counts.")

# The jump counts are instead checked against the limits directly. A
# right-censored process is known to be in a transient state at every one of
# the first limit steps, so it makes exactly limit jumps between transient
# states in that window, whatever the parameters are.
if abs(np.sum(resright[2]) - np.sum(limits)) > TOL:
    sys.exit("Validation test failed at case 3: the number of jumps between transient states does not equal the censoring limits for right-censored observations.")


# ------------------------------------------------------------------
# CASE 4: Interval censoring on (y-1,y] equals the uncensored observation y
# ------------------------------------------------------------------

# The observations are integers, so the event y-1 < Y <= y is the event Y = y
censinterval = np.column_stack([(OBSSMALL - 1).astype(float), OBSSMALL.astype(float)])

if maxdiff(runestep(OBSSMALL), runestep(OBSSMALL, censinterval)) > TOL:
    sys.exit("Validation test failed at case 4: interval censoring on (y-1,y] does not equal the uncensored observation y.")


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
        edph(nphases=NPHASES).run(**arguments)
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
        edph(nphases=NPHASES).run(**arguments)
    except ValueError as error:
        return str(error)

    return None


# only the row sum is at fault here, the elements staying non-negative
LARGEROWGEN = np.copy(PHGEN)
LARGEROWGEN[0, 0] = 1.0

NEGATIVEGEN = np.copy(PHGEN)
NEGATIVEGEN[0, 1] = -PHGEN[0, 1]

# feasible except for one fractional limit, which run() would truncate
FRACTIONALLIMIT = np.full((NSMALL, 2), np.nan)
FRACTIONALLIMIT[0, 1] = 2.5

INFEASIBLE = (
    ("no observations", {"obs": np.array([])}),
    ("two-dimensional observations", {"obs": np.copy(OBSSMALL).reshape(-1, 1)}),
    ("an observation at zero", {"obs": np.append(OBSSMALL, 0)}),
    ("a negative observation", {"obs": np.append(OBSSMALL, -1)}),
    ("a fractional observation", {"obs": np.append(OBSSMALL, 2.5).astype(float)}),
    ("a missing observation", {"obs": np.append(OBSSMALL, np.nan).astype(float)}),
    ("a negative initial probability", {"initdist": np.array([1.2, -0.2, 0.0])}),
    ("an initial distribution summing above one", {"initdist": np.array([0.5, 0.3, 0.3])}),
    ("a missing initial probability", {"initdist": np.array([0.5, 0.3, np.nan])}),
    ("a generator with a negative element", {"phgen": NEGATIVEGEN}),
    ("a generator with a row summing above one", {"phgen": LARGEROWGEN}),
    ("a negative exit probability", {"exitrates": np.array([PROB, -PROB, PROB])}),
    ("an exit probability above one", {"exitrates": np.array([PROB, 1.5, PROB])}),
    ("censoring with one column", {"censoring": np.full((NSMALL, 1), np.nan)}),
    ("censoring with too few rows", {"censoring": np.full((NSMALL - 1, 2), np.nan)}),
    ("a fractional censoring limit", {"censoring": FRACTIONALLIMIT}),
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
    ("exit probabilities of the wrong length", {"exitrates": EXITRATES[:2]}),
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

# the fitting class passes integers, a caller by hand may pass floats
if refuses(obs=OBSSMALL.astype(float)):
    sys.exit("Validation test failed at case 5: the E-step refused whole-numbered observations held as floating point.")

# The number of phases is checked when the class is built
for badphases in (0, -1, 2.5, "3"):
    try:
        edph(nphases=badphases)
        sys.exit("Validation test failed at case 5: the E-step was built with %s phases." % repr(badphases))
    except ValueError:
        pass

# The sample checks are made once per array, so a later call carrying a
# different and infeasible sample has to be checked afresh
model = edph(nphases=NPHASES)
model.run(obs=OBSSMALL, initdist=INITDIST, phgen=PHGEN, exitrates=EXITRATES)

try:
    model.run(obs=np.append(OBSSMALL, 2.5).astype(float), initdist=INITDIST,
              phgen=PHGEN, exitrates=EXITRATES)
    sys.exit("Validation test failed at case 5: a second call with a fractional observation was accepted, so the caching of the checks over the sample is not keyed on the array.")
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

print("edph: All tests passed.")
