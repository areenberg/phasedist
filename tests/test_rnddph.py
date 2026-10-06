'''
UNIT TEST FOR THE rnddph CLASS (generates a random discrete-time phase-type
distribution with a specified structure).

The structure is given by which elements of the initial distribution, the
sub-transition matrix and the exit-probability vector are non-zero. The class
must fill those elements with random values and leave every other element at
zero, so that the generated distribution belongs to the intended subclass.

The structures tested are the ones fit.py can ask for -- general, generalized
Erlang, hyper-exponential, Coxian and generalized Coxian -- plus a tridiagonal
structure standing in for a custom one, which adds transitions back to the
previous phase and so covers a pattern none of the presets produce.

Sub-tests:
    Case 1: The elements intended to be zero are zero in the output, and the
            elements intended to be non-zero are non-zero, for every structure.
    Case 2: The output is random.
    Case 3: The output is a feasible discrete phase-type distribution.
    Case 4: The output is floating point, whatever the structure is given as.
    Case 5: The structure arrays passed in by the caller are not modified.
    Case 6: The input checks refuse a structure no DPH can be generated from,
            and accept every structure of case 1.

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

from phasedist.rnddph import rnddph


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

TOL = 1e-12  # tolerance for comparisons that are exact in theory
SEED = 0

PHASECOUNTS = (1, 2, 3, 5)  # includes the single-phase edge case
NDRAWS = 40                 # number of draws used to judge randomness

# Every structure the test claims to cover. Declared separately from the
# structures themselves so that a structure quietly disappearing from the table
# is caught rather than silently reducing the coverage of case 1.
STRUCTURENAMES = ("general", "generlang", "hyperexp", "coxian", "gencoxian",
                  "custom")


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def structures(nphases):
    '''
    Returns the supported structures for a given number of phases, as a
    dictionary of (initdist, phgen, exitrates) triples. The first five mirror
    the structures set up in fit.py; the last one is an arbitrary custom
    structure.
    '''
    firstphase = np.zeros(nphases)
    firstphase[0] = 1.0

    lastphase = np.zeros(nphases)
    lastphase[nphases - 1] = 1.0

    # diagonal plus superdiagonal, i.e. phases in series that may be repeated
    band = np.zeros((nphases, nphases))

    # the same, plus transitions back to the previous phase
    tridiagonal = np.zeros((nphases, nphases))

    for i in range(nphases):
        band[i, i] = 1.0
        tridiagonal[i, i] = 1.0
        if i < nphases - 1:
            band[i, i + 1] = 1.0
            tridiagonal[i, i + 1] = 1.0
        if i > 0:
            tridiagonal[i, i - 1] = 1.0

    return {
        "general": (np.ones(nphases), np.ones((nphases, nphases)), np.ones(nphases)),
        "generlang": (firstphase.copy(), band.copy(), lastphase.copy()),
        "hyperexp": (np.ones(nphases), np.eye(nphases), np.ones(nphases)),
        "coxian": (firstphase.copy(), band.copy(), np.ones(nphases)),
        "gencoxian": (np.ones(nphases), band.copy(), np.ones(nphases)),
        "custom": (np.ones(nphases), tridiagonal.copy(), np.ones(nphases)),
    }


def generate(nphases, structure, seed=None):
    '''
    Runs rnddph on a copy of the given structure and returns the generated
    parameters. The copy keeps the caller's arrays available for case 5.
    '''
    initdist, phgen, exitrates = structure
    if seed is not None:
        np.random.seed(seed)
    rnd = rnddph(nphases=nphases,
                 initdist=np.copy(initdist),
                 phgen=np.copy(phgen),
                 exitrates=np.copy(exitrates))
    return rnd.run()


def samepattern(generated, structure):
    '''
    Returns True if the generated array is zero in exactly the positions where
    the structure is zero, and non-zero everywhere else.
    '''
    return np.array_equal(np.asarray(generated) == 0.0,
                          np.asarray(structure) == 0.0)


def isfeasible(initdist, phgen, exitrates):
    '''
    Returns True if the parameters are a feasible DPH representation.
    '''
    if not (np.all(np.isfinite(initdist))
            and np.all(np.isfinite(phgen))
            and np.all(np.isfinite(exitrates))):
        return False
    if np.any(initdist < 0.0) or abs(np.sum(initdist) - 1.0) > TOL:
        return False
    if np.any(phgen < 0.0) or np.any(exitrates < 0.0):
        return False
    if np.any(np.abs(np.sum(phgen, axis=1) + exitrates - 1.0) > TOL):
        return False
    return True


# The structures must be well formed, otherwise the cases below would be
# exercising something that could not describe a phase-type distribution
for nphases in PHASECOUNTS:

    available = structures(nphases)

    if tuple(sorted(available)) != tuple(sorted(STRUCTURENAMES)):
        sys.exit("Validation test failed at initialization: the structures do not match the list of structures the test claims to cover.")

    for name, (initdist, phgen, exitrates) in available.items():
        if np.count_nonzero(initdist) == 0 or np.count_nonzero(exitrates) == 0:
            sys.exit("Validation test failed at initialization: structure '%s' has no non-zero initial probability or no non-zero exit probability." % name)
        if np.any(np.count_nonzero(phgen, axis=1) == 0):
            sys.exit("Validation test failed at initialization: structure '%s' has a phase with no transitions to other transient phases." % name)

np.random.seed(SEED)


# ------------------------------------------------------------------
# CASE 1: The intended zeros are zero and the intended non-zeros are not
# ------------------------------------------------------------------

for nphases in PHASECOUNTS:
    for name, structure in structures(nphases).items():
        for draw in range(NDRAWS):

            initdist, phgen, exitrates = generate(nphases, structure)

            if not samepattern(initdist, structure[0]):
                sys.exit("Validation test failed at case 1: the generated initial distribution does not have the zeros of the '%s' structure." % name)

            if not samepattern(phgen, structure[1]):
                sys.exit("Validation test failed at case 1: the generated sub-transition matrix does not have the zeros of the '%s' structure." % name)

            if not samepattern(exitrates, structure[2]):
                sys.exit("Validation test failed at case 1: the generated exit probabilities do not have the zeros of the '%s' structure." % name)


# ------------------------------------------------------------------
# CASE 2: The output is random
# ------------------------------------------------------------------

# Counters for the checks that only apply where the structure leaves room for
# variation, so that the case cannot pass by quietly skipping all of them
withindrawchecks = 0
initdistchecks = 0

for nphases in PHASECOUNTS:
    for name, structure in structures(nphases).items():

        initzeros, phgenzeros, exitzeros = (structure[0] != 0,
                                            structure[1] != 0,
                                            structure[2] != 0)

        # the same seed must reproduce the same parameters
        first = generate(nphases, structure, seed=SEED)
        again = generate(nphases, structure, seed=SEED)

        if not all(np.array_equal(a, b) for a, b in zip(first, again)):
            sys.exit("Validation test failed at case 2: the same seed did not reproduce the same parameters for the '%s' structure." % name)

        # different seeds must give different parameters. The initial
        # distribution is excluded here: where the structure allows only one
        # phase to be the initial one, that probability is one whatever is
        # drawn, and the exit probabilities and the generator carry the
        # randomness instead.
        other = generate(nphases, structure, seed=SEED + 1)

        if np.array_equal(first[2][exitzeros], other[2][exitzeros]):
            sys.exit("Validation test failed at case 2: two different seeds gave the same exit probabilities for the '%s' structure." % name)

        if np.array_equal(first[1][phgenzeros], other[1][phgenzeros]):
            sys.exit("Validation test failed at case 2: two different seeds gave the same sub-transition matrix for the '%s' structure." % name)

        # every element that is free to vary must take a different value in
        # every draw, which rules out a constant being returned
        draws = [generate(nphases, structure) for _ in range(NDRAWS)]

        for position in np.flatnonzero(exitzeros):
            if np.unique([d[2][position] for d in draws]).size != NDRAWS:
                sys.exit("Validation test failed at case 2: an exit probability of the '%s' structure does not change between draws." % name)

        for position in zip(*np.nonzero(phgenzeros)):
            if np.unique([d[1][position] for d in draws]).size != NDRAWS:
                sys.exit("Validation test failed at case 2: an element of the sub-transition matrix of the '%s' structure does not change between draws." % name)

        if np.count_nonzero(initzeros) >= 2:
            initdistchecks += 1
            for position in np.flatnonzero(initzeros):
                if np.unique([d[0][position] for d in draws]).size != NDRAWS:
                    sys.exit("Validation test failed at case 2: an initial probability of the '%s' structure does not change between draws." % name)

        # within a single draw, elements sharing a structure must not all be
        # given the same value. This is what a single sampled value broadcast
        # across the non-zero positions would look like.
        for initdist, phgen, exitrates in draws:

            if np.count_nonzero(initzeros) >= 2:
                withindrawchecks += 1
                values = initdist[initzeros]
                if values.max() == values.min():
                    sys.exit("Validation test failed at case 2: every non-zero initial probability of the '%s' structure was given the same value." % name)

            if np.count_nonzero(exitzeros) >= 2:
                withindrawchecks += 1
                values = exitrates[exitzeros]
                if values.max() == values.min():
                    sys.exit("Validation test failed at case 2: every non-zero exit probability of the '%s' structure was given the same value." % name)

            for i in range(nphases):
                if np.count_nonzero(phgenzeros[i]) >= 2:
                    withindrawchecks += 1
                    values = phgen[i][phgenzeros[i]]
                    if values.max() == values.min():
                        sys.exit("Validation test failed at case 2: every non-zero element in a row of the sub-transition matrix of the '%s' structure was given the same value." % name)

if withindrawchecks == 0 or initdistchecks == 0:
    sys.exit("Validation test failed at case 2: no structure had enough non-zero elements to check that they are given different values.")


# ------------------------------------------------------------------
# CASE 3: The output is a feasible discrete phase-type distribution
# ------------------------------------------------------------------

for nphases in PHASECOUNTS:
    for name, structure in structures(nphases).items():
        for draw in range(NDRAWS):

            initdist, phgen, exitrates = generate(nphases, structure)

            if not isfeasible(initdist, phgen, exitrates):
                sys.exit("Validation test failed at case 3: the '%s' structure produced parameters that are not a feasible discrete phase-type distribution." % name)


# ------------------------------------------------------------------
# CASE 4: The output is floating point
# ------------------------------------------------------------------

# The structure may reasonably be written with integers, as the class
# documentation itself does. The sampled values must not be truncated to fit an
# integer array.
INTEGERSTRUCTURE = (np.array([1, 1, 1]),
                    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]]),
                    np.array([1, 1, 1]))

initdist, phgen, exitrates = generate(3, INTEGERSTRUCTURE, seed=SEED)

if not all(np.issubdtype(np.asarray(a).dtype, np.floating)
           for a in (initdist, phgen, exitrates)):
    sys.exit("Validation test failed at case 4: an integer structure did not produce floating point parameters.")

if not isfeasible(initdist, phgen, exitrates):
    sys.exit("Validation test failed at case 4: an integer structure did not produce a feasible discrete phase-type distribution.")

# truncation would leave only zeros and ones behind
if not (np.any((initdist > 0.0) & (initdist < 1.0))
        and np.any((exitrates > 0.0) & (exitrates < 1.0))
        and np.any((phgen > 0.0) & (phgen < 1.0))):
    sys.exit("Validation test failed at case 4: the values generated from an integer structure were truncated.")

# fit.py holds the structure as a numpy matrix, which must not be carried into
# the generated parameters, since a matrix stays two-dimensional through every
# operation applied to it later on
initdist, phgen, exitrates = generate(3, (np.ones(3),
                                          np.matrix(np.ones((3, 3))),
                                          np.ones(3)), seed=SEED)

if any(isinstance(a, np.matrix) for a in (initdist, phgen, exitrates)):
    sys.exit("Validation test failed at case 4: a structure given as a numpy matrix produced parameters that are still a numpy matrix.")

if not isfeasible(initdist, np.asarray(phgen), exitrates):
    sys.exit("Validation test failed at case 4: a structure given as a numpy matrix did not produce a feasible discrete phase-type distribution.")


# ------------------------------------------------------------------
# CASE 5: The caller's structure arrays are left alone
# ------------------------------------------------------------------

for nphases in PHASECOUNTS:
    for name, structure in structures(nphases).items():

        given = tuple(np.copy(a) for a in structure)

        rnd = rnddph(nphases=nphases,
                     initdist=given[0],
                     phgen=given[1],
                     exitrates=given[2])
        rnd.run()

        if not all(np.array_equal(a, b) for a, b in zip(given, structure)):
            sys.exit("Validation test failed at case 5: generating from the '%s' structure modified the arrays passed in by the caller." % name)


# ------------------------------------------------------------------
# CASE 6: The input checks
# ------------------------------------------------------------------

def refuses(nphases, initdist, phgen, exitrates):
    '''
    Returns True if the class refuses the given structure.
    '''
    try:
        rnddph(nphases=nphases, initdist=initdist, phgen=phgen,
               exitrates=exitrates)
    except ValueError:
        return True

    return False


def refusalmessage(nphases, initdist, phgen, exitrates):
    '''
    Returns the message the class refuses the structure with, or None.
    '''
    try:
        rnddph(nphases=nphases, initdist=initdist, phgen=phgen,
               exitrates=exitrates)
    except ValueError as error:
        return str(error)

    return None


# the checks must not have broken the class they were added to
for nphases in PHASECOUNTS:
    for name, structure in structures(nphases).items():
        if refuses(nphases, *structure):
            sys.exit("Validation test failed at case 6: the '%s' structure of %d phases was refused." % (name, nphases))

# a feasible two-phase structure, varied one element at a time below
BASEINIT = np.ones(2)
BASEGEN = np.array([[1.0, 1.0],
                    [1.0, 1.0]])
BASEEXIT = np.ones(2)

# normalizing over no non-zero element would divide by zero
NOINIT = np.zeros(2)

# a row is scaled to one minus the exit probability, so a phase with neither
# is never left
DEADPHASEGEN = np.array([[1.0, 1.0],
                         [0.0, 0.0]])
DEADPHASEEXIT = np.array([1.0, 0.0])

INFEASIBLE = (
    ("an initial distribution of the wrong length", 2, np.ones(3), BASEGEN, BASEEXIT),
    ("a generator of the wrong shape", 2, BASEINIT, np.ones((3, 3)), BASEEXIT),
    ("exit probabilities of the wrong length", 2, BASEINIT, BASEGEN, np.ones(3)),
    ("a negative element in the initial distribution", 2, np.array([1.0, -1.0]), BASEGEN, BASEEXIT),
    ("a negative element in the generator", 2, BASEINIT, np.array([[1.0, -1.0], [1.0, 1.0]]), BASEEXIT),
    ("a negative exit probability", 2, BASEINIT, BASEGEN, np.array([1.0, -1.0])),
    ("a missing element in the structure", 2, BASEINIT, BASEGEN, np.array([1.0, np.nan])),
    ("an all-zero initial distribution", 2, NOINIT, BASEGEN, BASEEXIT),
    ("a phase that is never left", 2, BASEINIT, DEADPHASEGEN, DEADPHASEEXIT),
)

for description, nphases, initdist, phgen, exitrates in INFEASIBLE:
    if not refuses(nphases, initdist, phgen, exitrates):
        sys.exit("Validation test failed at case 6: the class accepted %s." % description)

# A bad number of phases is caught by the check on the dimensions in any
# case, so the message is what is asserted here.
for badphases in (0, -1, 2.5, "3"):
    message = refusalmessage(badphases, BASEINIT, BASEGEN, BASEEXIT)

    if message is None or "number of phases" not in message:
        sys.exit("Validation test failed at case 6: given %s phases the class reported '%s', which does not name the number of phases as the cause." % (repr(badphases), message))

# a self-loop is a transition in the discrete case, so this phase is left
# through the loop and is not dead
SELFLOOPGEN = np.array([[1.0, 1.0],
                        [0.0, 1.0]])

if refuses(2, BASEINIT, SELFLOOPGEN, np.array([1.0, 0.0])):
    sys.exit("Validation test failed at case 6: a phase whose only transition is a self-loop was refused as a phase that is never left.")

# a list has to be accepted, since the checks convert it
if refuses(2, [1, 1], [[1, 1], [1, 1]], [1, 1]):
    sys.exit("Validation test failed at case 6: a structure given as lists was refused.")


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print("rnddph: All tests passed.")
