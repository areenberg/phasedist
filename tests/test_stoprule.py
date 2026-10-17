'''
UNIT TEST FOR THE _stoprule CLASS (the stopping rule of fitreg's EM).

Sub-tests:
    Case 1: On a geometric trace the Aitken estimate of the remaining gain is
            exact, infinite before window + 1 steps, and the rule stops within
            the tolerance of the limit.
    Case 2: A plateau before a slow phase fires the Aitken estimate, but the
            confirmation block rejects the stop until the slow phase is over.
    Case 3: The confirmation block is counted in iterations; a decrease or a
            rate of one is handled; reset() forgets the history.
'''

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from phasedist.regtools import _stoprule


TOL = 1e-12
TOLERANCE = 1e-5     # the rule's tolerance
WINDOW = 10
LIMIT = -100.0       # limit of the traces
RATE = 0.95          # geometric rate of case 1
GAP = 3.0            # initial distance to the limit


def geometric(k):
    """The log-likelihood after k steps of a geometric approach to LIMIT."""
    return LIMIT - GAP * RATE ** k


# ------------------------------------------------------------------
# CASE 1: A geometric trace
# ------------------------------------------------------------------

# for ll_k = L - c r^k the windowed Aitken estimate step * r / (1 - r) equals
# the remaining gain c r^k exactly
rule = _stoprule(tolerance=TOLERANCE, window=WINDOW)
stopped = None
for k in range(5000):
    done = rule.update(geometric(k))
    if k < WINDOW + 1 and rule.eps != np.inf:
        sys.exit("Validation test failed at case 1: The estimate is finite after %d steps." % (k + 1))
    if k >= WINDOW + 1 and abs(rule.eps - GAP * RATE ** k) > TOL * GAP:
        sys.exit("Validation test failed at case 1: The estimate %.3e differs from the remaining gain %.3e." % (rule.eps, GAP * RATE ** k))
    if done:
        stopped = k
        break
if stopped is None or LIMIT - geometric(stopped) > TOLERANCE:
    sys.exit("Validation test failed at case 1: The rule stopped at %s, %.3e from the limit." % (stopped, LIMIT - geometric(stopped or 0)))
# it fired where the remaining gain first fell below the tolerance, and
# stopped one confirmation block later
first = int(np.ceil(np.log(TOLERANCE / GAP) / np.log(RATE)))
block = max(200, int(np.ceil(0.25 * (first + 1))))
if stopped != first + block:
    sys.exit("Validation test failed at case 1: The rule stopped at step %d, not %d (fired at %d, block %d)." % (stopped, first + block, first, block))


# ------------------------------------------------------------------
# CASE 2: A plateau before a slow phase
# ------------------------------------------------------------------

# fast to within 4e-7 of a plateau, a plateau of 1e-12 steps, then a slow
# phase that gains 0.3 more: The plateau fires the estimate, and only the
# confirmation block (which ends inside the slow phase) keeps the rule from
# stopping 0.3 short
FAST, PLATEAU = 150, 60


def trace(k):
    """Fast phase, plateau, slow phase, converging to LIMIT."""
    base = LIMIT - 0.3
    if k < FAST:
        return base - GAP * 0.9 ** k
    value = base - GAP * 0.9 ** FAST + 1e-12 * min(k - FAST, PLATEAU)
    if k > FAST + PLATEAU:
        value += 0.3 * (1.0 - 0.995 ** (k - FAST - PLATEAU))
    return value


rule = _stoprule(tolerance=TOLERANCE, window=WINDOW)
fired = stopped = None
for k in range(20000):
    done = rule.update(trace(k))
    if fired is None and rule.confirmstart is not None:
        fired = k
    if done:
        stopped = k
        break
if fired is None or fired > FAST + PLATEAU:
    sys.exit("Validation test failed at case 2: The estimate did not fire on the plateau (fired at %s)." % fired)
if stopped is None or trace(20000) - trace(stopped) > TOLERANCE:
    sys.exit("Validation test failed at case 2: The rule stopped at %s, %.3e short of the limit." % (stopped, trace(20000) - trace(stopped or 0)))


# ------------------------------------------------------------------
# CASE 3: Iterations, decreases, rate one, reset
# ------------------------------------------------------------------

# the block is counted in iterations: Steps of two iterations each
rule = _stoprule(tolerance=TOLERANCE, window=WINDOW)
for k in range(5000):
    rule.update(geometric(k), iterations=2)
    if rule.confirmstart is not None:
        done = 2 * (k + 1)
        if rule.confirmend != done + max(200, int(np.ceil(0.25 * done))):
            sys.exit("Validation test failed at case 3: The confirmation block is not counted in iterations.")
        break

# a decrease gives the (negative) step itself; a constant step ratio of one
# gives an infinite estimate
rule = _stoprule(tolerance=TOLERANCE, window=WINDOW)
for k in range(WINDOW + 1):
    rule.update(float(k))
rule.update(WINDOW - 0.5)
if rule.eps != -0.5:
    sys.exit("Validation test failed at case 3: After a decrease the estimate is %s, not the step -0.5." % rule.eps)
rule = _stoprule(tolerance=TOLERANCE, window=WINDOW)
for k in range(WINDOW + 3):
    rule.update(float(k))
if rule.eps != np.inf:
    sys.exit("Validation test failed at case 3: With a rate of one the estimate is %s, not infinite." % rule.eps)

# reset forgets everything
rule.reset()
if rule.history or rule.eps != np.inf or rule.confirmstart is not None or rule.converged:
    sys.exit("Validation test failed at case 3: reset() kept part of the history.")
rule.update(1.0)
if rule.eps != np.inf:
    sys.exit("Validation test failed at case 3: After reset() the estimate is finite at once.")


print("_stoprule: All tests passed.")
