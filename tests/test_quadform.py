'''
UNIT TEST FOR THE _quadform CLASS (the distribution of a weighted sum of
independent chi-square variables, used by fitreg's score test of the random
effect).

Sub-tests:
    Case 1: One weight is a scaled chi-square distribution, for several
            degrees of freedom; equal weights add their degrees of freedom.
    Case 2: Weights of both signs against the closed form of 2a E1 - 2b E2,
            E1, E2 exponential, on both sides of zero.
    Case 3: Mean and variance; zero weights are dropped; with no weight left
            the tail is NaN.
'''

import os
import sys

import numpy as np
from scipy.stats import chi2

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from phasedist.regtools import _quadform


TOL = 1e-8                     # the integral's accuracy (measured 3e-11 or better)
# points from just above zero (where the integrand barely oscillates) to far out
POINTS = np.array([1e-6, 1e-3, 0.3, 1.0, 2.5, 7.0, 15.0, 40.0, 200.0])


# ------------------------------------------------------------------
# CASE 1: One weight, and equal weights
# ------------------------------------------------------------------

for weight, dof in ((1.0, 1), (1.0, 2), (2.5, 3), (1.0, 4), (0.4, 10)):
    q = _quadform(np.array([weight]), np.array([dof]))
    if abs(q.sf(0.0) - 1.0) > TOL:
        sys.exit("Validation test failed at case 1: P(Q > 0) is %.12f for a positive weight, not 1." % q.sf(0.0))
    for x in POINTS:
        if abs(q.sf(x) - chi2.sf(x / weight, dof)) > TOL:
            sys.exit("Validation test failed at case 1: P(%.1f chi2(%d) > %.1f) is %.12f, not %.12f." % (weight, dof, x, q.sf(x), chi2.sf(x / weight, dof)))
# three equal weights of 1, 2 and 4 degrees of freedom are chi2(7) scaled
q = _quadform(np.array([1.5, 1.5, 1.5]), np.array([1, 2, 4]))
for x in POINTS:
    if abs(q.sf(x) - chi2.sf(x / 1.5, 7)) > TOL:
        sys.exit("Validation test failed at case 1: Equal weights do not add their degrees of freedom at %.1f." % x)


# ------------------------------------------------------------------
# CASE 2: Weights of both signs
# ------------------------------------------------------------------

# chi2(2) / 2 is a standard exponential, so Q = a chi2(2) - b chi2(2) is
# 2a E1 - 2b E2, whose tail is a e^(-x/2a) / (a + b) for x >= 0 and
# 1 - b e^(x/2b) / (a + b) for x < 0
for a, b in ((1.0, 0.5), (0.3, 2.0)):
    q = _quadform(np.array([a, -b]), np.array([2, 2]))
    for x in np.r_[-POINTS, POINTS]:
        exact = a / (a + b) * np.exp(-x / (2 * a)) if x >= 0 else 1.0 - b / (a + b) * np.exp(x / (2 * b))
        if abs(q.sf(x) - exact) > TOL:
            sys.exit("Validation test failed at case 2: P(Q > %.1f) is %.12f, not %.12f (a = %.1f, b = %.1f)." % (x, q.sf(x), exact, a, b))


# ------------------------------------------------------------------
# CASE 3: Moments, zero weights, no weight
# ------------------------------------------------------------------

q = _quadform(np.array([2.0, 0.0, -0.5]), np.array([3, 5, 4]))
if q.weights.size != 2 or abs(q.mean - (6.0 - 2.0)) > TOL or abs(q.var - 2.0 * (3 * 4.0 + 4 * 0.25)) > TOL:
    sys.exit("Validation test failed at case 3: The weights are %s, the mean %.6f and the variance %.6f." % (q.weights, q.mean, q.var))
if not np.isnan(_quadform(np.zeros(3), np.ones(3)).sf(1.0)):
    sys.exit("Validation test failed at case 3: With every weight zero the tail is not NaN.")


print("_quadform: All tests passed.")
