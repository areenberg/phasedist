'''
UNIT TEST FOR THE _aghq CLASS (adaptive Gauss-Hermite quadrature of fitreg).

Sub-tests:
    Case 1: The nodes are mode + sqrt(2) scale x for the Gauss-Hermite points x.
    Case 2: Exactness: A Gaussian integrand times a polynomial of degree up to
            2R - 1 is integrated exactly; the node weights sum to one and are
            the posterior probabilities of the nodes.
    Case 3: A non-Gaussian integrand (an exponential likelihood times a normal
            prior) against scipy's quad, the error falling with the nodes;
            logarithms keep tiny integrands finite.
'''

import os
import sys

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from phasedist.regtools import _aghq


TOL = 1e-12
TOLQUAD = 1e-10      # 31 nodes against quad (measured 1.5e-13)
MODES = np.array([-1.3, 0.0, 2.5])
SCALES = np.array([0.4, 1.0, 2.2])


# ------------------------------------------------------------------
# CASE 1: Nodes
# ------------------------------------------------------------------

for R in (1, 3, 7):
    rule = _aghq(R)
    points, _ = hermgauss(R)
    nodes = rule.nodes(MODES, SCALES)
    if nodes.shape != (MODES.size, R) or \
            np.max(np.abs(nodes - (MODES[:, None] + np.sqrt(2.0) * SCALES[:, None] * points[None, :]))) > TOL:
        sys.exit("Validation test failed at case 1: The nodes of %d points are wrong." % R)


# ------------------------------------------------------------------
# CASE 2: Exactness for a Gaussian times a polynomial
# ------------------------------------------------------------------

# integral of (1 + (b - m)^2 + (b - m)^4) exp(-(b - m)^2 / (2 s^2) + c)
# = exp(c) s sqrt(2 pi) (1 + s^2 + 3 s^4), exact for R >= 3 (degree 4 <= 2R - 1)
c = np.array([0.7, -2.0, 5.0])
exact = c + np.log(SCALES * np.sqrt(2.0 * np.pi) * (1.0 + SCALES ** 2 + 3.0 * SCALES ** 4))
for R in (3, 5, 9):
    rule = _aghq(R)
    nodes = rule.nodes(MODES, SCALES)
    d = nodes - MODES[:, None]
    logintegrand = np.log(1.0 + d ** 2 + d ** 4) - d ** 2 / (2.0 * SCALES[:, None] ** 2) + c[:, None]
    logintegral, weights = rule.weights(logintegrand, SCALES)
    if np.max(np.abs(logintegral - exact)) > TOL:
        sys.exit("Validation test failed at case 2: With %d nodes the log-integral is off by %.3e." % (R, np.max(np.abs(logintegral - exact))))
    posterior = np.exp(logintegrand + np.log(hermgauss(R)[1]) + hermgauss(R)[0] ** 2
                       + np.log(np.sqrt(2.0) * SCALES)[:, None] - logintegral[:, None])
    if np.max(np.abs(weights.sum(axis=1) - 1.0)) > TOL or np.max(np.abs(weights - posterior)) > TOL:
        sys.exit("Validation test failed at case 2: The node weights do not sum to one or are not the posterior.")
# with fewer nodes the degree-4 polynomial is not integrated exactly
rule = _aghq(2)
nodes = rule.nodes(MODES, SCALES)
d = nodes - MODES[:, None]
logintegral, _ = rule.weights(np.log(1.0 + d ** 2 + d ** 4) - d ** 2 / (2.0 * SCALES[:, None] ** 2) + c[:, None], SCALES)
if np.min(np.abs(logintegral - exact)) < 1e-3:
    sys.exit("Validation test failed at case 2: Two nodes integrated a degree-4 polynomial exactly.")


# ------------------------------------------------------------------
# CASE 3: A non-Gaussian integrand
# ------------------------------------------------------------------

# a group of exponential observations y with rate exp(b) and b ~ N(0, sigma^2):
# log integrand n b - exp(b) sum(y) - b^2 / (2 sigma^2) - log(sigma sqrt(2 pi))
y = np.array([0.3, 1.7, 0.9, 2.4])
sigma = 0.8


def logintegrand(b):
    return y.size * b - np.exp(b) * y.sum() - b ** 2 / (2 * sigma ** 2) - np.log(sigma * np.sqrt(2 * np.pi))


# mode and curvature scale by Newton, as the class's caller does
mode = 0.0
for _ in range(50):
    gradient = y.size - np.exp(mode) * y.sum() - mode / sigma ** 2
    hessian = -np.exp(mode) * y.sum() - 1.0 / sigma ** 2
    mode -= gradient / hessian
scale = np.sqrt(-1.0 / (-np.exp(mode) * y.sum() - 1.0 / sigma ** 2))
reference = np.log(quad(lambda b: np.exp(logintegrand(b)), -15, 15, epsabs=0, epsrel=1e-13, limit=200)[0])
errors = []
for R in (5, 11, 21, 31):
    rule = _aghq(R)
    nodes = rule.nodes(np.array([mode]), np.array([scale]))
    value, _ = rule.weights(logintegrand(nodes), np.array([scale]))
    errors.append(abs(value[0] - reference))
# the error falls with the number of nodes (measured 2e-4, 8e-7, 1e-10, 2e-13)
if errors[-1] > TOLQUAD or not all(a > b for a, b in zip(errors, errors[1:])):
    sys.exit("Validation test failed at case 3: The errors against quad for 5, 11, 21 and 31 nodes are %s." % errors)

# the same integrand shifted far down stays finite in logarithms
value, weights = rule.weights(logintegrand(nodes) - 1e4, np.array([scale]))
if not np.isfinite(value[0]) or abs(value[0] - (reference - 1e4)) > TOLQUAD * 1e4 or abs(weights.sum() - 1.0) > TOL:
    sys.exit("Validation test failed at case 3: A tiny integrand did not stay finite in logarithms.")


print("_aghq: All tests passed.")
