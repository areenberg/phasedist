'''
UNIT TEST FOR THE _ppband CLASS (the simultaneous confidence band of
fitreg's PP plot).

Sub-tests:
    Case 1: The shape of the band: Limits inside (0, 1), increasing in k, and
            around the mean k/(n+1) of the k-th sorted uniform.
    Case 2: Equal precision: Every limit pair holds the same probability
            1 - gamma under Beta(k, n-k+1), and gamma is below 1 - confidence
            (a simultaneous band is wider than a pointwise one).
    Case 3: One transform: The band is [gamma/2, 1 - gamma/2] with gamma
            close to 1 - confidence.
    Case 4: Coverage: All n sorted uniforms lie inside the band with the
            requested probability, by Clopper-Pearson intervals at a
            Bonferroni-corrected level over several n and levels.
    Case 5: The band of a given n and level is always the same.
'''

import os
import sys

import numpy as np
from scipy.special import betainc
from scipy.stats import beta

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from phasedist.regtools import _ppband


TOL = 1e-12
SEED = 4
# the coverage cases: (n, confidence); n = 2 to 1,000 and two levels
COVERAGE = [(2, 0.95), (12, 0.95), (120, 0.95), (1000, 0.95), (40, 0.90), (300, 0.90)]
NSAMPLES = 4000          # simulated samples of sorted uniforms per case
ALPHA = 0.01             # level of all the coverage intervals together
LEVEL = 1.0 - ALPHA / len(COVERAGE)
# The band's level gamma is itself simulated, so its coverage differs from the
# requested one by 0.2-0.5 points; with 4,000 samples per case the intervals
# are about +-1 point wide, and the case passed for seeds 0-9.


# ------------------------------------------------------------------
# CASE 1: The shape of the band
# ------------------------------------------------------------------

for n in (1, 2, 7, 120):
    lower, upper = _ppband().run(n, 0.95)
    k = np.arange(1, n + 1)
    if lower.shape != (n,) or upper.shape != (n,):
        sys.exit("Validation test failed at case 1: The band of %d transforms has %s and %s limits." % (n, lower.shape, upper.shape))
    if np.any(lower <= 0.0) or np.any(upper >= 1.0) or np.any(lower >= upper):
        sys.exit("Validation test failed at case 1: The band of %d transforms is not inside (0, 1) with lower below upper." % n)
    if n > 1 and (np.any(np.diff(lower) <= 0.0) or np.any(np.diff(upper) <= 0.0)):
        sys.exit("Validation test failed at case 1: The limits of the band of %d transforms do not increase in k." % n)
    if np.any(lower >= k / (n + 1.0)) or np.any(upper <= k / (n + 1.0)):
        sys.exit("Validation test failed at case 1: The band of %d transforms does not contain the means k/(n+1)." % n)


# ------------------------------------------------------------------
# CASE 2: Equal precision
# ------------------------------------------------------------------

for n, confidence in ((5, 0.95), (120, 0.95), (120, 0.80), (800, 0.99)):
    band = _ppband()
    lower, upper = band.run(n, confidence)
    k = np.arange(1, n + 1)
    held = betainc(k, n - k + 1.0, upper) - betainc(k, n - k + 1.0, lower)
    if np.max(np.abs(held - (1.0 - band.gamma))) > 1e-10:
        sys.exit("Validation test failed at case 2: The intervals of the band of %d transforms hold %.12f to %.12f, not 1 - gamma = %.12f each." % (n, held.min(), held.max(), 1.0 - band.gamma))
    # each tail holds gamma/2
    if np.max(np.abs(betainc(k, n - k + 1.0, lower) - band.gamma / 2.0)) > 1e-10:
        sys.exit("Validation test failed at case 2: The lower limits of the band of %d transforms do not leave gamma/2 below them." % n)
    if not 0.0 < band.gamma < 1.0 - confidence:
        sys.exit("Validation test failed at case 2: gamma = %.6f for %d transforms at %.2f, not between 0 and the pointwise level %.2f." % (band.gamma, n, confidence, 1.0 - confidence))


# ------------------------------------------------------------------
# CASE 3: One transform
# ------------------------------------------------------------------

# the smallest two-sided tail probability of one uniform, 2 min(u, 1 - u), is
# itself uniform, so gamma estimates 1 - confidence; at the default 20,000
# simulated samples its standard error is 0.0015 at 0.95
band = _ppband()
lower, upper = band.run(1, 0.95)
if abs(band.gamma - 0.05) > 0.006:
    sys.exit("Validation test failed at case 3: gamma = %.5f for one transform, not within 0.006 of 0.05." % band.gamma)
if abs(lower[0] - band.gamma / 2.0) > TOL or abs(upper[0] - (1.0 - band.gamma / 2.0)) > TOL:
    sys.exit("Validation test failed at case 3: The band of one transform is [%.12f, %.12f], not [gamma/2, 1 - gamma/2]." % (lower[0], upper[0]))


# ------------------------------------------------------------------
# CASE 4: Coverage
# ------------------------------------------------------------------

rng = np.random.default_rng(SEED)
for n, confidence in COVERAGE:
    lower, upper = _ppband().run(n, confidence)
    inside = 0
    for start in range(0, NSAMPLES, 500):
        ordered = np.sort(rng.random((500, n)), axis=1)
        inside += int(np.sum(np.all((ordered >= lower) & (ordered <= upper), axis=1)))
    # Clopper-Pearson interval for the coverage
    a = 1.0 - LEVEL
    low = 0.0 if inside == 0 else beta.ppf(a / 2.0, inside, NSAMPLES - inside + 1)
    high = 1.0 if inside == NSAMPLES else beta.ppf(1.0 - a / 2.0, inside + 1, NSAMPLES - inside)
    if not low <= confidence <= high:
        sys.exit("Validation test failed at case 4: The band of %d transforms at %.2f covered %d of %d samples, whose interval [%.4f, %.4f] excludes %.2f." % (n, confidence, inside, NSAMPLES, low, high, confidence))


# ------------------------------------------------------------------
# CASE 5: The same band every time
# ------------------------------------------------------------------

first = _ppband().run(57, 0.95)
second = _ppband().run(57, 0.95)
if np.any(first[0] != second[0]) or np.any(first[1] != second[1]):
    sys.exit("Validation test failed at case 5: Two bands of the same n and level differ.")


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print("_ppband: All tests passed.")
