'''
UNIT TEST FOR THE _unif CLASS (matrix exponential by uniformization).

The test compares exp(Ty) as computed by uniformization against scipy's expm,
over three sub-generators and a range of y.

A note on the choice of oracle. expm is a fair reference only where expm is
itself reliable. Its accuracy degrades when the rates of T are nearly equal,
because the eigenvalues are then nearly confluent and the divided differences
behind exp(Ty) lose digits to cancellation -- which is the very situation
uniformization was adopted to handle, and where it is expm that is wrong rather
than _unif. The three generators below therefore have well-separated rates, and
a generator with near-equal rates (an Erlang, say) must NOT be added here: the
case would fail, and the failure would be the oracle's.

Sub-tests:
    Case 1: exp(Ty) agrees with expm at every evaluation time, for a generator
            with no zero entries, one with zeros, and one of a different order.
    Case 2: the nrows argument returns the leading rows of the full result.
    Case 3: a generator whose spectrum is wide enough that exp(-Gamma*y)
            underflows to zero while exp(Ty) is still far from zero.

References:
    Stewart, W. J. Introduction to the Numerical Solution of Markov Chains.
    Princeton University Press. https://www.jstor.org/stable/j.ctvcm4gtc
'''

import os
import sys
import numpy as np
from scipy.linalg import expm

# Load phasedist from the src-folder so the test can be run without installing
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from phasedist.unif import _unif


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

TOL = 1e-10          # the two methods agree to about 1e-14 on these generators
TOLERANCE = 1e-12    # truncation tolerance handed to the class

# Evaluation times, spanning from zero to well into the tail of each
# distribution, since the number of terms in the series grows with Gamma*y and
# a method that is accurate at small y may not be at large y.
TIMES = np.array([0.0, 0.05, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0])

# A fully dense generator: every phase reaches every other, so no entry is zero.
DENSEGEN = np.array([[-1.20, 0.40, 0.30],
                     [0.20, -0.90, 0.20],
                     [0.50, 0.30, -1.30]])

# A Coxian generator: the phases in series, which leaves zeros below the
# diagonal and in the corner.
COXIANGEN = np.array([[-1.50, 0.90, 0.00],
                      [0.00, -1.10, 0.70],
                      [0.00, 0.00, -0.80]])

# A fourth-order generator with a feedback loop, so that the test is not run
# at a single order and the zero pattern is neither dense nor triangular.
LOOPGEN = np.array([[-2.10, 1.30, 0.00, 0.40],
                    [0.00, -1.70, 0.90, 0.00],
                    [0.60, 0.00, -1.40, 0.50],
                    [0.00, 0.20, 0.00, -0.95]])

GENERATORS = (
    ("dense", DENSEGEN),
    ("Coxian", COXIANGEN),
    ("loop", LOOPGEN),
)


# A generator for case 3, with one fast phase feeding a very slow one. Gamma is
# set by the fast phase and the decay of exp(Ty) by the slow one, so the two can
# be driven apart: at UNDERFLOWTIME the Poisson weights are centred far beyond
# where exp(-Gamma*y) is representable, while exp(Ty) itself is still of ordinary
# size. None of the three generators above can do this, since their rates are
# close enough that exp(Ty) has decayed to nothing long before exp(-Gamma*y)
# underflows, which would make the comparison a comparison of zeros.
SPREADGEN = np.array([[-2.00, 1.90],
                      [0.00, -0.01]])

UNDERFLOWTIME = 500.0


# The constants above are the whole test, so a typo in one of them would quietly
# make a case meaningless rather than fail it. Each one is checked to be a
# sub-intensity matrix, and to carry the zero pattern the case is named for.
for name, gen in GENERATORS:
    offdiagonal = gen - np.diag(np.diag(gen))

    if np.any(np.diag(gen) >= 0.0):
        sys.exit("Validation test failed at initialization: the %s generator has a diagonal entry that is not negative." % name)

    if np.any(offdiagonal < 0.0):
        sys.exit("Validation test failed at initialization: the %s generator has a negative off-diagonal entry." % name)

    if np.any(np.sum(gen, axis=1) > TOL):
        sys.exit("Validation test failed at initialization: the rows of the %s generator do not sum to zero or less, so it is not a sub-intensity matrix." % name)

    rates = -np.diag(gen)
    if np.min(np.abs(np.diff(np.sort(rates)))) < 0.1:
        sys.exit("Validation test failed at initialization: two rates of the %s generator are within 0.1 of one another. expm is the reference here and loses accuracy as the rates converge, so the generators have to keep theirs apart." % name)

offdiagonal = SPREADGEN - np.diag(np.diag(SPREADGEN))

if (np.any(np.diag(SPREADGEN) >= 0.0) or np.any(offdiagonal < 0.0)
        or np.any(np.sum(SPREADGEN, axis=1) > TOL)):
    sys.exit("Validation test failed at initialization: the spread generator of case 3 is not a sub-intensity matrix.")

if np.any(DENSEGEN == 0.0):
    sys.exit("Validation test failed at initialization: the dense generator contains a zero, so the case for a generator without zeros is not being run.")

if not np.any(COXIANGEN == 0.0) or not np.any(LOOPGEN == 0.0):
    sys.exit("Validation test failed at initialization: a generator meant to carry zeros does not, so the case for a sparse generator is not being run.")


# ------------------------------------------------------------------
# CASE 1: exp(Ty) against expm
# ------------------------------------------------------------------

for name, gen in GENERATORS:
    model = _unif(tolerance=TOLERANCE)
    computed = model.run(gen, TIMES)

    for index, y in enumerate(TIMES):
        reference = expm(gen * y)
        difference = float(np.max(np.abs(computed[index] - reference)))

        if not np.all(np.isfinite(computed[index])):
            sys.exit("Validation test failed at case 1: exp(Ty) of the %s generator is not a number at y = %s." % (name, y))

        if difference > TOL:
            sys.exit("Validation test failed at case 1: for the %s generator at y = %s, uniformization and expm differ by %.3e, which is more than %.1e." % (name, y, difference, TOL))

    # the reported bound is the class's own claim about its truncation error,
    # and it has to hold against the reference as well
    if model.errorbound > TOLERANCE:
        sys.exit("Validation test failed at case 1: for the %s generator the class reports an error bound of %.3e, above the tolerance of %.1e it was given." % (name, model.errorbound, TOLERANCE))


# ------------------------------------------------------------------
# CASE 2: the nrows argument
# ------------------------------------------------------------------

# Asking for the leading rows has to give the same values as taking those rows
# from the full result, since the two are computed by different routes: the
# whole matrix is propagated in one and only the requested rows in the other.
for name, gen in GENERATORS:
    model = _unif(tolerance=TOLERANCE)
    nphases = gen.shape[0]

    full = model.run(gen, TIMES)

    for nrows in range(1, nphases + 1):
        leading = model.run(gen, TIMES, nrows=nrows)

        if leading.shape != (TIMES.size, nrows, nphases):
            sys.exit("Validation test failed at case 2: asking for %d rows of the %s generator returns an array of shape %s." % (nrows, name, str(leading.shape)))

        difference = float(np.max(np.abs(leading - full[:, :nrows, :])))

        if difference > TOL:
            sys.exit("Validation test failed at case 2: the leading %d rows of the %s generator differ by %.3e from the same rows of the full result." % (nrows, name, difference))


# ------------------------------------------------------------------
# CASE 3: the underflow regime
# ------------------------------------------------------------------

# The Poisson weights are exp(-Gamma*y) (Gamma*y)^k / k!. Formed directly, the
# first factor underflows to zero for Gamma*y above about 745 and the second
# overflows well before that, so the product is unusable exactly where the
# series is longest. The class forms them in logarithms instead. This case puts
# it in that regime and checks the answer is still right.

gamma = float(np.max(np.abs(np.diag(SPREADGEN))))
naiveweight = np.exp(-gamma * UNDERFLOWTIME)

# the case is only worth running if the direct formula really does fail here
if naiveweight != 0.0:
    sys.exit("Validation test failed at case 3: exp(-Gamma*y) is %.3e at the chosen time, so it has not underflowed and the case is not testing what it is named for." % naiveweight)

model = _unif(tolerance=TOLERANCE)
computed = model.run(SPREADGEN, np.array([UNDERFLOWTIME]))
reference = expm(SPREADGEN * UNDERFLOWTIME)

# and only worth running if the answer is far from zero, or it would amount to
# comparing one underflowed result with another
if float(np.max(np.abs(reference))) < 1e-6:
    sys.exit("Validation test failed at case 3: exp(Ty) has decayed to %.3e at the chosen time, so the case would compare one vanishing result with another." % float(np.max(np.abs(reference))))

if not np.all(np.isfinite(computed)):
    sys.exit("Validation test failed at case 3: exp(Ty) is not a number where exp(-Gamma*y) underflows, so the weights are being formed directly rather than in logarithms.")

difference = float(np.max(np.abs(computed[0] - reference)))

if difference > TOL:
    sys.exit("Validation test failed at case 3: where exp(-Gamma*y) underflows, uniformization and expm differ by %.3e, which is more than %.1e." % (difference, TOL))


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print("unif: All tests passed.")
