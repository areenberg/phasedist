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
    Case 4: vectorrun() gives left exp(Ty) right as run() contracted with the
            same vectors, within the error bound it reports, for vector,
            matrix and absent right-hand sides.
    Case 5: the compiled kernels against the numpy path: The same truncation
            point, error bound and norm, and the same values to rounding, for
            run() (all rows and leading rows) and vectorrun(), on generators,
            a Van Loan block whose norm has to be measured, a fast phase with a
            long series, a slow phase whose last terms still count, and the
            time zero, at the tolerance of the other cases and a coarse one.
    Case 6: the Poisson weights of the compiled kernels against 50-digit
            decimal arithmetic. (scipy's poisson.pmf is no oracle here: Its
            error grows to 1e-11 at a parameter of 25000.)

Cases 5 and 6 need the compiled kernels. Without them (a clone that has not
been built, or PHASEDIST_PUREPYTHON set) they are not run and the final line
says so; with PHASEDIST_REQUIRE_KERNEL set, as in CI, that is a failure.

References:
    Stewart, W. J. Introduction to the Numerical Solution of Markov Chains.
    Princeton University Press. https://www.jstor.org/stable/j.ctvcm4gtc
'''

import decimal
import itertools
import math
import os
import sys
import numpy as np
from scipy.linalg import expm

# Load phasedist from the src-folder so the test can be run without installing
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

import phasedist.unif as unifmodule
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

# Cases 4 and 5: an initial distribution, and the times of the comparisons,
# from zero to well into the tail, so that the windows of the compiled kernels
# range from a single term to the full series.
INITDIST = {"dense": np.array([0.5, 0.3, 0.2]),
            "Coxian": np.array([1.0, 0.0, 0.0]),
            "loop": np.array([0.1, 0.4, 0.3, 0.2])}
VECTORTIMES = np.array([0.0, 1e-9, 0.05, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 60.0])

# Case 5: a fast phase multiplies Gamma, so the series runs to hundreds of terms
# and each time's window is a small part of it.
FASTFACTOR = 40.0

# Case 5: the compiled kernels and numpy sum the same series in a different
# order and with differently formed weights, so they agree to rounding only.
# The comparison is made at the tolerance of the other cases and at a coarse
# one, under which the last terms of the series weigh far more than rounding,
# so that a term lost at the end of a window or of the series shows.
TOLKERNEL = 1e-13
KERNELTOLERANCES = (TOLERANCE, 1e-6)

# Case 6: Poisson parameters and terms where the weights are formed: The
# mode, both tails, the table of Stirling's error (below 16) and its series.
POISSONCASES = [(1e-6, 0), (0.3, 0), (0.3, 2), (1.0, 1), (2.5, 2), (7.0, 7), (7.0, 20),
                (15.5, 15), (16.0, 16), (35.7, 35), (80.2, 80), (80.2, 40), (499.9, 499),
                (500.5, 500), (1000.0, 1000), (1000.0, 1150), (25000.0, 25000)]
TOLPOISSON = 1e-14    # the weights are within 3e-15 of the reference here


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
# CASE 4: vectorrun() against run()
# ------------------------------------------------------------------

def vanloan(gen, initdist):
    """The block [[T, e pi], [0, T]] of a censored E-step, whose P is not sub-stochastic."""
    nphases = gen.shape[0]
    return np.block([[gen, np.outer(np.ones(nphases), initdist)],
                     [np.zeros((nphases, nphases)), gen]])


for name, gen in GENERATORS:
    model = _unif(tolerance=TOLERANCE)
    initdist = INITDIST[name]
    nphases = gen.shape[0]
    exitrates = -np.sum(gen, axis=1)

    full = model.run(gen, VECTORTIMES)
    rows = np.einsum("k,mkl->ml", initdist, full)

    for label, right in (("exit rates", exitrates), ("ones", np.ones(nphases)),
                         ("two columns", np.c_[np.ones(nphases), exitrates]), ("none", None)):
        computed = model.vectorrun(gen, VECTORTIMES, initdist, right)
        reference = rows if right is None else rows @ right

        if computed.shape != reference.shape:
            sys.exit("Validation test failed at case 4: vectorrun() with right-hand side '%s' of the %s generator returns shape %s, where run() contracted gives %s." % (label, name, str(computed.shape), str(reference.shape)))

        # the reported bound, scaled by the norms of the vectors, plus rounding
        scale = 1.0 if right is None else float(np.max(np.abs(right)))
        allowed = model.errorbound * float(np.sum(np.abs(initdist))) * scale + TOL
        difference = float(np.max(np.abs(computed - reference)))

        if not np.all(np.isfinite(computed)) or difference > allowed:
            sys.exit("Validation test failed at case 4: vectorrun() with right-hand side '%s' of the %s generator differs from run() contracted by %.3e, which is more than %.1e." % (label, name, difference, allowed))

    # the zero time gives the product of the two vectors, to rounding
    atzero = float(model.vectorrun(gen, np.array([0.0]), initdist, exitrates)[0])
    if abs(atzero - float(initdist @ exitrates)) > 1e-14 * abs(float(initdist @ exitrates)):
        sys.exit("Validation test failed at case 4: vectorrun() of the %s generator at y = 0 is %.17g, not the product of the two vectors, %.17g." % (name, atzero, float(initdist @ exitrates)))


# ------------------------------------------------------------------
# CASE 5: the compiled kernels against the numpy path
# ------------------------------------------------------------------

def bothpaths(call, tolerance):
    """Runs a call with the compiled kernels and with numpy, with the attributes."""
    out = []
    for kernel in (compiled, None):
        unifmodule._kernel = kernel
        model = _unif(tolerance=tolerance)
        values = call(model)
        out.append((values, (model.nterms, model.errorbound, model.rho, model.gamma)))
    unifmodule._kernel = compiled
    return out


compiled = unifmodule._kernel
required = bool(os.environ.get("PHASEDIST_REQUIRE_KERNEL"))

if compiled is None and required:
    sys.exit("Validation test failed at case 5: the compiled kernels are required (PHASEDIST_REQUIRE_KERNEL is set) but phasedist._unifkernel was not loaded.")

if compiled is not None:
    fastgen = DENSEGEN.copy()
    fastgen[0] *= FASTFACTOR
    matrices = [(name, gen, INITDIST[name]) for name, gen in GENERATORS]
    matrices.append(("fast-phase", fastgen, INITDIST["dense"]))
    # its slow phase keeps the powers of P from decaying, so the last terms of
    # the series still weigh something at the coarse tolerance
    matrices.append(("spread", SPREADGEN, np.array([0.6, 0.4])))
    matrices.append(("Van Loan", vanloan(LOOPGEN, INITDIST["loop"]), np.r_[INITDIST["loop"], np.zeros(4)]))

    # the case has to cover a measured norm and a long series to mean anything
    probe = _unif(tolerance=TOLERANCE)
    probe.run(matrices[-1][1], VECTORTIMES)
    if not probe.rho > 1.0:
        sys.exit("Validation test failed at case 5: the norm of the Van Loan block is not measured (rho = %s), so the measured path is not being compared." % probe.rho)
    probe.run(fastgen, VECTORTIMES)
    if probe.nterms < 500:
        sys.exit("Validation test failed at case 5: the fast-phase series has only %d terms, so the windows are not being compared on a long series." % probe.nterms)

    probe = _unif(tolerance=KERNELTOLERANCES[-1])
    probe.run(SPREADGEN, VECTORTIMES)
    tail = float(np.max(np.abs(np.linalg.matrix_power(SPREADGEN / probe.gamma + np.eye(2), probe.nterms))))
    if tail < 1e-3:
        sys.exit("Validation test failed at case 5: the last power of the spread series has decayed to %.3e, so a term lost at the end of the series would not show." % tail)

    for name, matrix, left in matrices:
        n = matrix.shape[0]
        # the time zero is P^0 in both, exactly, except in a product with a
        # right-hand side, which is summed in a different order
        calls = [("run()", True, lambda model: model.run(matrix, VECTORTIMES)),
                 ("run() with one row", True, lambda model: model.run(matrix, VECTORTIMES, nrows=1)),
                 ("vectorrun() with a vector", False, lambda model: model.vectorrun(matrix, VECTORTIMES, left, -np.sum(matrix, axis=1))),
                 ("vectorrun() with two columns", False, lambda model: model.vectorrun(matrix, VECTORTIMES, left, np.c_[np.ones(n), -np.sum(matrix, axis=1)])),
                 ("vectorrun() without right", True, lambda model: model.vectorrun(matrix, VECTORTIMES, left))]

        for (label, exactatzero, call), tolerance in itertools.product(calls, KERNELTOLERANCES):
            (kernelvalues, kernelstate), (numpyvalues, numpystate) = bothpaths(call, tolerance)
            label = "%s at tolerance %.0e" % (label, tolerance)

            # the bound depends on the measured norm, which the two paths sum
            # in a different order, so it agrees to rounding only
            if kernelstate[0] != numpystate[0] or abs(kernelstate[1] - numpystate[1]) > TOLKERNEL * numpystate[1]:
                sys.exit("Validation test failed at case 5: %s of the %s matrix uses %d terms with a bound of %.15g compiled, and %d terms with %.15g in numpy." % (label, name, kernelstate[0], kernelstate[1], numpystate[0], numpystate[1]))

            if abs(kernelstate[2] - numpystate[2]) > TOLKERNEL * numpystate[2]:
                sys.exit("Validation test failed at case 5: %s of the %s matrix measures the norm %.15g compiled and %.15g in numpy." % (label, name, kernelstate[2], numpystate[2]))

            if kernelvalues.shape != numpyvalues.shape:
                sys.exit("Validation test failed at case 5: %s of the %s matrix returns shape %s compiled and %s in numpy." % (label, name, str(kernelvalues.shape), str(numpyvalues.shape)))

            # rounding, and the terms the windows leave out: At most a
            # millionth of the tolerance, scaled as the vectors scale it
            scale = float(np.sum(np.abs(left))) * float(np.max(np.abs(np.sum(matrix, axis=1)))) if "vectorrun" in label else 1.0
            allowed = TOLKERNEL + 1e-6 * tolerance * kernelstate[2] * max(scale, 1.0)
            difference = float(np.max(np.abs(kernelvalues - numpyvalues)))
            if os.environ.get("CALIBRATE"):
                print("%-60s %-12s %.2e %.2e" % (label, name, difference, allowed))
            if not np.all(np.isfinite(kernelvalues)) or difference > allowed:
                sys.exit("Validation test failed at case 5: %s of the %s matrix differs by %.3e between the compiled kernels and numpy, which is more than %.1e." % (label, name, difference, allowed))

            if exactatzero and np.any(kernelvalues[0] != numpyvalues[0]):
                sys.exit("Validation test failed at case 5: %s of the %s matrix at y = 0 is not the same compiled and in numpy." % (label, name))


# ------------------------------------------------------------------
# CASE 6: the Poisson weights of the compiled kernels
# ------------------------------------------------------------------

def poissonreference(k, mu):
    """exp(-mu) mu^k / k! in 50-digit decimal arithmetic, mu taken exactly."""
    with decimal.localcontext() as context:
        context.prec = 50
        exact = decimal.Decimal(mu)
        return float((-exact).exp() * exact ** k / decimal.Decimal(math.factorial(k)))


if compiled is not None:
    for mu, k in POISSONCASES:
        reference = poissonreference(k, mu)
        weight = float(compiled.poisson(k, mu))

        if not reference > 0.0:
            sys.exit("Validation test failed at case 6: the reference weight at mu = %s, k = %d is %s, so the case compares nothing." % (mu, k, reference))

        if abs(weight - reference) > TOLPOISSON * reference:
            sys.exit("Validation test failed at case 6: the Poisson weight at mu = %s, k = %d is %.17g, where the reference is %.17g." % (mu, k, weight, reference))


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

if compiled is None:
    print("unif: All tests passed. (The compiled kernels are not loaded, so cases 5 and 6 were not run.)")
else:
    print("unif: All tests passed.")
