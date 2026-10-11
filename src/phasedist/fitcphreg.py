import time
import warnings

import numpy as np
from phasedist.mcph import mcph
from phasedist.rndcph import rndcph
from phasedist.unif import _unif
from phasedist.regtools import _stoprule, _aghq


class _fitcphreg:
    """
    Fits a continuous-time phase-type regression with proportional
    intensities, Bladt (2022), optionally with a Gaussian random intercept
    per group.

    Given covariates x and the random intercept b of its group, an observation
    is the absorption time of a Markov jump process whose every intensity is
    multiplied by exp(eta), eta = x'beta + b:

        Y | x, b ~ PH(pi, exp(eta) T),

    so that U = exp(eta) Y ~ PH(pi, T). The design matrix carries no
    intercept: the intercept is the scale of T. Only the homogeneous model is
    fitted, as in fitcph; the inhomogeneous variants of Bladt (2022) are left
    out.

    The parameters are estimated by an ECM algorithm (Meng and Rubin, 1993).
    The E-step is the one of p. 678 in Bladt and Nielsen (2017) on the
    transformed times U, with the Van Loan block matrix evaluated by
    uniformization. The conditional M-steps are the closed-form phase-type
    M-step for (pi, T) (mcph), a Poisson regression for beta, and, with a
    random intercept, a closed form for its variance. A random intercept is
    integrated out by adaptive Gauss-Hermite quadrature: the
    nodes are adapted for at least 30 iterations, then frozen so that the EM
    runs on a fixed objective, and re-checked after convergence. Before that,
    the derivative of the likelihood in sigma^2 at zero is evaluated at the
    fixed-effects fit (Verbeke and Molenberghs, 2003); where it is not
    positive, sigma = 0 is the estimate and the fit ends there, instead of an
    EM crawling towards the boundary.

    Starting values: beta from the quasi-likelihood equations of a Gamma
    regression (Gardner, 2024, Chapter 4), (pi, T) from an ordered preset for
    the chain structures and `restarts` random structures (rndcph), every
    start run to convergence. A fit that ends with a phase holding a
    negligible share of the expected time is a lower-order model in disguise
    that EM cannot leave, and the best other start is used instead.

    Convergence is assessed by the windowed Aitken rule with a confirmation
    block (_stoprule).

    Warning: This class does not contain any input checks.

    References:
        Bladt, M. (2022). Phase-type distributions for claim severity regression modeling.
        ASTIN Bulletin, 52(2), 417-448. https://doi.org/10.1017/asb.2021.40

        Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in Applied Probability.
        Springer. https://doi.org/10.1007/978-1-4939-7049-0

        Gardner, C. B. (2024). Multivariate Phase-type Distributions in Stochastic Modelling.
        PhD thesis, Technical University of Denmark.

        Meng, X.-L., & Rubin, D. B. (1993). Maximum likelihood estimation via the ECM
        algorithm. Biometrika, 80(2), 267-278. https://doi.org/10.1093/biomet/80.2.267

        Verbeke, G., & Molenberghs, G. (2003). The use of score tests for inference on
        variance components. Biometrics, 59(2), 254-262.
        https://doi.org/10.1111/1541-0420.00032
    """

    # the share of the expected time below which a phase counts as unused,
    # and the number of iterations a start may stay so before it is abandoned
    UNUSEDSHARE = 1e-6
    UNUSEDPATIENCE = 20
    # a start is given up once, after at least HOPELESSITER iterations, even
    # its projected limit lies more than HOPELESSGAP below the best start so
    # far
    HOPELESSITER = 1000
    HOPELESSGAP = 1.0

    # the node schedule of the random intercept
    MINADAPT = 30
    MAXADAPT = 1000
    ADAPTCHANGE = 1e-3
    MAXREADAPT = 20
    # the confirmation block after a re-adaptation, which starts from a fit
    # that has already passed the full block on nearly the same objective
    READAPTCONFIRM = 30
    # the largest Poisson parameter (rate times time) handed to _unif; longer
    # times are halved until below it and the result squared back
    MAXPOISSON = 1000.0
    # an extrapolated SQUAREM candidate whose largest Poisson parameter (rate
    # times transformed time) exceeds this multiple of the plain iterate's is
    # rejected without being evaluated
    SQUAREMSCALE = 10.0

    def __init__(
        self,
        obs: np.array = None,
        design: np.array = None,
        groups: np.array = None,
        initpi: np.array = None,
        initphgen: np.array = None,
        initexitrates: np.array = None,
        quadnodes: int = 7,
        restarts: int = 10,
        seed: int = None,
        tolerance: float = 1e-5,
        itermax: int = 100000,
        accelerate: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the phase-type regression fitter.

        Args:
            obs (ndarray): Positive observations, sorted by group when there
                are groups.
            design (ndarray): Design matrix without intercept column, one row
                per observation, centred column by column.
            groups (ndarray, optional): Group code 0, ..., J-1 of each
                observation, in non-decreasing order. None for a model
                without random intercept.
            initpi (ndarray): Structure of the initial distribution.
            initphgen (ndarray): Structure of the phase-type generator.
            initexitrates (ndarray): Structure of the exit-rate vector.
            quadnodes (int, default=7): Quadrature nodes per group.
            restarts (int, default=10): Number of random starts.
            seed (int): Random seed for reproducibility.
            tolerance (float, default=1e-5): Convergence tolerance.
            itermax (int, default=100000): Maximum number of iterations of each
                run to convergence.
            accelerate (bool, default=True): Whether to accelerate the
                iterations on a fixed objective by SQUAREM.
            verbose (bool, default=False): Whether to print intermediate
                fitting information.
        """

        self.obs = np.asarray(obs, dtype=float)
        self.X = np.asarray(design, dtype=float).reshape(self.obs.size, -1)
        self.groups = None if groups is None else np.asarray(groups, dtype=int)
        self.initpi = np.asarray(initpi, dtype=float).reshape(-1)
        self.initphgen = np.asarray(initphgen, dtype=float)
        self.initexitrates = np.asarray(initexitrates, dtype=float).reshape(-1)
        self.nphases = self.initphgen.shape[0]
        self.quadnodes = quadnodes
        self.restarts = restarts
        self.seed = seed
        self.tolerance = tolerance
        self.itermax = itermax
        self.accelerate = accelerate
        self.verbose = verbose

        self.__initialize()

        return None

    def fit(self) -> None:
        """
        Fits the phase-type regression.

        Args:
            None

        Returns:
            None
        """

        if self.seed is not None:
            np.random.seed(self.seed)

        self.__fitfixed()

        if self.groups is not None and not self.failed:
            self.__fitrandom()

        if not self.failed:
            self.__finalize()

        return None

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS: SET-UP
    # ----------------------------------------------------------------------

    def __initialize(self) -> None:
        """
        Prepares the quantities that stay fixed during the fit.

        Args:
            None

        Returns:
            None
        """

        self.n, self.d = self.X.shape
        self.unif = _unif(tolerance=1e-14)
        self.mstep = mcph(nphases=self.nphases, nobs=self.n)

        # the generator entries that are free; the structural zeros stay zero
        # through every iteration and every extrapolation
        self.offmask = (self.initphgen != 0.0) & ~np.eye(self.nphases, dtype=bool)

        if self.groups is not None:
            self.starts = np.flatnonzero(np.r_[True, self.groups[1:] != self.groups[:-1]])
            self.ngroups = self.starts.size
            self.aghq = _aghq(self.quadnodes)
            self.rowindex = np.repeat(np.arange(self.n), self.quadnodes)
        else:
            self.ngroups = 0

        self.failed = False
        self.converged = False
        self.degenerate = False
        self.boundary = False
        self.boundaryscore = None
        self.readapts = 0
        self.iterations = 0
        self.loglikelihood = -np.inf

        return None

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS: MATRIX EXPONENTIALS AND THE E-STEP
    # ----------------------------------------------------------------------

    def __expm(self, matrix: np.array, times: np.array, nrows: int = None) -> np.array:
        """
        Returns exp(matrix * y), or its leading rows, for every y, evaluated by
        uniformization. The cost is the Poisson weights, times by terms, and
        the number of terms is set by the longest time of a call. The times
        are therefore sorted and processed in chunks whose Poisson parameter
        (the uniformization rate times y) spans at most a factor of two, or
        32 at the short end, so that each time gets about the terms it needs
        rather than those of the longest time. A chunk is split further to
        keep its weights within a fixed memory budget.

        Uniformization needs about as many terms as the Poisson parameter, so
        a stiff generator (one fast phase) makes it slow: in one
        random-intercept fit the ECM moved a first phase to rate 2,400 to
        model the shortest times, the Poisson parameter reached 50,000, and an
        iteration took seconds. For a chunk whose Poisson parameter exceeds
        MAXPOISSON the times are halved s times until it does not,
        exp(matrix y / 2^s) is evaluated by uniformization, and squared s
        times: exp(matrix y) = exp(matrix y / 2^s)^(2^s).

        Args:
            matrix (ndarray): A sub-intensity matrix.
            times (ndarray): Non-negative times.
            nrows (int, optional): Number of leading rows to return.

        Returns:
            ndarray: Shape (len(times), nrows, n).
        """

        times = np.asarray(times, dtype=float).reshape(-1)
        n = matrix.shape[0]
        nrows = n if nrows is None else nrows
        out = np.empty((times.size, nrows, n))

        if times.size == 0:
            return out

        order = np.argsort(times)
        ordered = times[order]
        gamma = float(np.max(np.abs(np.diag(matrix))))
        mus = gamma * ordered
        budget = 4e6

        start = 0
        while start < times.size:
            limit = max(2.0 * mus[start], mus[start] + 32.0)
            stop = max(int(np.searchsorted(mus, limit, side="right")), start + 1)
            terms = mus[stop - 1] + 6.0 * np.sqrt(mus[stop - 1]) + 20.0
            stop = min(stop, start + max(16, int(budget / (terms + 1.0))))
            if mus[stop - 1] <= self.MAXPOISSON:
                out[order[start:stop]] = self.unif.run(matrix, ordered[start:stop], nrows=nrows)
            else:
                halvings = int(np.ceil(np.log2(mus[stop - 1] / self.MAXPOISSON)))
                block = self.unif.run(matrix, ordered[start:stop] / 2.0 ** halvings)
                for _ in range(halvings):
                    block = block @ block
                out[order[start:stop]] = block[:, :nrows, :]
            start = stop

        return out

    def __rowexp(self, pi: np.array, T: np.array, u: np.array) -> np.array:
        """
        Returns pi exp(T u) for every u, one row each.

        Args:
            pi (ndarray): Initial distribution.
            T (ndarray): Sub-intensity matrix.
            u (ndarray): Transformed times.

        Returns:
            ndarray: Shape (len(u), nphases).
        """

        return np.einsum("k,mkl->ml", pi, self.__expm(T, u))

    def __logdensity(self, s: dict, eta: np.array, y: np.array) -> np.array:
        """
        Returns the log-density of each row, eta + log(pi exp(e^eta y T) t).

        Args:
            s (dict): The parameters.
            eta (ndarray): Linear predictor of each row.
            y (ndarray): Observation of each row.

        Returns:
            ndarray: The log-densities.
        """

        u = np.exp(eta) * y
        f0 = self.__rowexp(s["pi"], s["T"], u) @ s["t"]
        with np.errstate(divide="ignore", invalid="ignore"):
            return eta + np.log(f0)

    def __derivs(self, s: dict, eta: np.array, y: np.array) -> tuple:
        """
        Returns the first and second derivatives of each row's log-density in
        eta. With a = pi exp(uT), f_k = a T^k t and u = e^eta y,

            l'  = 1 + u f1/f0,
            l'' = u f1/f0 + u^2 (f2/f0 - (f1/f0)^2),

        T commuting with exp(uT).

        Args:
            s (dict): The parameters.
            eta (ndarray): Linear predictor of each row.
            y (ndarray): Observation of each row.

        Returns:
            tuple: l' (ndarray) and l'' (ndarray).
        """

        T, t = s["T"], s["t"]
        u = np.exp(eta) * y
        a = self.__rowexp(s["pi"], T, u)
        Tt = T @ t
        f0 = a @ t
        with np.errstate(divide="ignore", invalid="ignore"):
            r1 = (a @ Tt) / f0
            r2 = (a @ (T @ Tt)) / f0
            return 1.0 + u * r1, u * r1 + u ** 2 * (r2 - r1 ** 2)

    def __estep(self, s: dict, u: np.array) -> dict:
        """
        Performs the E-step on the transformed times u, one row each.
        Exponentiated over u, the block matrix [[T, t pi],[0, T]] has top-left
        block exp(Tu) and top-right block J(u), Theorem A.2.1 (Van Loan).

        Returns per row the density of U and the quantities the conditional
        expectations are built from; __weigh() turns them into the weighted
        sums of the M-step. The two steps are separate because, with a random
        intercept, the weights depend on the densities computed here. A row
        whose density underflows gets zero statistics; this only happens at
        quadrature nodes of negligible weight.

        Args:
            s (dict): The parameters.
            u (ndarray): Transformed times.

        Returns:
            dict: Per row f0, 1/f0, pi exp(Tu), exp(Tu) t and J(u).
        """

        p = self.nphases
        pi, T, t = s["pi"], s["T"], s["t"]

        block = np.block([[T, np.outer(t, pi)],
                          [np.zeros((p, p)), T]])
        mats = self.__expm(block, u, nrows=p)
        eTu = mats[:, :, :p]

        a = np.einsum("k,mkl->ml", pi, eTu)
        f0 = a @ t
        ok = f0 > 0.0

        return dict(f0=f0,
                    invf=np.where(ok, 1.0 / np.where(ok, f0, 1.0), 0.0),
                    a=a,
                    eTut=eTu @ t,
                    J=mats[:, :, p:])

    def __weigh(self, s: dict, e: dict, weights: np.array) -> dict:
        """
        Returns the weighted sums of the conditional expectations that the
        M-step needs, and per row the two inputs of the beta step: the
        expected number of jumps, the absorption included, and the expected
        time in each phase on the transformed clock u = exp(eta) y.

        Args:
            s (dict): The parameters.
            e (dict): The output of __estep().
            weights (ndarray): Weight of each row.

        Returns:
            dict: bi, zi, ni, nij (sums), jumps and zrow (per row).
        """

        pi, T, t = s["pi"], s["T"], s["t"]
        invf = e["invf"]
        inv = weights * invf
        J = e["J"]

        offdiagonal = np.array(T, dtype=float)
        np.fill_diagonal(offdiagonal, 0.0)

        Jdiag = np.diagonal(J, axis1=1, axis2=2)

        out = dict()
        out["bi"] = pi * (inv @ e["eTut"])
        out["zi"] = inv @ Jdiag
        out["ni"] = t * (inv @ e["a"])
        # element k,s is t_ks times the weighted sum of J[s,k], i.e. the
        # generator against the transpose of the J-matrix
        out["nij"] = offdiagonal * np.einsum("m,msk->ks", inv, J)
        out["jumps"] = 1.0 + np.einsum("ks,msk->m", offdiagonal, J) * invf
        out["zrow"] = Jdiag * invf[:, None]

        return out

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS: THE CONDITIONAL M-STEPS
    # ----------------------------------------------------------------------

    def __betastep(self, rows: np.array, jumps: np.array, exposure: np.array,
                   beta: np.array, offset) -> np.array:
        """
        Maximizes the part of the expected complete-data log-likelihood that
        depends on beta,

            Q(beta) = sum m (x'beta + o) - exp(x'beta + o) w,

        a Poisson log-likelihood with pseudo-counts m (the expected jumps) and
        exposures w. Q is concave, so Newton's method with step halving on Q
        converges in a few steps, and no matrix exponential is needed.

        Args:
            rows (ndarray): Design rows.
            jumps (ndarray): Pseudo-counts, weighted.
            exposure (ndarray): Exposures, weighted.
            beta (ndarray): Current coefficients.
            offset (ndarray or float): Offset of each row.

        Returns:
            ndarray: The updated coefficients.
        """

        if self.d == 0:
            return beta

        def objective(b):
            with np.errstate(over="ignore", invalid="ignore"):
                e = rows @ b + offset
                value = np.sum(jumps * e - np.exp(e) * exposure)
            return value if np.isfinite(value) else -np.inf

        current = objective(beta)

        for _ in range(50):
            with np.errstate(over="ignore"):
                mu = np.exp(rows @ beta + offset) * exposure
            gradient = rows.T @ (jumps - mu)
            hessian = (rows * mu[:, None]).T @ rows
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                break

            scale = 1.0
            while True:
                candidate = beta + scale * step
                value = objective(candidate)
                if value >= current - 1e-12 * max(1.0, abs(current)) or scale < 1e-8:
                    break
                scale /= 2.0

            if not value >= current - 1e-12 * max(1.0, abs(current)):
                break

            beta, current = candidate, value

            if np.max(np.abs(scale * step)) < 1e-10:
                break

        return beta

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS: ACCELERATION
    # ----------------------------------------------------------------------

    def __pack(self, s: dict) -> np.array:
        """
        Returns the parameters as one vector: the initial distribution, the
        exit rates, the off-diagonal generator entries, beta, and sigma where
        the model has it. The generator's diagonal is left out
        since it is minus the rest of its row.

        Args:
            s (dict): The parameters.

        Returns:
            ndarray: The parameters as one vector.
        """

        parts = [s["pi"], s["t"], s["T"][self.offmask], s["beta"]]
        if s.get("sigma") is not None:
            parts.append([s["sigma"]])

        return np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in parts])

    def __unpack(self, values: np.array, template: dict) -> dict:
        """
        Returns the parameters held in a vector from __pack, rebuilding the
        generator's diagonal and normalizing the initial distribution.

        Args:
            values (ndarray): The parameters as one vector.
            template (dict): Parameters of the same model, telling whether
                sigma is present.

        Returns:
            dict: The parameters.
        """

        p = self.nphases
        k = 0
        s = dict(template)
        s["pi"] = values[k:k + p] / np.sum(values[k:k + p])
        k += p
        s["t"] = np.copy(values[k:k + p])
        k += p
        T = np.zeros((p, p))
        noff = int(np.sum(self.offmask))
        T[self.offmask] = values[k:k + noff]
        k += noff
        np.fill_diagonal(T, -(T.sum(axis=1) + s["t"]))
        s["T"] = T
        s["beta"] = np.copy(values[k:k + self.d])
        k += self.d
        if template.get("sigma") is not None:
            s["sigma"] = float(values[k])

        return s

    def __isfeasible(self, values: np.array, template: dict) -> bool:
        """
        Returns whether a vector of parameters describes a model: rates and
        probabilities non-negative, every phase possible to leave, and sigma
        positive. beta is unrestricted.

        Args:
            values (ndarray): The parameters as one vector.
            template (dict): Parameters of the same model.

        Returns:
            bool: True when the parameters are usable.
        """

        if not np.all(np.isfinite(values)):
            return False

        p = self.nphases
        noff = int(np.sum(self.offmask))
        phasepart = values[:2 * p + noff]

        if np.any(phasepart < 0.0) or np.sum(values[:p]) <= 0.0:
            return False

        offdiagonal = np.zeros((p, p))
        offdiagonal[self.offmask] = values[2 * p:2 * p + noff]
        if not np.all(values[p:2 * p] + offdiagonal.sum(axis=1) > 0.0):
            return False

        # sigma, where present, follows beta at the end
        tail = values[2 * p + noff + self.d:]

        return bool(np.all(tail > 0.0))

    def __poissonbound(self, s: dict) -> float:
        """
        Returns the largest Poisson parameter of the uniformization at s, the
        largest rate of T times the largest transformed time exp(x'beta) y,
        random intercepts left out. The cost of a
        likelihood evaluation grows with it.

        Args:
            s (dict): The parameters.

        Returns:
            float: The bound.
        """

        with np.errstate(over="ignore"):
            u = np.max(np.exp(self.X @ s["beta"]) * self.obs) if self.d else np.max(self.obs)
            return float(np.max(np.abs(np.diag(s["T"]))) * u)

    def __squaremstep(self, s: dict, mapping, objective) -> tuple:
        """
        Performs one accelerated iteration, moving the parameter vector x to
        x - 2 alpha r + alpha^2 v for the step r of two iterations and its
        change v, see Varadhan and Roland (2008), as in fitcph2dist. At
        alpha = -1 this is the second iterate exactly, so a candidate that is
        infeasible or does not improve on it is rejected and backtracking ends
        at the plain algorithm: the likelihood stays monotone.

        When the change v is small against the step r, alpha is large and the
        first candidates can have rates or coefficients orders of magnitude
        beyond the iterates. Their likelihood is poor, but evaluating it
        costs time in proportion to the largest Poisson parameter of the
        uniformization, which in one fit reached 50,000 and kept a single
        backtracking search busy for minutes. A candidate whose bound exceeds
        SQUAREMSCALE times the plain iterate's is therefore rejected without
        evaluation, and one whose uniformization still reaches its term limit
        is rejected after it. Rejecting a candidate never lowers the
        likelihood.

        Args:
            s (dict): The parameters.
            mapping (callable): One plain iteration, returning the objective at
                its argument and the updated parameters (None on failure).
            objective (callable): The objective at a set of parameters.

        Returns:
            tuple: The objective at s (float) and the updated parameters
                (dict, None on failure).
        """

        loglik, first = mapping(s)
        if first is None:
            return loglik, None

        try:
            _, second = mapping(first)
        except (ValueError, np.linalg.LinAlgError):
            second = None
        if second is None:
            return loglik, first

        start = self.__pack(s)
        r = self.__pack(first) - start
        v = self.__pack(second) - self.__pack(first) - r
        lengthv = float(np.linalg.norm(v))

        if lengthv > 0.0:
            alpha = -float(np.linalg.norm(r)) / lengthv
            plain = objective(second)
            largest = self.SQUAREMSCALE * self.__poissonbound(second)

            while alpha < -1.001:
                candidate = start - 2.0 * alpha * r + alpha * alpha * v
                if self.__isfeasible(candidate, s):
                    trial = self.__unpack(candidate, s)
                    if not self.__poissonbound(trial) <= largest:
                        alpha = (alpha - 1.0) / 2.0
                        continue
                    try:
                        # a candidate whose likelihood cannot be evaluated to
                        # the uniformization's tolerance is rejected
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            value = objective(trial)
                        if any("maxterms" in str(w.message) for w in caught):
                            value = -np.inf
                        if np.isfinite(value) and value >= plain:
                            # one more iteration from the accepted candidate
                            _, stabilized = mapping(trial)
                            if stabilized is not None:
                                return loglik, stabilized
                            break
                    except (ValueError, np.linalg.LinAlgError):
                        pass
                alpha = (alpha - 1.0) / 2.0

        return loglik, second

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS: FIXED EFFECTS
    # ----------------------------------------------------------------------

    def __fixedloglik(self, s: dict) -> float:
        """
        Returns the log-likelihood of the fixed-effects model.

        Args:
            s (dict): The parameters.

        Returns:
            float: The log-likelihood.
        """

        return float(np.sum(self.__logdensity(s, self.X @ s["beta"], self.obs)))

    def __fixediteration(self, s: dict) -> tuple:
        """
        One ECM iteration of the fixed-effects model.

        Args:
            s (dict): The parameters.

        Returns:
            tuple: The log-likelihood at s (float) and the updated parameters
                (dict), the latter None if the likelihood is not finite.
        """

        eta = self.X @ s["beta"]
        u = np.exp(eta) * self.obs

        e = self.__estep(s, u)

        with np.errstate(divide="ignore", invalid="ignore"):
            loglik = float(np.sum(eta + np.log(e["f0"])))
        if not np.isfinite(loglik):
            return loglik, None

        e = self.__weigh(s, e, np.ones(self.n))

        new = dict(s)
        new["pi"], new["T"], new["t"] = self.mstep.run(bi=e["bi"], zi=e["zi"],
                                                       ni=e["ni"], nij=e["nij"])

        # the exposure is the expected occupation on the time axis of y,
        # weighted by the new rates: E(H_k) = E(Z_k) / exp(eta), Z being on
        # the time axis of u
        exposure = (e["zrow"] @ (-np.diag(new["T"]))) / np.exp(eta)
        new["beta"] = self.__betastep(self.X, e["jumps"], exposure, s["beta"], 0.0)

        return loglik, new

    def __quasilikelihood(self, y: np.array) -> tuple:
        """
        Returns the starting coefficients from the quasi-likelihood equations
        of a Gamma regression with log link, sum x (y/mu - 1) = 0, fitted with
        an intercept. The mean of the model scales as exp(-x'beta) and its
        variance as the square of the mean, so these are its quasi-likelihood
        equations (Gardner, 2024, Chapter 4); their root is unique, needs no
        phase-type quantities, and coincides with the maximum likelihood
        estimate of the one-phase model. The sign is flipped to the
        convention of the regression, where positive means shorter times.

        Args:
            y (ndarray): The observations.

        Returns:
            tuple: The coefficients (ndarray) and the fitted mean at x = 0
                (float).
        """

        design = np.column_stack([np.ones(self.n), self.X])
        b = np.zeros(self.d + 1)
        b[0] = np.log(np.mean(y))

        def objective(c):
            eta = design @ c
            with np.errstate(over="ignore"):
                return -np.sum(y * np.exp(-eta) + eta)

        current = objective(b)
        for _ in range(100):
            ratio = y * np.exp(-(design @ b))
            gradient = design.T @ (ratio - 1.0)
            hessian = design.T @ (design * ratio[:, None])
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                break
            scale = 1.0
            while objective(b + scale * step) < current and scale > 1e-8:
                scale /= 2.0
            b = b + scale * step
            current = objective(b)
            if np.max(np.abs(scale * step)) < 1e-12:
                break

        return -b[1:], float(np.exp(b[0]))

    def __ischain(self) -> bool:
        """
        Returns whether the structure is an ordered chain: every phase but the
        last may move to the next, and no phase may move back. Its phases are
        not exchangeable, so a deterministic preset is not a fixed point of EM
        there, unlike for the general and hyper-exponential structures.

        Args:
            None

        Returns:
            bool: True for the Coxian, generalized Coxian and generalized
                Erlang structures.
        """

        p = self.nphases
        if p == 1:
            return False

        forward = all(self.offmask[i, i + 1] for i in range(p - 1))
        backward = bool(np.any(np.tril(self.offmask, -1)))
        others = bool(np.any(np.triu(self.offmask, 2)))

        return forward and not backward and not others

    def __preset(self) -> tuple:
        """
        Returns a deterministic start for an ordered chain: distinct total
        rates, decreasing along the chain so the phases are not exchangeable,
        nine tenths of each rate moving forward where the phase may also exit,
        and the initial distribution spread evenly over its free entries.

        Args:
            None

        Returns:
            tuple: The initial distribution, the generator and the exit rates.
        """

        p = self.nphases
        rates = np.linspace(2.0, 0.5, p)
        pi = (self.initpi != 0.0).astype(float)
        pi = pi / np.sum(pi)
        T = np.zeros((p, p))
        t = np.zeros(p)

        for i in range(p):
            allowed = np.flatnonzero(self.offmask[i])
            canexit = self.initexitrates[i] != 0.0
            if allowed.size and canexit:
                T[i, allowed] = 0.9 * rates[i] / allowed.size
                t[i] = 0.1 * rates[i]
            elif allowed.size:
                T[i, allowed] = rates[i] / allowed.size
            else:
                t[i] = rates[i]

        np.fill_diagonal(T, -(T.sum(axis=1) + t))

        return pi, T, t

    def __startvalues(self) -> list:
        """
        Returns the starting values: the quasi-likelihood coefficients
        combined with phase-type representations of the requested structure,
        each scaled so that its mean matches the transformed data. The
        representations are a deterministic preset for an ordered chain and
        `restarts` random ones (rndcph). A symmetric preset is not used for the
        general or hyper-exponential structures, being a fixed point of EM
        there (design note, Section 5.3).

        Args:
            None

        Returns:
            list: One parameter dictionary per start.
        """

        beta, _ = self.__quasilikelihood(self.obs)
        target = np.mean(np.exp(self.X @ beta) * self.obs)

        representations = []
        if self.__ischain():
            representations.append(self.__preset())

        for _ in range(self.restarts):
            rnd = rndcph(nphases=self.nphases,
                         initdist=self.initpi,
                         phgen=self.initphgen,
                         exitrates=self.initexitrates)
            representations.append(rnd.run())

        starts = []
        for pi, T, t in representations:
            mean = float(pi @ np.linalg.solve(-T, np.ones(self.nphases)))
            factor = mean / target
            starts.append(dict(pi=np.array(pi, dtype=float), T=T * factor, t=t * factor,
                               beta=beta.copy(), sigma=None))

        return starts

    def __itermaxwarning(self, rule: _stoprule, iteration: int) -> None:
        """
        Warns that a run to convergence reached itermax, saying whether the
        stopping rule had already been satisfied and was being confirmed.

        Args:
            rule (_stoprule): The stopping rule of the run.
            iteration (int): Number of iterations made.

        Returns:
            None
        """

        if not np.isfinite(self.tolerance):
            return None

        if rule.confirmstart is not None:
            gain = rule.history[-1] - rule.history[rule.confirmstart]
            message = ("Algorithm terminated with iter==itermax while confirming "
                       "convergence: after %d iterations the estimated distance to the "
                       "limit of the log-likelihood was %.3e, and the gain over the "
                       "confirmation block so far %.3e, against a tolerance of %.3e."
                       % (iteration, float(rule.eps), gain, float(self.tolerance)))
        else:
            message = ("Algorithm terminated with iter==itermax. Results might be "
                       "misleading. After %d iterations the estimated distance to the "
                       "limit of the log-likelihood was still %.3e, against a "
                       "tolerance of %.3e." % (iteration, float(rule.eps), float(self.tolerance)))

        warnings.warn(message, RuntimeWarning, stacklevel=4)

        return None

    def __runfixed(self, s: dict, maxiter: int, final: bool, abandon: bool = False,
                   target: float = None) -> tuple:
        """
        Runs the fixed-effects ECM from s, for at most maxiter iterations or
        until the stopping rule is satisfied.

        With abandon, a run whose representation has had an unused phase for
        UNUSEDPATIENCE iterations is stopped as degenerate. Such a run cannot
        leave the lower-order model it has become, and its rates then grow
        without bound: in one run the unused phases' rates went from 100 to
        7,000 in 60 iterations, the uniformization needed ever more terms, and
        an iteration came to take seconds while the likelihood stayed put.

        Args:
            s (dict): Starting parameters.
            maxiter (int): Largest number of iterations.
            final (bool): Whether this is a run to convergence, which reports
                on reaching maxiter.
            abandon (bool, default=False): Whether to stop a degenerate run.
            target (float, optional): The log-likelihood of the best start so
                far. A run whose projected limit, the log-likelihood plus the
                stopping rule's estimate of the remaining gain, lies more than
                HOPELESSGAP below it after HOPELESSITER iterations is stopped
                as inferior: it cannot become the best start. In one data set
                a start crawled for more than 22,000 iterations two units below
                the best, gaining 3e-9 per iteration.

        Returns:
            tuple: The parameters (dict), their log-likelihood (float), the
                status ("converged", "itermax", "degenerate", "inferior" or
                "failed") and
                the number of iterations (int).
        """

        rule = _stoprule(tolerance=self.tolerance)
        previous = None
        status = "itermax"
        iteration = 0
        unusedfor = 0

        while iteration < maxiter:
            try:
                if self.accelerate:
                    loglik, new = self.__squaremstep(s, self.__fixediteration,
                                                     self.__fixedloglik)
                else:
                    loglik, new = self.__fixediteration(s)
            except (ValueError, np.linalg.LinAlgError):
                return s, -np.inf, "failed", iteration
            if new is None:
                return s, -np.inf, "failed", iteration

            self.__checkdecrease(previous, loglik, iteration)
            previous = loglik
            # an accelerated iteration costs about two plain ones, so it counts
            # twice and itermax keeps its meaning
            iteration += 2 if self.accelerate else 1
            s = new

            if self.verbose and final and iteration % 25 == 0:
                print("fixed effects: iter =", iteration, "  eps =", rule.eps,
                      "  loglik =", loglik)

            if abandon:
                unusedfor = unusedfor + (2 if self.accelerate else 1) if self.__unused(s) else 0
                if unusedfor >= self.UNUSEDPATIENCE:
                    status = "degenerate"
                    break

            if rule.update(loglik, 2 if self.accelerate else 1):
                status = "converged"
                break

            if (target is not None and iteration >= self.HOPELESSITER
                    and np.isfinite(rule.eps) and loglik + rule.eps < target - self.HOPELESSGAP):
                status = "inferior"
                break

        if final and status == "itermax":
            self.__itermaxwarning(rule, iteration)

        try:
            loglik = self.__fixedloglik(s)
        except (ValueError, np.linalg.LinAlgError):
            return s, -np.inf, "failed", iteration

        return s, loglik, status, iteration

    def __unused(self, s: dict) -> bool:
        """
        Returns whether some phase holds a negligible share of the expected
        time before absorption, pi (-T)^(-1). Such a fit is a lower-order
        model in disguise: a vanishing transition behaves like a structural
        zero, so EM cannot leave it.

        Args:
            s (dict): The parameters.

        Returns:
            bool: True if a phase is effectively unused.
        """

        if self.nphases == 1:
            return False

        occupation = np.linalg.solve(-s["T"].T, s["pi"])
        total = float(np.sum(occupation))

        return bool(np.any(occupation < self.UNUSEDSHARE * total))

    def __fitfixed(self) -> None:
        """
        Fits the fixed-effects model: runs every start to convergence and
        keeps the best fit without an unused phase. Every start is run to the
        end because the order of the starts after a short run need not be
        their order at convergence; a start that becomes degenerate is
        abandoned, and so is one that can no longer become the best (see
        __runfixed). If every start ends degenerate, the best of them is run to
        convergence and kept with a warning.

        Args:
            None

        Returns:
            None
        """

        best = None
        fallback = None
        self.startlogliks = []
        self.startiterations = []
        self.startseconds = []
        # per start: "converged", "itermax", "degenerate" (an unused phase),
        # "inferior" (given up below the best start) or "failed"
        self.startstatus = []

        for number, start in enumerate(self.__startvalues()):
            began = time.perf_counter()
            s, loglik, status, iterations = self.__runfixed(
                start, self.itermax, final=True, abandon=True,
                target=None if best is None else best[1])
            self.startlogliks.append(loglik)
            self.startiterations.append(iterations)
            self.startseconds.append(time.perf_counter() - began)
            unused = status == "degenerate" or (status != "failed" and np.isfinite(loglik)
                                                 and self.__unused(s))
            if status == "failed" or not np.isfinite(loglik):
                self.startstatus.append("failed")
            else:
                self.startstatus.append("degenerate" if unused else status)
            if self.verbose:
                print("start", number + 1, ": loglik =", loglik, " status =", self.startstatus[-1])
            if status in ("failed", "inferior") or not np.isfinite(loglik):
                continue
            if unused:
                if fallback is None or loglik > fallback[1]:
                    fallback = (s, loglik, status, iterations)
                continue
            if best is None or loglik > best[1]:
                best = (s, loglik, status, iterations)

        if best is None and fallback is None:
            self.failed = True
            return None

        if best is None:
            s, loglik, status, iterations = fallback
            if status == "degenerate":
                s, loglik, status, more = self.__runfixed(s, self.itermax, final=True)
                iterations += more
                if status == "failed" or not np.isfinite(loglik):
                    self.failed = True
                    return None
            best = (s, loglik, status, iterations)
            self.degenerate = True
            warnings.warn(
                "Every start ended with a phase that holds a negligible share of the "
                "expected time, so the fit is effectively of lower order. The number "
                "of phases is probably larger than the data support.",
                RuntimeWarning, stacklevel=3)

        self.state, self.loglikelihood, status, self.iterations = best
        self.converged = status == "converged"
        # the log-likelihood at sigma = 0, the reference of a likelihood-ratio
        # test of the random intercept
        self.fixedloglikelihood = self.loglikelihood

        return None

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS: RANDOM INTERCEPT
    # ----------------------------------------------------------------------

    def __modes(self, s: dict, start: np.array) -> tuple:
        """
        Returns the mode of each group's integrand, l_j(b) + log phi(b; 0,
        sigma^2), found by damped Newton steps on all groups at once, and the
        curvature scale (-H)^(-1/2) at the mode. Where the curvature is not
        negative (possible for a log-convex base density) the prior's
        curvature is used instead.

        Args:
            s (dict): The parameters.
            start (ndarray): Starting point of each group.

        Returns:
            tuple: The modes (ndarray), the scales (ndarray) and the number of
                groups that needed the fallback (int).
        """

        base = self.X @ s["beta"]
        variance = s["sigma"] ** 2
        b = np.array(start, dtype=float)

        for _ in range(50):
            d1, d2 = self.__derivs(s, base + b[self.groups], self.obs)
            gradient = np.add.reduceat(d1, self.starts) - b / variance
            hessian = np.add.reduceat(d2, self.starts) - 1.0 / variance
            hessian = np.where(hessian < 0.0, hessian, -1.0 / variance)
            step = -gradient / hessian
            step = np.where(np.isfinite(step), np.clip(step, -1.0, 1.0), 0.0)
            b = b + step
            if np.max(np.abs(step)) < 1e-10:
                break

        _, d2 = self.__derivs(s, base + b[self.groups], self.obs)
        hessian = np.add.reduceat(d2, self.starts) - 1.0 / variance
        bad = ~(hessian < 0.0)
        hessian = np.where(bad, -1.0 / variance, hessian)

        return b, np.sqrt(-1.0 / hessian), int(np.sum(bad))

    def __prior(self, nodes: np.array, sigma: float) -> np.array:
        """
        Returns log phi(b; 0, sigma^2) at the nodes.

        Args:
            nodes (ndarray): The nodes.
            sigma (float): The standard deviation of the random intercept.

        Returns:
            ndarray: The log-densities.
        """

        return -0.5 * nodes ** 2 / sigma ** 2 - 0.5 * np.log(2.0 * np.pi * sigma ** 2)

    def __frozenloglik(self, s: dict, nodes: np.array, scales: np.array) -> tuple:
        """
        Returns the marginal log-likelihood on the given nodes and the
        posterior node weights.

        Args:
            s (dict): The parameters.
            nodes (ndarray): One row of nodes per group.
            scales (ndarray): Their curvature scales.

        Returns:
            tuple: The log-likelihood (float) and the weights (ndarray).
        """

        R = nodes.shape[1]
        eta = (self.X @ s["beta"])[:, None] + nodes[self.groups]
        rows = self.__logdensity(s, eta.ravel(), np.repeat(self.obs, R)).reshape(self.n, R)
        logintegrand = np.add.reduceat(rows, self.starts, axis=0) + self.__prior(nodes, s["sigma"])
        logintegral, weights = self.aghq.weights(logintegrand, scales)

        return float(np.sum(logintegral)), weights

    def __randomiteration(self, s: dict, nodes: np.array, scales: np.array) -> tuple:
        """
        One ECM iteration of the random-intercept model on fixed nodes, the
        complete data being the node and the hidden path of each observation.

        Args:
            s (dict): The parameters.
            nodes (ndarray): One row of nodes per group.
            scales (ndarray): Their curvature scales.

        Returns:
            tuple: The marginal log-likelihood at s (float) and the updated
                parameters (dict), the latter None if the likelihood is not
                finite.
        """

        R = nodes.shape[1]
        nodesofrows = nodes[self.groups]
        eta = (self.X @ s["beta"])[:, None] + nodesofrows
        u = np.exp(eta) * self.obs[:, None]

        e = self.__estep(s, u.ravel())

        # the node weights are the posterior of the node given the group's
        # data, so they are only known once the densities are
        with np.errstate(divide="ignore", invalid="ignore"):
            rows = eta + np.log(e["f0"].reshape(self.n, R))
        logintegrand = np.add.reduceat(rows, self.starts, axis=0) + self.__prior(nodes, s["sigma"])
        logintegral, weights = self.aghq.weights(logintegrand, scales)
        loglik = float(np.sum(logintegral))

        if not np.isfinite(loglik):
            return loglik, None

        w = weights[self.groups].ravel()
        e = self.__weigh(s, e, w)

        new = dict(s)
        new["pi"], new["T"], new["t"] = self.mstep.run(bi=e["bi"], zi=e["zi"],
                                                       ni=e["ni"], nij=e["nij"])

        exposure = w * (e["zrow"] @ (-np.diag(new["T"]))) / np.exp(eta.ravel())
        new["beta"] = self.__betastep(self.X[self.rowindex], w * e["jumps"], exposure,
                                      s["beta"], nodesofrows.ravel())

        new["sigma"] = float(np.sqrt(np.sum(weights * nodes ** 2) / self.ngroups))

        return loglik, new

    def __change(self, old: dict, new: dict) -> float:
        """
        Returns the largest relative change of the identified parameters.

        Args:
            old (dict): Parameters before an iteration.
            new (dict): Parameters after it.

        Returns:
            float: The largest relative change.
        """

        change = abs(new["sigma"] - old["sigma"]) / max(old["sigma"], 1e-12)
        if self.d:
            change = max(change, float(np.max(np.abs(new["beta"] - old["beta"])
                                              / (1.0 + np.abs(old["beta"])))))

        return change

    def __boundaryscore(self, s: dict) -> float:
        """
        Returns the derivative of the marginal log-likelihood in sigma^2 at
        sigma = 0, the other parameters at s. With l_i the log-density of row
        i as a function of its linear predictor, group j has the likelihood
        E exp(sum_i l_i(eta_i + b)), b ~ N(0, sigma^2), which to first order
        in sigma^2 is exp(sum_i l_i(eta_i)) (1 + sigma^2 c_j / 2) with
        c_j = (sum_i l'_i)^2 + sum_i l''_i. The derivative is the sum of
        c_j / 2 over the groups. At the fixed-effects fit the gradient in the
        other parameters vanishes, so a negative derivative makes sigma = 0 a
        local maximum (Verbeke and Molenberghs, 2003); a derivative of exactly
        zero is treated the same.

        Args:
            s (dict): The parameters, normally the fixed-effects fit.

        Returns:
            float: The derivative.
        """

        d1, d2 = self.__derivs(s, self.X @ s["beta"], self.obs)

        return 0.5 * float(np.sum(np.add.reduceat(d1, self.starts) ** 2
                                  + np.add.reduceat(d2, self.starts)))

    def __atboundary(self, s: dict, converged: bool) -> None:
        """
        Ends the fit at sigma = 0 with the fixed-effects parameters s: the
        model is then the one without random intercept, and every group's
        predicted intercept is zero.

        Args:
            s (dict): The fixed-effects fit.
            converged (bool): Whether that fit converged.

        Returns:
            None
        """

        self.state = dict(s, sigma=0.0)
        self.boundary = True
        self.converged = converged
        warnings.warn(
            "The estimated standard deviation of the random intercept is zero: the "
            "likelihood does not increase as it moves away from zero, so the groups "
            "vary no more than the fixed effects explain. The fit is that of the "
            "model without random intercept.", RuntimeWarning, stacklevel=3)

        return None

    def __fitrandom(self) -> None:
        """
        Fits the random-intercept model from the fixed-effects fit. The nodes
        are adapted for at least MINADAPT iterations and until the parameters
        settle, then frozen; after convergence they are adapted once more, and
        the run resumes if that moves the log-likelihood by more than the
        tolerance. The stopping rule starts afresh whenever the nodes move,
        since the objective changes with them; after a re-adaptation with a
        confirmation block of READAPTCONFIRM iterations rather than the full
        one, the run starting from a fit that has passed the full block.

        Args:
            None

        Returns:
            None
        """

        s = dict(self.state)

        # sigma = 0 is the fixed-effects fit; where the likelihood does not
        # increase away from it, that is the estimate, and the ECM below would
        # only crawl towards it (design note, Section 6.6)
        self.boundaryscore = self.__boundaryscore(s)
        if not self.boundaryscore > 0.0:
            self.__atboundary(s, self.converged)
            return None

        s["sigma"] = float(np.sqrt(0.1))
        modes = np.zeros(self.ngroups)

        rule = _stoprule(tolerance=self.tolerance)
        adapting = True
        adapted = 0
        readapts = 0
        previous = None
        sigmas = []
        status = "itermax"
        iteration = 0

        while iteration < self.itermax:

            if adapting:
                modes, scales, _ = self.__modes(s, modes)
                nodes = self.aghq.nodes(modes, scales)

            try:
                if adapting or not self.accelerate:
                    loglik, new = self.__randomiteration(s, nodes, scales)
                else:
                    # the nodes are frozen, so the objective is fixed and the
                    # accelerated iteration keeps the likelihood monotone
                    loglik, new = self.__squaremstep(
                        s,
                        lambda x: self.__randomiteration(x, nodes, scales),
                        lambda x: self.__frozenloglik(x, nodes, scales)[0])
            except (ValueError, np.linalg.LinAlgError):
                new = None
            if new is None:
                self.failed = True
                return None

            if not adapting:
                self.__checkdecrease(previous, loglik, iteration)
            previous = loglik
            iteration += 2 if (self.accelerate and not adapting) else 1

            if adapting:
                adapted += 1
                settled = self.__change(s, new) < self.ADAPTCHANGE
                s = new
                if (adapted >= self.MINADAPT and settled) or adapted >= self.MAXADAPT:
                    adapting = False
                    rule.reset()
                    previous = None
                continue

            s = new
            sigmas.append(s["sigma"])

            if self.verbose and iteration % 25 == 0:
                print("random intercept: iter =", iteration, "  eps =", rule.eps,
                      "  loglik =", loglik, "  sigma =", s["sigma"])

            # a standard deviation heading for zero converges sublinearly, so
            # rather than crawl after it the fit moves to the boundary. The
            # score check above makes this rare: it takes a likelihood that
            # increases away from zero at the fixed-effects fit and yet has a
            # path to zero from the start of this run
            if (s["sigma"] < 1e-3 and len(sigmas) > 50 and s["sigma"] < sigmas[-51]):
                fixed = dict(s, sigma=None)
                fixed, _, status, more = self.__runfixed(fixed, self.itermax, final=True)
                self.iterations += iteration + more
                if status == "failed":
                    self.failed = True
                    return None
                self.__atboundary(fixed, status == "converged")
                return None

            if rule.update(loglik, 2 if self.accelerate else 1):
                newmodes, newscales, _ = self.__modes(s, modes)
                newnodes = self.aghq.nodes(newmodes, newscales)
                before = self.__frozenloglik(s, nodes, scales)[0]
                after = self.__frozenloglik(s, newnodes, newscales)[0]
                if abs(after - before) <= self.tolerance or readapts >= self.MAXREADAPT:
                    status = "converged"
                    break
                modes, scales, nodes = newmodes, newscales, newnodes
                readapts += 1
                rule = _stoprule(tolerance=self.tolerance, confirmmin=self.READAPTCONFIRM)
                previous = None

        if status == "itermax":
            self.__itermaxwarning(rule, iteration)

        self.state = s
        self.converged = status == "converged"
        self.iterations += iteration
        self.readapts = readapts

        return None

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS: RESULTS
    # ----------------------------------------------------------------------

    def __checkdecrease(self, previous: float, loglik: float, iteration: int) -> None:
        """
        Warns if the log-likelihood decreased by more than rounding, which an
        ECM iteration on a fixed objective cannot do.

        Args:
            previous (float): The log-likelihood of the iteration before, or None.
            loglik (float): The log-likelihood of this iteration.
            iteration (int): Index of this iteration.

        Returns:
            None
        """

        if previous is not None and loglik < previous - 1e-9 * max(1.0, abs(previous)):
            warnings.warn(
                "The log-likelihood decreased by %.3e at iteration %d."
                % (previous - loglik, iteration + 1),
                RuntimeWarning, stacklevel=4)

        return None

    def __rateparameters(self, s: dict) -> tuple:
        """
        Returns the free parameters of the phase-type representation at s on
        the log scale, and the function that rebuilds the parameters from
        them: the initial probabilities as log-ratios to the largest one, and
        the logarithms of the positive exit rates and off-diagonal entries.
        Entries that are zero at s stay zero.

        Args:
            s (dict): The parameters.

        Returns:
            tuple: The free parameters (ndarray) and the function that maps
                them back to a parameter dictionary.
        """

        p = self.nphases
        pi = np.asarray(s["pi"], dtype=float)
        positive = np.flatnonzero(pi > 0.0)
        reference = positive[np.argmax(pi[positive])]
        pifree = positive[positive != reference]
        tfree = np.flatnonzero(s["t"] > 0.0)
        offfree = self.offmask & (s["T"] > 0.0)
        noff = int(np.sum(offfree))

        values = np.r_[np.log(pi[pifree] / pi[reference]), np.log(s["t"][tfree]),
                       np.log(s["T"][offfree])]

        def build(v: np.array) -> dict:
            new = dict(s)
            k = 0
            pinew = np.zeros(p)
            pinew[reference] = 1.0
            pinew[pifree] = np.exp(v[k:k + pifree.size])
            k += pifree.size
            t = np.zeros(p)
            t[tfree] = np.exp(v[k:k + tfree.size])
            k += tfree.size
            T = np.zeros((p, p))
            T[offfree] = np.exp(v[k:k + noff])
            np.fill_diagonal(T, -(T.sum(axis=1) + t))
            new["pi"], new["t"], new["T"] = pinew / np.sum(pinew), t, T
            return new

        return values, build

    def __information(self) -> np.array:
        """
        Returns the observed information of beta in the fixed-effects model
        with (pi, T) held at the fit, on the centred design:
        -X' diag(l'') X. With every column centred, the information shared
        between beta and the scale of T vanishes to first order, which is
        what makes this conditional information valid for beta (design note,
        Sections 9 and 11.9).

        Args:
            None

        Returns:
            ndarray: The information matrix.
        """

        s = self.state
        _, d2 = self.__derivs(s, self.X @ s["beta"], self.obs)

        return -(self.X * d2[:, None]).T @ self.X

    def __randominformation(self) -> tuple:
        """
        Returns the observed information of all free parameters of the
        random-intercept model: those of the phase-type representation
        (__rateparameters), beta and log sigma, by central differences of the
        marginal log-likelihood with the nodes re-adapted at each evaluation.

        Holding (pi, T) at the fit is not enough here. sigma trades off
        against the shape of the base distribution, a wider base distribution
        explaining part of the spread that a larger sigma would, and the
        information it shares with the rates does not vanish under centring.
        On one data set the conditional standard errors of log sigma and of a
        coefficient were 0.14 and 0.17 against 0.25 and 0.34 from the full
        information (design note, Section 11.11).

        Args:
            None

        Returns:
            tuple: The information matrix (ndarray), the free parameters of
                the representation coming first, and their number (int).
        """

        s = self.state
        rates, build = self.__rateparameters(s)
        m = rates.size
        x0 = np.r_[rates, np.asarray(s["beta"], dtype=float).reshape(-1), np.log(s["sigma"])]

        def objective(x):
            trial = build(x[:m])
            trial["beta"] = x[m:m + self.d]
            trial["sigma"] = float(np.exp(x[-1]))
            modes, scales, _ = self.__modes(trial, self.modes)
            return self.__frozenloglik(trial, self.aghq.nodes(modes, scales), scales)[0]

        k = x0.size
        h = 1e-4
        hessian = np.zeros((k, k))
        center = objective(x0)
        for i in range(k):
            ei = np.zeros(k)
            ei[i] = h
            hessian[i, i] = (objective(x0 + 2 * ei) - 2 * center
                             + objective(x0 - 2 * ei)) / (4 * h * h)
            for j in range(i + 1, k):
                ej = np.zeros(k)
                ej[j] = h
                value = (objective(x0 + ei + ej) - objective(x0 + ei - ej)
                         - objective(x0 - ei + ej) + objective(x0 - ei - ej)) / (4 * h * h)
                hessian[i, j] = hessian[j, i] = value

        return -hessian, m

    @staticmethod
    def __profiled(information: np.array, m: int) -> np.array:
        """
        Returns the information of the parameters after the first m, the
        first m profiled out: A - B C^+ B' for the blocks C (the first m), B
        and A. The raw phase-type parameters need not be identified, so C can
        be singular; its pseudo-inverse drops the directions in which the
        likelihood is flat, which identified parameters do not load on.

        Args:
            information (ndarray): The full information matrix.
            m (int): The number of parameters to profile out.

        Returns:
            ndarray: The profiled information.
        """

        if m == 0:
            return information

        C = information[:m, :m]
        B = information[m:, :m]
        A = information[m:, m:]
        values, vectors = np.linalg.eigh(C)
        keep = values > 1e-9 * max(float(np.max(np.abs(values))), 1e-300)
        inverse = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T

        return A - B @ inverse @ B.T

    def __finalize(self) -> None:
        """
        Evaluates the final log-likelihood, the predicted random intercepts,
        the covariance of the identified parameters and the parameter count.

        Args:
            None

        Returns:
            None
        """

        s = self.state
        self.pi = s["pi"] / np.sum(s["pi"])
        self.phgen = s["T"]
        self.exitrates = s["t"]
        self.beta = s["beta"]
        self.sigma = s["sigma"]

        if self.groups is None:
            self.loglikelihood = self.__fixedloglik(s)
            self.modes = None
            self.ranef = None
        elif self.boundary:
            # sigma = 0: the model without random intercept
            self.loglikelihood = self.__fixedloglik(s)
            self.modes = np.zeros(self.ngroups)
            self.ranef = np.zeros(self.ngroups)
        else:
            self.modes, scales, _ = self.__modes(s, np.zeros(self.ngroups))
            nodes = self.aghq.nodes(self.modes, scales)
            self.loglikelihood, weights = self.__frozenloglik(s, nodes, scales)
            # the posterior mean of each group's intercept
            self.ranef = np.sum(weights * nodes, axis=1)

        # the covariance of the identified parameters, (beta, log sigma);
        # unavailable where the information is not positive definite. Without
        # a random intercept, and at the boundary sigma = 0, that of beta with
        # (pi, T) held at the fit; with a random intercept the full
        # information with (pi, T) profiled out
        k = self.d + (self.groups is not None)
        kidentified = k - 1 if self.boundary else k
        self.covariance = np.full((k, k), np.nan)
        if kidentified:
            if self.groups is None or self.boundary:
                information = self.__information()
            else:
                information = self.__profiled(*self.__randominformation())
            if np.all(np.isfinite(information)) and np.all(np.linalg.eigvalsh(information) > 0.0):
                self.covariance[:kidentified, :kidentified] = np.linalg.inv(information)
            else:
                warnings.warn(
                    "The observed information is not positive definite, so no "
                    "standard errors are reported.", RuntimeWarning, stacklevel=3)

        # independent parameters: those of the phase-type representation, the
        # diagonal of T being fixed by its row, plus beta and sigma
        offdiagonal = np.array(self.phgen, dtype=float)
        np.fill_diagonal(offdiagonal, 0.0)
        self.nparam = (int(np.count_nonzero(self.pi)) - 1
                       + int(np.count_nonzero(offdiagonal))
                       + int(np.count_nonzero(self.exitrates))
                       + k)

        return None
