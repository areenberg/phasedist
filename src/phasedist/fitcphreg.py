import time
import warnings

import numpy as np
from phasedist.mcph import mcph
from phasedist.rndcph import rndcph
from phasedist.unif import _unif
from phasedist.regtools import _stoprule, _aghq, _quadform


class _fitcphreg:
    """
    Fits a continuous-time phase-type regression with proportional intensities,
    Y | x, b ~ PH(pi, exp(x'beta + b) T), see Bladt (2022), optionally with a
    Gaussian random effect b per group.

    The parameters are estimated by an ECM algorithm (Meng and Rubin, 1993):
    the E-step of p. 678 in Bladt and Nielsen (2017) on the transformed times
    exp(x'beta + b) y, and conditional M-steps for (pi, T), beta (a Poisson
    regression) and sigma. A random effect is integrated out by adaptive
    Gauss-Hermite quadrature, with sigma = 0 detected by the score test of
    Verbeke and Molenberghs (2003). The starting coefficients are those of
    Gardner (2024), Chapter 4. Convergence is assessed by Aitken acceleration
    with a confirmation block (_stoprule), and the iterations are accelerated
    by SQUAREM, see Varadhan and Roland (2008).

    Warning: This class does not contain any input checks.

    References:
        Bladt, M. (2022). Phase-type distributions for claim severity regression modeling.
        ASTIN Bulletin, 52(2), 417-448. https://doi.org/10.1017/asb.2021.40

        Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in Applied Probability.
        Springer. https://doi.org/10.1007/978-1-4939-7049-0

        Commenges, D., & Andersen, P. K. (1995). Score test of homogeneity for survival
        data. Lifetime Data Analysis, 1(2), 145-156. https://doi.org/10.1007/BF00985764

        Gardner, C. B. (2024). Multivariate Phase-type Distributions in Stochastic Modelling.
        PhD thesis, Technical University of Denmark.

        Lin, X. (1997). Variance component testing in generalised linear models with
        random effects. Biometrika, 84(2), 309-326. https://doi.org/10.1093/biomet/84.2.309

        Meng, X.-L., & Rubin, D. B. (1993). Maximum likelihood estimation via the ECM
        algorithm. Biometrika, 80(2), 267-278. https://doi.org/10.1093/biomet/80.2.267

        Varadhan, R., & Roland, C. (2008). Simple and Globally Convergent Methods for
        Accelerating the Convergence of Any EM Algorithm. Scandinavian Journal of
        Statistics, 35(2), 335-353. https://doi.org/10.1111/j.1467-9469.2007.00585.x

        Verbeke, G., & Molenberghs, G. (2003). The use of score tests for inference on
        variance components. Biometrics, 59(2), 254-262.
        https://doi.org/10.1111/1541-0420.00032
    """

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
        starts: list = None,
        covariance: bool = True,
        censoring: np.array = None,
    ) -> None:
        """
        Initializes the phase-type regression fitter.

        Args:
            obs (ndarray): Positive observations, sorted by group.
            design (ndarray): Centred design matrix without intercept column.
            groups (ndarray, optional): Group code 0, ..., J-1 of each observation,
                non-decreasing; None without random effect.
            initpi (ndarray): Structure of the initial distribution.
            initphgen (ndarray): Structure of the phase-type generator.
            initexitrates (ndarray): Structure of the exit-rate vector.
            quadnodes (int, default=7): Quadrature nodes per group.
            restarts (int, default=10): Number of random starts.
            seed (int): Random seed.
            tolerance (float, default=1e-5): Convergence tolerance.
            itermax (int, default=100000): Maximum number of iterations of each start.
            accelerate (bool, default=True): Whether to accelerate by SQUAREM.
            verbose (bool, default=False): Verbosity flag.
            starts (list, optional): Further starting values, run first:
                dictionaries with 'pi', 'T' and 'beta' on the scale of the design.
            covariance (bool, default=True): Whether to compute the covariance.
            censoring (ndarray, optional): The censoring limits [lower, upper] of
                each row, NaN where there is none;
                None if no row is censored.
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
        self.extrastarts = [] if starts is None else list(starts)
        self.wantcovariance = covariance
        self.censoring = None if censoring is None else np.asarray(censoring, dtype=float).reshape(-1, 2)

        # unused phases (share of the expected time) and their patience
        self.unusedshare = 1e-6
        self.unusedpatience = 20
        # a start that cannot become the best is given up
        self.hopelessiter = 1000
        self.hopelessgap = 1.0
        # the node schedule of the random effect
        self.minadapt = 30
        self.maxadapt = 1000
        self.adaptchange = 1e-3
        self.maxreadapt = 20
        self.readaptconfirm = 30
        # scaling and squaring above this Poisson parameter
        self.maxpoisson = 1000.0
        # SQUAREM candidates costlier than this multiple are rejected unevaluated
        self.squaremscale = 10.0

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

        # censored rows: kind 0 exact, 1 right, 2 left, 3 interval, with the
        # lower limit lo (the observation of an exact row) and the upper hi.
        # Without censored rows the fit takes the uncensored path unchanged.
        self.censored = False
        self.bound = self.obs
        if self.censoring is not None:
            lower, upper = self.censoring[:, 0], self.censoring[:, 1]
            self.kind = np.where(np.isnan(lower), np.where(np.isnan(upper), 0, 2),
                                 np.where(np.isnan(upper), 1, 3))
            self.lo = np.where(self.kind == 0, self.obs, np.where(self.kind == 2, upper, lower))
            self.hi = np.where(self.kind == 3, upper, np.nan)
            self.censored = bool(np.any(self.kind != 0))
            self.bound = np.where(self.kind == 3, self.hi, self.lo)
        self.mstep = mcph(nphases=self.nphases, nobs=self.n)

        # the free generator entries; structural zeros stay zero
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
        self.sigmascore = None
        self.readapts = 0
        self.iterations = 0
        self.loglikelihood = -np.inf

        return None

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS: MATRIX EXPONENTIALS AND THE E-STEP
    # ----------------------------------------------------------------------

    def __expm(self, matrix: np.array, times: np.array, nrows: int = None) -> np.array:
        """
        Returns exp(matrix * y), or its leading rows, for every y by
        uniformization. The times are processed in chunks of similar Poisson
        parameter, and a chunk above maxpoisson is scaled and squared.

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
            if mus[stop - 1] <= self.maxpoisson:
                out[order[start:stop]] = self.unif.run(matrix, ordered[start:stop], nrows=nrows)
            else:
                halvings = int(np.ceil(np.log2(mus[stop - 1] / self.maxpoisson)))
                block = self.unif.run(matrix, ordered[start:stop] / 2.0 ** halvings)
                for _ in range(halvings):
                    block = block @ block
                out[order[start:stop]] = block[:, :nrows, :]
            start = stop

        return out

    def __rowexp(self, pi: np.array, T: np.array, u: np.array) -> np.array:
        """
        Returns pi exp(T u) for every u.

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
        eta, l' = 1 + u f1/f0 and l'' = u f1/f0 + u^2 (f2/f0 - (f1/f0)^2) with
        f_k = pi exp(uT) T^k t and u = e^eta y.

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
        Performs the E-step on the transformed times u. The integral over the time
        of the jump is the upper right block of exp(Bu) for B = [[T, t pi], [0, T]],
        see Van Loan (1978).

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
        Returns the weighted sums of the conditional expectations for the M-step,
        and per row the expected jumps and phase times for the beta step.

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
        # the generator against the transpose of the J-matrix
        out["nij"] = offdiagonal * np.einsum("m,msk->ks", inv, J)
        out["jumps"] = e.get("absorbed", 1.0) + np.einsum("ks,msk->m", offdiagonal, J) * invf
        out["zrow"] = Jdiag * invf[:, None]

        return out

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS: CENSORED ROWS
    # ----------------------------------------------------------------------

    def __limits(self, eta: np.array, repeat: int) -> tuple:
        """
        Returns the kind and the transformed limits of every row, each row
        repeated `repeat` times (once per quadrature node).

        Args:
            eta (ndarray): Linear predictor of each (repeated) row.
            repeat (int): Number of repetitions of each row.

        Returns:
            tuple: Kind, transformed lower limit u and upper limit v (ndarray).
        """

        kind = np.repeat(self.kind, repeat)
        scale = np.exp(eta)
        u = scale * np.repeat(self.lo, repeat)
        v = scale * np.repeat(self.hi, repeat)

        return kind, u, v

    def __rowderivs(self, s: dict, eta: np.array) -> tuple:
        """
        Returns l' and l'' in eta of every row: the uncensored formulas
        unchanged when no row is censored.

        Args:
            s (dict): The parameters.
            eta (ndarray): Linear predictor of each row.

        Returns:
            tuple: l' (ndarray) and l'' (ndarray).
        """

        if not self.censored:
            return self.__derivs(s, eta, self.obs)

        T, t = s["T"], s["t"]
        Tt = T @ t
        kind, u, v = self.__limits(eta, 1)
        d1 = np.zeros(self.n)
        d2 = np.zeros(self.n)

        exact = kind == 0
        if np.any(exact):
            d1[exact], d2[exact] = self.__derivs(s, eta[exact], self.lo[exact])

        with np.errstate(divide="ignore", invalid="ignore"):
            # right-censored: l = log S(u)
            rows = kind == 1
            if np.any(rows):
                a = self.__rowexp(s["pi"], T, u[rows])
                S, f, f1 = np.sum(a, axis=1), a @ t, a @ Tt
                d1[rows] = -u[rows] * f / S
                d2[rows] = -u[rows] * f / S - u[rows] ** 2 * (f1 / S + (f / S) ** 2)

            # left- and interval-censored: l = log(S(u) - S(v)), S(0) = 1
            rows = kind >= 2
            if np.any(rows):
                lower = np.where(kind[rows] == 3, u[rows], 0.0)
                upper = np.where(kind[rows] == 3, v[rows], u[rows])
                a = self.__rowexp(s["pi"], T, np.r_[lower, upper])
                m = lower.size
                Sl, fl, f1l = np.sum(a[:m], axis=1), a[:m] @ t, a[:m] @ Tt
                Su, fu, f1u = np.sum(a[m:], axis=1), a[m:] @ t, a[m:] @ Tt
                D = Sl - Su
                G = upper * fu - lower * fl
                H = upper * fu + upper ** 2 * f1u - lower * fl - lower ** 2 * f1l
                d1[rows] = G / D
                d2[rows] = H / D - (G / D) ** 2

        return d1, d2

    def __censoredloglik(self, s: dict, eta: np.array, repeat: int) -> np.array:
        """
        Returns the log-likelihood contribution of every (repeated) row: the
        log-density for an exact row, log S(u), log F(u) or log(S(u) - S(v))
        for a right-, left- or interval-censored one.

        Args:
            s (dict): The parameters.
            eta (ndarray): Linear predictor of each (repeated) row.
            repeat (int): Number of repetitions of each row.

        Returns:
            ndarray: The contributions.
        """

        kind, u, v = self.__limits(eta, repeat)
        out = np.empty(kind.size)

        exact = kind == 0
        if np.any(exact):
            out[exact] = self.__logdensity(s, eta[exact], np.repeat(self.lo, repeat)[exact])

        censored = ~exact
        interval = kind[censored] == 3
        a = self.__rowexp(s["pi"], s["T"], np.r_[u[censored], v[censored][interval]])
        m = int(np.count_nonzero(censored))
        S = np.sum(a[:m], axis=1)
        prob = np.where(kind[censored] == 1, S, 1.0 - S)
        prob[interval] = S[interval] - np.sum(a[m:], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[censored] = np.where(prob > 0.0, np.log(np.where(prob > 0.0, prob, 1.0)), -np.inf)

        return out

    def __censoredestep(self, s: dict, eta: np.array, repeat: int) -> tuple:
        """
        Performs the E-step with censored rows. Exact rows use J(u), the
        upper right block of exp(Bu) for B = [[T, t pi], [0, T]]; censored
        rows K(u), the same with e pi in place of t pi, and M(u) = (-T)^(-1)
        (I - exp(Tu)). Every row gets the matrix C whose diagonal gives the
        phase times and whose transpose, against the generator, gives the
        jumps, with the vectors for the initial and exit counts.

        Args:
            s (dict): The parameters.
            eta (ndarray): Linear predictor of each (repeated) row.
            repeat (int): Number of repetitions of each row.

        Returns:
            tuple: The log-likelihood of each row (ndarray) and a dictionary
                in the form of __estep() plus 'absorbed'.
        """

        p = self.nphases
        pi, T = s["pi"], s["T"]
        kind, u, v = self.__limits(eta, repeat)
        N = kind.size

        den = np.zeros(N)
        avec = np.zeros((N, p))
        bvec = np.zeros((N, p))
        C = np.zeros((N, p, p))
        absorbed = np.where(kind == 1, 0.0, 1.0)

        exact = kind == 0
        if np.any(exact):
            e = self.__estep(s, u[exact])
            den[exact] = e["f0"]
            avec[exact] = e["a"]
            bvec[exact] = e["eTut"]
            C[exact] = e["J"]

        censored = ~exact
        if np.any(censored):
            interval = kind == 3
            lower = np.where(interval, u, 0.0)[censored]
            upper = np.where(interval, v, u)[censored]
            block = np.block([[T, np.outer(np.ones(p), pi)],
                              [np.zeros((p, p)), T]])
            times = np.r_[upper, lower[interval[censored]]]
            mats = self.__expm(block, times, nrows=p)
            m = upper.size
            eTh, Kh = mats[:m, :, :p], mats[:m, :, p:]
            eTl = np.broadcast_to(np.eye(p), (m, p, p)).copy()
            Kl = np.zeros((m, p, p))
            eTl[interval[censored]] = mats[m:, :, :p]
            Kl[interval[censored]] = mats[m:, :, p:]

            right = kind[censored] == 1
            g = np.linalg.solve(-T.T, pi)        # pi (-T)^(-1)
            ones = np.ones(p)
            # right-censored: C = K(u), B from exp(Tu) 1, no exit
            # left and interval: C = 1 pi dM - dK with pi dM = g (exp(Tl) - exp(Th))
            diffe = eTl - eTh
            piM = np.einsum("k,mkl->ml", g, diffe)
            Cc = piM[:, None, :] - (Kh - Kl)
            Cc[right] = Kh[right]
            bc = diffe @ ones
            bc[right] = eTh[right] @ ones
            ac = piM
            ac[right] = 0.0
            den[censored] = bc @ pi
            bvec[censored] = bc
            avec[censored] = ac
            C[censored] = Cc

        ok = den > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            rows = np.where(ok, np.log(np.where(ok, den, 1.0)), -np.inf)
        rows[exact] = rows[exact] + eta[exact]

        return rows, dict(f0=den,
                          invf=np.where(ok, 1.0 / np.where(ok, den, 1.0), 0.0),
                          a=avec,
                          eTut=bvec,
                          J=C,
                          absorbed=absorbed)

    def __censoredexponential(self) -> tuple:
        """
        Returns the starting coefficients and rate from the one-phase model
        with censoring, exponential regression on z = log(rate) + x'beta,
        maximized by Newton's method with step halving; the problem is
        concave. Without censoring it is the
        quasi-likelihood start.

        Args:
            None

        Returns:
            tuple: The coefficients (ndarray) and the rate at x = 0 (float).
        """

        design = np.column_stack([np.ones(self.n), self.X])
        kind, lo, hi = self.kind, self.lo, self.hi
        finite = np.isfinite(self.bound) & (self.bound > 0.0)
        c = np.zeros(self.d + 1)
        c[0] = -np.log(np.mean(self.bound[finite]))

        def parts(c):
            z = design @ c
            r = np.exp(z)
            value = np.zeros(self.n)
            g = np.zeros(self.n)
            h = np.zeros(self.n)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                rows = kind == 0
                value[rows] = z[rows] - r[rows] * lo[rows]
                g[rows] = 1.0 - r[rows] * lo[rows]
                h[rows] = -r[rows] * lo[rows]
                rows = kind == 1
                value[rows] = -r[rows] * lo[rows]
                g[rows] = -r[rows] * lo[rows]
                h[rows] = -r[rows] * lo[rows]
                rows = kind >= 2
                a = np.where(kind[rows] == 3, r[rows] * lo[rows], 0.0)
                b = np.where(kind[rows] == 3, r[rows] * hi[rows], r[rows] * lo[rows])
                Sa, Sb = np.exp(-a), np.exp(-b)
                D = Sa - Sb
                G = b * Sb - a * Sa
                H = b * Sb - b ** 2 * Sb - a * Sa + a ** 2 * Sa
                value[rows] = np.log(D)
                g[rows] = G / D
                h[rows] = H / D - (G / D) ** 2
            total = float(np.sum(value))
            return (total if np.isfinite(total) else -np.inf), design.T @ g, (design * h[:, None]).T @ design

        current, gradient, hessian = parts(c)
        for _ in range(100):
            try:
                step = np.linalg.solve(-hessian, gradient)
            except np.linalg.LinAlgError:
                break
            scale = 1.0
            while True:
                value, newgradient, newhessian = parts(c + scale * step)
                if value >= current or scale < 1e-10:
                    break
                scale /= 2.0
            if not value >= current:
                break
            c = c + scale * step
            current, gradient, hessian = value, newgradient, newhessian
            if np.max(np.abs(scale * step)) < 1e-12:
                break

        return c[1:], float(np.exp(c[0]))

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS: THE CONDITIONAL M-STEPS
    # ----------------------------------------------------------------------

    def __betastep(self, rows: np.array, jumps: np.array, exposure: np.array,
                   beta: np.array, offset) -> np.array:
        """
        Maximizes Q(beta) = sum m (x'beta + o) - exp(x'beta + o) w, a Poisson
        log-likelihood with pseudo-counts m and exposures w, by Newton's method
        with step halving.

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
        Returns the parameters as one vector: pi, t, the off-diagonal entries of T,
        beta and sigma where present.

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
        Returns the parameters held in a vector from __pack.

        Args:
            values (ndarray): The parameters as one vector.
            template (dict): Parameters of the same model.

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
        Returns whether a vector of parameters describes a model.

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
        Returns the largest Poisson parameter of the uniformization at s, which
        sets the cost of a likelihood evaluation.

        Args:
            s (dict): The parameters.

        Returns:
            float: The bound.
        """

        with np.errstate(over="ignore"):
            u = np.max(np.exp(self.X @ s["beta"]) * self.bound) if self.d else np.max(self.bound)
            return float(np.max(np.abs(np.diag(s["T"]))) * u)

    def __squaremstep(self, s: dict, mapping, objective) -> tuple:
        """
        Performs one SQUAREM iteration, see Varadhan and Roland (2008), as in
        fitcph2dist. A candidate is rejected if it is infeasible, does not improve
        on the second plain iterate, or is too costly to evaluate (squaremscale),
        so the likelihood stays monotone.

        Args:
            s (dict): The parameters.
            mapping (callable): One plain iteration.
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
            largest = self.squaremscale * self.__poissonbound(second)

            while alpha < -1.001:
                candidate = start - 2.0 * alpha * r + alpha * alpha * v
                if self.__isfeasible(candidate, s):
                    trial = self.__unpack(candidate, s)
                    if not self.__poissonbound(trial) <= largest:
                        alpha = (alpha - 1.0) / 2.0
                        continue
                    try:
                        # reject a candidate the uniformization cannot evaluate
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

        if self.censored:
            return float(np.sum(self.__censoredloglik(s, self.X @ s["beta"], 1)))
        return float(np.sum(self.__logdensity(s, self.X @ s["beta"], self.obs)))

    def __fixediteration(self, s: dict) -> tuple:
        """
        One ECM iteration of the fixed-effects model.

        Args:
            s (dict): The parameters.

        Returns:
            tuple: The log-likelihood at s (float) and the updated parameters
                (dict, None if the likelihood is not finite).
        """

        eta = self.X @ s["beta"]
        if self.censored:
            rows, e = self.__censoredestep(s, eta, 1)
            loglik = float(np.sum(rows))
        else:
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

        # the exposure on the time axis of y, E(Z_k) / exp(eta), weighted by the new rates
        exposure = (e["zrow"] @ (-np.diag(new["T"]))) / np.exp(eta)
        new["beta"] = self.__betastep(self.X, e["jumps"], exposure, s["beta"], 0.0)

        return loglik, new

    def __quasilikelihood(self, y: np.array) -> tuple:
        """
        Returns the starting coefficients from the quasi-likelihood equations of a
        Gamma regression with log link, see Gardner (2024), Chapter 4, with the
        sign of the regression (positive means shorter times).

        Args:
            y (ndarray): The observations.

        Returns:
            tuple: The coefficients (ndarray) and the fitted mean at x = 0 (float).
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
        Returns whether the structure is an ordered chain (Coxian, generalized
        Coxian, generalized Erlang), for which a deterministic preset is used.

        Args:
            None

        Returns:
            bool: True for an ordered chain.
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
        Returns a deterministic start for an ordered chain, with distinct rates
        decreasing along the chain.

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
        Returns the starting values: the given starts, then the preset and
        `restarts` random representations (rndcph), scaled to the transformed
        data, with the quasi-likelihood coefficients.

        Args:
            None

        Returns:
            list: One parameter dictionary per start.
        """

        if self.censored:
            beta, rate = self.__censoredexponential()
            target = 1.0 / rate
        else:
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

        # starts given by the caller run first, usually close to the maximum
        starts = []
        for given in self.extrastarts:
            T = np.array(given["T"], dtype=float)
            starts.append(dict(pi=np.array(given["pi"], dtype=float).reshape(-1), T=T,
                               t=-np.sum(T, axis=1), beta=np.array(given["beta"], dtype=float).reshape(-1),
                               sigma=None))

        for pi, T, t in representations:
            mean = float(pi @ np.linalg.solve(-T, np.ones(self.nphases)))
            factor = mean / target
            starts.append(dict(pi=np.array(pi, dtype=float), T=T * factor, t=t * factor,
                               beta=beta.copy(), sigma=None))

        return starts

    def __itermaxwarning(self, rule: _stoprule, iteration: int) -> None:
        """
        Warns that a run reached itermax.

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
                       "convergence: After %d iterations the estimated distance to the "
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
        Runs the fixed-effects ECM from s until the stopping rule is satisfied. A
        run with an unused phase for unusedpatience iterations is stopped as
        degenerate (with abandon), and one whose projected limit lies more than
        hopelessgap below target after hopelessiter iterations as inferior.

        Args:
            s (dict): Starting parameters.
            maxiter (int): Largest number of iterations.
            final (bool): Whether to warn on reaching maxiter.
            abandon (bool, default=False): Whether to stop a degenerate run.
            target (float, optional): The log-likelihood of the best start so far.

        Returns:
            tuple: The parameters (dict), their log-likelihood (float), the status
                ("converged", "itermax", "degenerate", "inferior" or "failed") and
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
            # an accelerated iteration counts as two plain ones
            iteration += 2 if self.accelerate else 1
            s = new

            if self.verbose and final and iteration % 25 == 0:
                print("Fixed effects: Iteration", iteration, "  eps =", rule.eps,
                      "  loglik =", loglik)

            if abandon:
                unusedfor = unusedfor + (2 if self.accelerate else 1) if self.__unused(s) else 0
                if unusedfor >= self.unusedpatience:
                    status = "degenerate"
                    break

            if rule.update(loglik, 2 if self.accelerate else 1):
                status = "converged"
                break

            if (target is not None and iteration >= self.hopelessiter
                    and np.isfinite(rule.eps) and loglik + rule.eps < target - self.hopelessgap):
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
        Returns whether some phase holds a negligible share of the expected time
        before absorption, pi (-T)^(-1), which EM cannot leave.

        Args:
            s (dict): The parameters.

        Returns:
            bool: True if a phase is effectively unused.
        """

        if self.nphases == 1:
            return False

        occupation = np.linalg.solve(-s["T"].T, s["pi"])
        total = float(np.sum(occupation))

        return bool(np.any(occupation < self.unusedshare * total))

    def __fitfixed(self) -> None:
        """
        Fits the fixed-effects model: runs every start to convergence and keeps the
        best fit without an unused phase, or the best degenerate fit with a warning.

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
        # per start: "converged", "itermax", "degenerate", "inferior" or "failed"
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
                print("Start %d: Loglik =" % (number + 1), loglik, " status =", self.startstatus[-1])
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
        # the log-likelihood at sigma = 0, for the test of the random effect
        self.fixedloglikelihood = self.loglikelihood

        return None

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS: RANDOM EFFECT
    # ----------------------------------------------------------------------

    def __modes(self, s: dict, start: np.array) -> tuple:
        """
        Returns the mode of each group's integrand, l_j(b) + log phi(b; 0, sigma^2),
        by damped Newton steps, and the curvature scale (-H)^(-1/2) at the mode.

        Args:
            s (dict): The parameters.
            start (ndarray): Starting point of each group.

        Returns:
            tuple: The modes (ndarray), the scales (ndarray) and the number of
                groups whose curvature fell back on the prior's (int).
        """

        base = self.X @ s["beta"]
        variance = s["sigma"] ** 2
        b = np.array(start, dtype=float)

        for _ in range(50):
            d1, d2 = self.__rowderivs(s, base + b[self.groups])
            gradient = np.add.reduceat(d1, self.starts) - b / variance
            hessian = np.add.reduceat(d2, self.starts) - 1.0 / variance
            hessian = np.where(hessian < 0.0, hessian, -1.0 / variance)
            step = -gradient / hessian
            step = np.where(np.isfinite(step), np.clip(step, -1.0, 1.0), 0.0)
            b = b + step
            if np.max(np.abs(step)) < 1e-10:
                break

        _, d2 = self.__rowderivs(s, base + b[self.groups])
        hessian = np.add.reduceat(d2, self.starts) - 1.0 / variance
        bad = ~(hessian < 0.0)
        hessian = np.where(bad, -1.0 / variance, hessian)

        return b, np.sqrt(-1.0 / hessian), int(np.sum(bad))

    def __prior(self, nodes: np.array, sigma: float) -> np.array:
        """
        Returns log phi(b; 0, sigma^2) at the nodes.

        Args:
            nodes (ndarray): The nodes.
            sigma (float): The standard deviation of the random effect.

        Returns:
            ndarray: The log-densities.
        """

        return -0.5 * nodes ** 2 / sigma ** 2 - 0.5 * np.log(2.0 * np.pi * sigma ** 2)

    def __frozenloglik(self, s: dict, nodes: np.array, scales: np.array) -> tuple:
        """
        Returns the marginal log-likelihood on the given nodes and the posterior
        node weights.

        Args:
            s (dict): The parameters.
            nodes (ndarray): One row of nodes per group.
            scales (ndarray): Their curvature scales.

        Returns:
            tuple: The log-likelihood (float) and the weights (ndarray).
        """

        R = nodes.shape[1]
        eta = (self.X @ s["beta"])[:, None] + nodes[self.groups]
        if self.censored:
            rows = self.__censoredloglik(s, eta.ravel(), R).reshape(self.n, R)
        else:
            rows = self.__logdensity(s, eta.ravel(), np.repeat(self.obs, R)).reshape(self.n, R)
        logintegrand = np.add.reduceat(rows, self.starts, axis=0) + self.__prior(nodes, s["sigma"])
        logintegral, weights = self.aghq.weights(logintegrand, scales)

        return float(np.sum(logintegral)), weights

    def __randomiteration(self, s: dict, nodes: np.array, scales: np.array) -> tuple:
        """
        One ECM iteration of the random-effect model on fixed nodes.

        Args:
            s (dict): The parameters.
            nodes (ndarray): One row of nodes per group.
            scales (ndarray): Their curvature scales.

        Returns:
            tuple: The marginal log-likelihood at s (float) and the updated
                parameters (dict, None if the likelihood is not finite).
        """

        R = nodes.shape[1]
        nodesofrows = nodes[self.groups]
        eta = (self.X @ s["beta"])[:, None] + nodesofrows
        if self.censored:
            rows, e = self.__censoredestep(s, eta.ravel(), R)
            rows = rows.reshape(self.n, R)
        else:
            u = np.exp(eta) * self.obs[:, None]

            e = self.__estep(s, u.ravel())

            # the node weights are the posterior of the node given the group's data
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
        Returns the largest relative change of beta and sigma.

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
        sigma = 0, the sum over the groups of ((sum_i l'_i)^2 + sum_i l''_i) / 2,
        see Verbeke and Molenberghs (2003).

        Args:
            s (dict): The parameters, normally the fixed-effects fit.

        Returns:
            float: The derivative.
        """

        d1, d2 = self.__rowderivs(s, self.X @ s["beta"])

        return 0.5 * float(np.sum(np.add.reduceat(d1, self.starts) ** 2
                                  + np.add.reduceat(d2, self.starts)))

    def __sigmascore(self, s: dict) -> dict:
        """
        Returns the score test of sigma = 0 at the fixed-effects fit s, see
        Commenges and Andersen (1995) and Lin (1997). The rows' scores in eta
        are standardized by their standard deviation, pooled within each kind
        of row (exact, right-, left- and interval-censored), and the statistic
        is T = 1/2 sum_j sum_{i != k in j} e_i e_k, the within-group cross
        products of the standardized scores e. Under sigma = 0 the scores are
        independent with mean zero, apart from the constraints that the
        estimated scale and beta put on them, A' D^(1/2) e = 0 with A = [1, X].
        The p-value is exact for normal scores: with the pooled variance
        estimated from the same scores, T exceeds its observed value when
        sum_j S_j^2 / sum_i e_i^2 does (S_j the group sums), and that ratio
        is a quadratic form whose tail is a weighted sum of chi-square
        variables, so the few-groups case is covered.

        Args:
            s (dict): The fixed-effects fit.

        Returns:
            dict: 'statistic' (T centred and scaled to unit variance under
                sigma = 0) and 'pvalue'.
        """

        n = self.n
        a, _ = self.__rowderivs(s, self.X @ s["beta"])
        kind = self.kind if self.censored else np.zeros(n, dtype=int)

        # the variance of a row's score, pooled within its kind, with the
        # degrees of freedom of the scale and beta taken off
        A = np.column_stack([np.ones(n), self.X])
        rank = int(np.linalg.matrix_rank(A))
        d = np.empty(n)
        for k in np.unique(kind):
            rows = kind == k
            d[rows] = np.mean(a[rows] ** 2)
        d *= n / max(n - rank, 1)
        e = a / np.sqrt(d)

        # the group sums and the statistic
        sums = np.add.reduceat(e, self.starts)
        squares = float(np.sum(e ** 2))
        T = 0.5 * (float(np.sum(sums ** 2)) - squares)

        # Z'(I - H) Z for H the projection on D^(1/2) A, Z the group indicators:
        # the scores' covariance is I - H, so the quadratic form e' Z Z' e - c e' e
        # has the weights mu_k - c (mu the eigenvalues of Z'(I - H) Z) and -c
        # with the remaining n - rank - J degrees of freedom
        W = np.sqrt(d)[:, None] * A
        K = np.linalg.pinv(W.T @ W)
        groupW = np.add.reduceat(W, self.starts, axis=0)
        sizes = np.diff(np.r_[self.starts, n]).astype(float)
        mu = np.linalg.eigvalsh(np.diag(sizes) - groupW @ K @ groupW.T)
        rest = max(n - rank - self.ngroups, 0)
        if not np.any(sizes > 1):
            return dict(statistic=np.nan, pvalue=np.nan)

        ratio = float(np.sum(sums ** 2)) / squares
        pvalue = _quadform(np.r_[mu - ratio, -ratio], np.r_[np.ones(mu.size), rest]).sf(0.0)

        # T standardized by its moments under sigma = 0, with the weights
        # (mu_k - 1) / 2 and -1/2
        moments = _quadform(np.r_[0.5 * (mu - 1.0), -0.5], np.r_[np.ones(mu.size), rest])

        return dict(statistic=(T - moments.mean) / np.sqrt(moments.var), pvalue=pvalue)

    def __atboundary(self, s: dict, converged: bool) -> None:
        """
        Ends the fit at sigma = 0 with the fixed-effects parameters s.

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
            "The estimated standard deviation of the random effect is zero: The "
            "likelihood does not increase as it moves away from zero, so the groups "
            "vary no more than the fixed effects explain. The fit is that of the "
            "model without random effect.", RuntimeWarning, stacklevel=3)

        return None

    def __fitrandom(self) -> None:
        """
        Fits the random-effect model from the fixed-effects fit. The nodes are
        adapted for at least minadapt iterations, then frozen, and re-adapted
        after convergence.

        Args:
            None

        Returns:
            None
        """

        s = dict(self.state)

        # the score test of sigma = 0, at the fixed-effects fit
        self.sigmascore = self.__sigmascore(s)

        # sigma = 0 when the likelihood does not increase away from it
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
                    # frozen nodes: a fixed objective, so SQUAREM stays monotone
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
                settled = self.__change(s, new) < self.adaptchange
                s = new
                if (adapted >= self.minadapt and settled) or adapted >= self.maxadapt:
                    adapting = False
                    rule.reset()
                    previous = None
                continue

            s = new
            sigmas.append(s["sigma"])

            if self.verbose and iteration % 25 == 0:
                print("Random effect: Iteration", iteration, "  eps =", rule.eps,
                      "  loglik =", loglik, "  std. dev. =", s["sigma"])

            # sigma heading for zero converges sublinearly: move to the boundary
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
                if abs(after - before) <= self.tolerance or readapts >= self.maxreadapt:
                    status = "converged"
                    break
                modes, scales, nodes = newmodes, newscales, newnodes
                readapts += 1
                rule = _stoprule(tolerance=self.tolerance, confirmmin=self.readaptconfirm)
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
        Warns if the log-likelihood decreased by more than rounding.

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
        Returns the free parameters of the phase-type representation on the log
        scale, and the function that rebuilds the parameters from them.

        Args:
            s (dict): The parameters.

        Returns:
            tuple: The free parameters (ndarray) and the rebuilding function.
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

    def __randominformation(self) -> tuple:
        """
        Returns the observed information of the phase-type parameters, beta and
        log sigma by central differences of the marginal log-likelihood.

        Args:
            None

        Returns:
            tuple: The information matrix (ndarray) and the number of phase-type
                parameters, which come first (int).
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

    def __fixedinformation(self) -> tuple:
        """
        Returns the observed information of the phase-type parameters and beta
        of the fixed-effects model by central differences of its
        log-likelihood.

        Args:
            None

        Returns:
            tuple: The information matrix (ndarray) and the number of phase-type
                parameters, which come first (int).
        """

        s = self.state
        rates, build = self.__rateparameters(s)
        m = rates.size
        x0 = np.r_[rates, np.asarray(s["beta"], dtype=float).reshape(-1)]

        def objective(x):
            trial = build(x[:m])
            trial["beta"] = x[m:]
            return self.__fixedloglik(trial)

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
        Returns the information of the parameters after the first m with the first
        m profiled out, A - B C^+ B'.

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
        Evaluates the final log-likelihood, the predicted random effects, the
        covariance and the parameter count.

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
            # sigma = 0: the model without random effect
            self.loglikelihood = self.__fixedloglik(s)
            self.modes = np.zeros(self.ngroups)
            self.ranef = np.zeros(self.ngroups)
        else:
            self.modes, scales, _ = self.__modes(s, np.zeros(self.ngroups))
            nodes = self.aghq.nodes(self.modes, scales)
            self.loglikelihood, weights = self.__frozenloglik(s, nodes, scales)
            # the posterior mean of each group's random effect
            self.ranef = np.sum(weights * nodes, axis=1)

        # the covariance of (beta, log sigma)
        k = self.d + (self.groups is not None)
        kidentified = k - 1 if self.boundary else k
        self.covariance = np.full((k, k), np.nan)
        if kidentified and self.wantcovariance:
            if self.groups is None or self.boundary:
                information = self.__profiled(*self.__fixedinformation())
            else:
                information = self.__profiled(*self.__randominformation())
            if np.all(np.isfinite(information)) and np.all(np.linalg.eigvalsh(information) > 0.0):
                self.covariance[:kidentified, :kidentified] = np.linalg.inv(information)
            else:
                warnings.warn(
                    "The observed information is not positive definite, so no "
                    "standard errors are reported.", RuntimeWarning, stacklevel=3)

        # independent parameters of the representation, plus beta and sigma
        offdiagonal = np.array(self.phgen, dtype=float)
        np.fill_diagonal(offdiagonal, 0.0)
        self.nparam = (int(np.count_nonzero(self.pi)) - 1
                       + int(np.count_nonzero(offdiagonal))
                       + int(np.count_nonzero(self.exitrates))
                       + k)

        return None
