import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad


class _stoprule:
    """
    Decides when an EM-type iteration has converged from its log-likelihoods,
    by Aitken extrapolation, see McLachlan and Krishnan (2008), Section 4.9,
    with the rate averaged over a window of steps and a stop confirmed by a
    block of further iterations. The rule assumes a fixed objective, call
    reset() when it changes.

    Warning: This class does not contain any input checks.

    References:
        McLachlan, G. J., & Krishnan, T. (2008). The EM Algorithm and Extensions (2nd ed.).
        Wiley. https://doi.org/10.1002/9780470191613
    """

    def __init__(
        self,
        tolerance: float = 1e-5,
        window: int = 10,
        confirmmin: int = 200,
        confirmfrac: float = 0.25,
    ) -> None:
        """
        Initializes the stopping rule.

        Args:
            tolerance (float, default=1e-5): Largest accepted remaining gain in
                the log-likelihood.
            window (int, default=10): Number of step ratios the rate is
                averaged over.
            confirmmin (int, default=200): Smallest confirmation block.
            confirmfrac (float, default=0.25): Confirmation block as a fraction
                of the iterations made since the last reset.
        """

        self.tolerance = tolerance
        self.window = window
        self.confirmmin = confirmmin
        self.confirmfrac = confirmfrac
        self.reset()

        return None

    def reset(self) -> None:
        """
        Forgets the history, for use when the objective has changed.

        Args:
            None

        Returns:
            None
        """

        self.history = []
        self.iterations = []
        self.eps = np.inf
        self.confirmstart = None
        self.confirmend = None
        self.converged = False

        return None

    def update(self, loglik: float, iterations: int = 1) -> bool:
        """
        Adds the log-likelihood of the latest step and reports whether the
        iteration has converged. The confirmation block is counted in iterations,
        the window in steps.

        Args:
            loglik (float): The log-likelihood of the latest step.
            iterations (int, default=1): Number of iterations the step counts as.

        Returns:
            bool: True once a stop has been confirmed.
        """

        self.history.append(float(loglik))
        done = (self.iterations[-1] if self.iterations else 0) + iterations
        self.iterations.append(done)
        k = len(self.history) - 1
        self.eps = self.__remaining(k)

        if self.confirmstart is not None:
            if done < self.confirmend:
                return False
            gain = self.history[k] - self.history[self.confirmstart]
            if gain <= self.tolerance:
                self.converged = True
                return True
            # A premature stop: Apply the rule again from here
            self.confirmstart = None
            self.confirmend = None

        if self.eps <= self.tolerance:
            self.confirmstart = k
            self.confirmend = done + max(self.confirmmin,
                                         int(np.ceil(self.confirmfrac * done)))

        return False

    def __remaining(self, k: int) -> float:
        """
        Returns the estimated remaining gain after iteration k, infinite before
        window + 1 steps.

        Args:
            k (int): Index of the latest iteration since the last reset.

        Returns:
            float: The estimated remaining gain.
        """

        if k < self.window + 1:
            return np.inf

        step = self.history[k] - self.history[k - 1]
        old = self.history[k - self.window] - self.history[k - self.window - 1]

        if step <= 0.0 or old <= 0.0:
            return step

        rate = (step / old) ** (1.0 / self.window)

        if rate >= 1.0:
            return np.inf

        return max(step, step * rate / (1.0 - rate))


class _aghq:
    """
    Adaptive Gauss-Hermite quadrature for one Gaussian random effect per
    group, the nodes centred at the mode of the integrand and scaled by its
    curvature, see Liu and Pierce (1994) and Pinheiro and Bates (1995).

    Warning: This class does not contain any input checks.

    References:
        Liu, Q., & Pierce, D. A. (1994). A note on Gauss-Hermite quadrature.
        Biometrika, 81(3), 624-629. https://doi.org/10.1093/biomet/81.3.624

        Pinheiro, J. C., & Bates, D. M. (1995). Approximations to the log-likelihood
        function in the nonlinear mixed-effects model. Journal of Computational and
        Graphical Statistics, 4(1), 12-35. https://doi.org/10.1080/10618600.1995.10474663
    """

    def __init__(self, nnodes: int) -> None:
        """
        Initializes the quadrature rule.

        Args:
            nnodes (int): Number of nodes per group.
        """

        self.nnodes = nnodes
        self.x, self.w = hermgauss(nnodes)
        self.logw = np.log(self.w)
        self.x2 = self.x ** 2

        return None

    def nodes(self, mode: np.array, scale: np.array) -> np.array:
        """
        Returns the adapted nodes of every group.

        Args:
            mode (ndarray): Mode of each group's integrand.
            scale (ndarray): Curvature scale of each group, (-H)^(-1/2).

        Returns:
            ndarray: One row per group, one column per node.
        """

        return mode[:, None] + np.sqrt(2.0) * scale[:, None] * self.x[None, :]

    def weights(self, logintegrand: np.array, scale: np.array) -> tuple:
        """
        Returns each group's log-integral and its posterior node weights,
        computed in logarithms.

        Args:
            logintegrand (ndarray): Log of the integrand at the nodes, one row per group.
            scale (ndarray): Curvature scale of each group.

        Returns:
            tuple: The group log-integrals (ndarray) and the node weights (ndarray).
        """

        c = (logintegrand + self.logw[None, :] + self.x2[None, :]
             + np.log(np.sqrt(2.0) * scale)[:, None])

        largest = np.max(c, axis=1, keepdims=True)
        with np.errstate(under="ignore"):
            logintegral = largest[:, 0] + np.log(np.sum(np.exp(c - largest), axis=1))
            weights = np.exp(c - logintegral[:, None])

        return logintegral, weights


class _quadform:
    """
    The tail of Q = sum_k w_k chi2(h_k), weights of either sign, by Imhof's
    (1961) inversion of its characteristic function.

    Warning: This class does not contain any input checks.

    References:
        Imhof, J. P. (1961). Computing the distribution of quadratic forms in normal
        variables. Biometrika, 48(3/4), 419-426. https://doi.org/10.2307/2332763
    """

    def __init__(self, weights: np.array, dofs: np.array) -> None:
        """
        Initializes the distribution.

        Args:
            weights (ndarray): The weights w_k.
            dofs (ndarray): The degrees of freedom h_k of each weight.
        """

        weights = np.asarray(weights, dtype=float).reshape(-1)
        dofs = np.asarray(dofs, dtype=float).reshape(-1)
        scale = np.max(np.abs(weights)) if weights.size else 0.0
        keep = np.abs(weights) > 1e-12 * scale
        self.weights = weights[keep]
        self.dofs = dofs[keep]
        self.mean = float(np.sum(self.dofs * self.weights))
        self.var = float(2.0 * np.sum(self.dofs * self.weights ** 2))

        return None

    def sf(self, x: float) -> float:
        """
        Returns P(Q > x).

        Args:
            x (float): The point.

        Returns:
            float: The upper tail probability; NaN if every weight is zero.
        """

        if not self.weights.size:
            return np.nan
        w, h = self.weights, self.dofs
        omega = 0.5 * x

        def phi(u):
            return 0.5 * np.sum(h * np.arctan(w * u))

        def rho(u):
            return np.exp(0.25 * np.sum(h * np.logaddexp(0.0, 2.0 * np.log(np.abs(w) * u))))

        def integrand(u):
            if u == 0.0:
                return 0.5 * (self.mean - x)
            return np.sin(phi(u) - omega * u) / (u * rho(u))

        # Directly up to max(1 / max|w|, 1 / omega), then as Fourier integrals
        edge = 1.0 / np.max(np.abs(w))
        head, _ = quad(integrand, 0.0, edge, limit=200, epsabs=1e-12, epsrel=1e-10)
        if omega == 0.0:
            tail, _ = quad(integrand, edge, np.inf, limit=1000, epsabs=1e-12, epsrel=1e-10)
        else:
            frequency = abs(omega)
            sign = np.sign(omega)
            start = max(edge, 1.0 / frequency)
            if start > edge:
                head += quad(lambda t: integrand(np.exp(t)) * np.exp(t), np.log(edge), np.log(start),
                             limit=1000, epsabs=1e-12, epsrel=1e-10)[0]
            tail = (quad(lambda u: np.sin(phi(u)) / (u * rho(u)), start, np.inf, weight="cos", wvar=frequency,
                         limlst=200)[0]
                    - sign * quad(lambda u: np.cos(phi(u)) / (u * rho(u)), start, np.inf, weight="sin", wvar=frequency,
                                  limlst=200)[0])

        return float(np.clip(0.5 + (head + tail) / np.pi, 0.0, 1.0))
