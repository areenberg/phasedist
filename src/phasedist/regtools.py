import numpy as np
from numpy.polynomial.hermite import hermgauss


class _stoprule:
    """
    Decides when an EM-type iteration has converged from its log-likelihoods,
    by Aitken extrapolation, see McLachlan and Krishnan (2008), Section 4.9,
    with the rate averaged over a window of steps and a stop confirmed by a
    block of further iterations (design note, Section 7). The rule assumes a
    fixed objective; call reset() when it changes.

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
            # a premature stop: apply the rule again from here
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
