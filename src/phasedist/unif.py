import warnings

import numpy as np
from scipy.special import gammainc, gammaln, pdtrik

class _unif:
    """
    Evaluates the matrix exponential exp(Ty) over several values of y using
    uniformization (also denoted as Jensen's method or randomization).

    The matrix is written as T = Gamma*(P - I) with Gamma = max_i |t_ii|, so that
    P = T/Gamma + I, and

        exp(Ty) = sum_k exp(-Gamma*y) (Gamma*y)^k / k! P^k,

    which is a Poisson-weighted sum of powers of P. Note that when T is a
    sub-intensity matrix, the matrix P is sub-stochastic.

    Warning: This class does not contain any input checks.

    References:
        Stewart, W. J. Introduction to the Numerical Solution of Markov Chains.
        Princeton University Press. https://www.jstor.org/stable/j.ctvcm4gtc
        Section 10.7.2 and Example 10.32.
    """

    def __init__(
        self,
        tolerance: float = 1e-12,
        maxterms: int = 100000,
    ) -> None:
        """
        Initializes the uniformization class.

        Args:
            tolerance (float, default=1e-12): Largest accepted truncation error,
                measured in the infinity norm.
            maxterms (int, default=100000): Largest number of terms in the series.
        """

        self.tolerance = tolerance
        self.maxterms = maxterms

        # set by run()
        self.gamma = None       # uniformization rate, max_i |t_ii|
        self.rho = None         # row-sum norm of P, 1 when P is sub-stochastic
        self.nterms = None      # terms used, i.e. the truncation point K
        self.errorbound = None  # bound on the truncation error

        return None

    def __uniformize(self, matrix: np.array) -> tuple:
        """
        Returns the uniformization rate Gamma and the matrix P = T/Gamma + I.

        Args:
            matrix (ndarray): The matrix T.

        Returns:
            tuple: Gamma (float) and P (ndarray). P is None when Gamma is zero,
                in which case exp(Ty) is the identity.
        """

        gamma = float(np.max(np.abs(np.diag(matrix))))

        if gamma <= 0.0:
            return 0.0, None

        return gamma, matrix / gamma + np.eye(matrix.shape[0])

    def __normbound(self, P: np.array, horizon: int) -> float:
        """
        Returns C = max_k ||P^k||_inf over k = 0,...,horizon.

        The row sums of P^k are P^k e, so iterating r <- P r gives the norms
        exactly at one matrix-vector product each.

        Args:
            P (ndarray): The uniformized matrix.
            horizon (int): How many powers to examine.

        Returns:
            float: The largest norm found.
        """

        r = np.ones(P.shape[0])
        largest = 1.0

        for _ in range(horizon):
            r = P @ r
            value = float(np.max(r))
            if not np.isfinite(value):
                return np.inf
            largest = max(largest, value)

        return largest

    def __terms(self, mu: float, normbound: float) -> tuple:
        """
        Returns the truncation point K and the resulting error bound.

        Truncating after K terms leaves sum_{k>K} w_k P^k, bounded in the
        infinity norm by C * P(Poisson(mu) > K) with C = max_k ||P^k||_inf.
        For a sub-stochastic P, C is 1 and this is Stewart's Equation (10.49).

        Args:
            mu (float): Poisson parameter, Gamma times the largest y.
            normbound (float): C, the largest norm among the powers of P.

        Returns:
            tuple: K (int) and the error bound (float). The bound is infinite
                when K would exceed maxterms or cannot be resolved.
        """

        if mu <= 0.0:
            return 0, 0.0

        if not np.isfinite(normbound) or normbound <= 0.0:
            return self.maxterms, np.inf

        required = self.tolerance / normbound

        if required <= 0.0:
            return self.maxterms, np.inf

        # P(Poisson(mu) > K) is gammainc(K+1, mu). The starting guess comes from
        # the Poisson quantile where 1-required is representable, and from the
        # mean and standard deviation otherwise.
        if 1.0 - required < 1.0:
            # the quantile is not a number for a mu large enough to overflow,
            # which is past the point where any truncation would serve
            guess = pdtrik(1.0 - required, mu)
            nterms = int(guess) + 1 if np.isfinite(guess) else self.maxterms
        else:
            nterms = int(mu + 6.0 * np.sqrt(mu) + 10.0)

        nterms = max(nterms, 1)

        while nterms < self.maxterms and gammainc(nterms + 1, mu) > required:
            nterms += max(1, nterms // 8)

        while nterms > 1 and gammainc(nterms, mu) <= required:
            nterms -= 1

        if nterms >= self.maxterms or gammainc(nterms + 1, mu) > required:
            return self.maxterms, np.inf

        bound = float(normbound * gammainc(nterms + 1, mu))

        return nterms, bound

    def __weights(self, mus: np.array, nterms: int) -> np.array:
        """
        Returns the Poisson weights exp(-mu) mu^k / k! for every mu and k.

        The weights are formed in logarithms, since exp(-mu) underflows for mu
        above roughly 745 and mu^k overflows well before that.

        Args:
            mus (ndarray): Poisson parameter for each evaluation time.
            nterms (int): The truncation point K.

        Returns:
            ndarray: One row per time, K+1 columns.
        """

        k = np.arange(nterms + 1, dtype=float)
        logfactorial = gammaln(k + 1.0)

        weights = np.zeros((mus.size, nterms + 1), dtype=float)

        positive = mus > 0.0

        if np.any(positive):
            mu = mus[positive][:, None]
            logweights = -mu + k[None, :] * np.log(mu) - logfactorial[None, :]
            # terms far from the mode underflow to zero, which is their worth
            with np.errstate(under="ignore"):
                weights[positive] = np.exp(logweights)

        # y = 0 puts all the mass on k = 0, giving exp(0) = I
        if np.any(~positive):
            weights[~positive, 0] = 1.0

        return weights

    def __powers(self, P: np.array, nterms: int, nrows: int) -> np.array:
        """
        Returns the leading nrows rows of P^k for k = 0,...,K, stacked so that
        the weighting in run() is a single matrix product. Only the requested
        rows are propagated, which is the cheaper path when nrows < n.

        Args:
            P (ndarray): The uniformized matrix.
            nterms (int): The truncation point K.
            nrows (int): Number of leading rows to propagate.

        Returns:
            ndarray: Shape (K+1, nrows*n), row k holding P^k flattened.
        """

        n = P.shape[0]
        stacked = np.empty((nterms + 1, nrows * n), dtype=float)

        # P^0 is the identity
        current = np.zeros((nrows, n), dtype=float)
        current[np.arange(nrows), np.arange(nrows)] = 1.0

        for k in range(nterms + 1):
            stacked[k] = current.ravel()
            if k < nterms:
                current = current @ P

        return stacked

    def run(
        self,
        matrix: np.array,
        times: np.array,
        nrows: int = None,
    ) -> np.array:
        """
        Returns exp(Ty) for every y, or its leading rows.

        Args:
            matrix (ndarray): The square matrix T.
            times (ndarray): Non-negative values of y.
            nrows (int, default=None): Number of leading rows of exp(Ty) to
                return. All rows are returned when None.

        Returns:
            ndarray: Shape (len(times), nrows, n).
        """

        matrix = np.asarray(matrix, dtype=float)
        times = np.atleast_1d(np.asarray(times, dtype=float))

        n = matrix.shape[0]

        if nrows is None:
            nrows = n

        gamma, P = self.__uniformize(matrix)

        if P is None:
            self.gamma, self.rho, self.nterms, self.errorbound = 0.0, 1.0, 0, 0.0
            return np.repeat(np.eye(n)[None, :nrows, :], times.size, axis=0)

        mus = gamma * times
        mu = float(np.max(mus))

        rowsums = np.sum(np.abs(P), axis=1)

        if float(np.max(rowsums)) <= 1.0 + 1e-12:
            # the powers of a sub-stochastic P have norm 1, so there is nothing
            # to measure and the horizon it would need is not computed either
            rho = 1.0
        else:
            horizon, _ = self.__terms(mu, 1.0)
            rho = self.__normbound(P, min(horizon, self.maxterms))

        nterms, bound = self.__terms(mu, rho)

        if not np.isfinite(bound):
            # the result is still returned, but its accuracy is not bounded
            warnings.warn(
                "The uniformization series reached maxterms (%d) before meeting the tolerance "
                "of %.3e."
                % (self.maxterms, self.tolerance),
                RuntimeWarning,
                stacklevel=2)

        self.gamma, self.rho, self.nterms, self.errorbound = gamma, rho, nterms, bound

        weights = self.__weights(mus, nterms)
        stacked = self.__powers(P, nterms, nrows)

        return np.matmul(weights, stacked).reshape(times.size, nrows, n)
