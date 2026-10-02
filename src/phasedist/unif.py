import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson


class unif:
    """
    Evaluates the matrix exponential exp(Ay) at many values of y by
    uniformization, also known as Jensen's method or randomization.

    The matrix is written as A = Gamma(R - I) with Gamma = max_i |a_ii|, so that
    R = A/Gamma + I, and

        exp(Ay) = sum_k exp(-Gamma y) (Gamma y)^k / k! R^k,

    a Poisson-weighted sum of powers of R. When A is a sub-intensity matrix, R
    is sub-stochastic and every entry of every term is non-negative: nothing is
    ever subtracted from anything, which is what makes this stable where a
    Pade-based matrix exponential is not. A generator whose rates are nearly
    equal has nearly confluent eigenvalues, the worst case for the divided
    differences behind exp(Ay), and uniformization is indifferent to it because
    it never forms an eigenvalue.

    The powers R^k do not depend on y. Only the Poisson weights do. The powers
    are therefore built once and shared across every y, and the weighting of
    them is a single matrix product, which is what makes this fast as well as
    stable.

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
            tolerance (float, default=1e-12): The largest truncation error
                accepted in any evaluation, measured in the infinity norm.
            maxterms (int, default=100000): The most terms the series may use.
                Reaching it means the requested tolerance was not met, which is
                reported rather than passed over.
        """

        self.tolerance = tolerance
        self.maxterms = maxterms

        # filled in by run(), and kept so a caller can see what it cost
        self.gamma = None       # the uniformization rate, max_i |a_ii|
        self.rho = None         # the row-sum norm of R, 1 for a sub-generator
        self.nterms = None      # terms used, i.e. the right truncation point K
        self.errorbound = None  # the bound on the truncation error

        return None

    def __checkinputs(self, matrix: np.array, times: np.array) -> bool:
        """
        Checks the matrix and the evaluation times before any work is done.

        Args:
            matrix (ndarray): The matrix to exponentiate.
            times (ndarray): The values of y.

        Returns:
            bool: True when the inputs can be used.
        """

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            print("Error: The matrix has to be square.")
            return False

        if not np.all(np.isfinite(matrix)):
            print("Error: The matrix contains values that are not finite.")
            return False

        if times.ndim != 1:
            print("Error: The evaluation times have to form a one-dimensional array.")
            return False

        if times.size == 0:
            print("Error: There are no evaluation times.")
            return False

        if not np.all(np.isfinite(times)):
            print("Error: The evaluation times contain values that are not finite.")
            return False

        if np.any(times < 0.0):
            print("Error: The evaluation times have to be non-negative.")
            return False

        if self.tolerance <= 0.0:
            print("Error: The tolerance has to be positive.")
            return False

        return True

    def __uniformize(self, matrix: np.array) -> tuple:
        """
        Returns the uniformization rate Gamma and the matrix R = A/Gamma + I.

        Gamma is the largest absolute diagonal entry. For a sub-intensity matrix
        this is the largest total exit rate, and R is then a sub-stochastic
        matrix. Note that for the block matrix [[T, t pi],[0, T]] the diagonal is
        that of T repeated, so doubling the dimension does not raise Gamma and
        therefore does not cost a single extra term.

        Args:
            matrix (ndarray): The matrix to exponentiate.

        Returns:
            tuple: Gamma (float) and R (ndarray).
        """

        gamma = float(np.max(np.abs(np.diag(matrix))))

        if gamma <= 0.0:
            # a zero matrix, or one with a zero diagonal; exp(Ay) is then
            # handled by the caller without a series
            return 0.0, None

        return gamma, matrix / gamma + np.eye(matrix.shape[0])

    def __terms(self, mu: float, rho: float) -> tuple:
        """
        Returns the number of terms needed and the bound on the truncation
        error that comes with it.

        Truncating after K terms leaves sum_{k>K} w_k R^k, which is bounded in
        the infinity norm by sum_{k>K} w_k ||R||^k. When R is sub-stochastic
        ||R|| is 1 and this is the Poisson tail, exactly as in Stewart's
        Equation (10.49). When it is not -- the block matrices behind the
        censored parts of an E-step are not sub-generators, since their rows can
        sum above zero when the exit rates are below one -- the tail carries a
        factor rho^k, and

            sum_{k>K} w_k rho^k = exp(mu(rho-1)) P(Poisson(mu rho) > K),

        so the Poisson tail is simply required to be that much smaller. Ignoring
        the factor would report an accuracy the result does not have.

        Args:
            mu (float): The Poisson parameter, Gamma times the largest y.
            rho (float): The row-sum norm of R.

        Returns:
            tuple: The number of terms K (int) and the error bound (float).
        """

        if mu <= 0.0:
            return 0, 0.0

        if rho <= 1.0:
            required = self.tolerance
            parameter = mu
            inflation = 1.0
        else:
            inflation = float(np.exp(mu * (rho - 1.0)))
            required = self.tolerance / inflation
            parameter = mu * rho

        if required <= 0.0:
            # the inflation factor has overflowed the tolerance to nothing
            return self.maxterms, np.inf

        nterms = int(poisson.isf(required, parameter)) + 1
        nterms = max(nterms, 1)

        if nterms > self.maxterms:
            return self.maxterms, np.inf

        bound = float(inflation * poisson.sf(nterms, parameter))

        return nterms, bound

    def __weights(self, mus: np.array, nterms: int) -> np.array:
        """
        Returns the Poisson weights exp(-mu) mu^k / k! for every mu and every
        k from 0 to K.

        The weights are formed in logarithms and exponentiated once. Written
        directly, exp(-mu) underflows to zero for mu above about 745 and mu^k
        overflows long before that, so the product of the two is unusable for
        exactly the large Gamma*y that makes a long series necessary in the
        first place. In logarithms the largest weight is near k = mu and is
        about 1/sqrt(2 pi mu), which is an ordinary number at any mu; the
        weights that do underflow are the ones far from the mode, which are
        negligible by construction.

        Args:
            mus (ndarray): The Poisson parameter for each evaluation time.
            nterms (int): The right truncation point K.

        Returns:
            ndarray: The weights, with one row per time and K+1 columns.
        """

        k = np.arange(nterms + 1, dtype=float)
        logfactorial = gammaln(k + 1.0)

        weights = np.zeros((mus.size, nterms + 1), dtype=float)

        positive = mus > 0.0

        if np.any(positive):
            mu = mus[positive][:, None]
            logweights = -mu + k[None, :] * np.log(mu) - logfactorial[None, :]
            # terms far below the mode underflow to zero, which is what they
            # are worth; the exponential is the only place that can happen
            with np.errstate(under="ignore"):
                weights[positive] = np.exp(logweights)

        # a time of zero asks for exp(0) = I, i.e. all the mass on k = 0
        if np.any(~positive):
            weights[~positive, 0] = 1.0

        return weights

    def __powers(self, R: np.array, nterms: int, nrows: int) -> np.array:
        """
        Returns the first nrows rows of R^k for every k from 0 to K, stacked
        into one array so that the weighting below is a single matrix product.

        Only the rows that are asked for are propagated. Carrying p rows of a
        2p-wide matrix costs p(2p)^2 per step where carrying the whole matrix
        costs (2p)^3, and for the E-step only the top p rows are ever read.

        Args:
            R (ndarray): The uniformized matrix.
            nterms (int): The right truncation point K.
            nrows (int): How many leading rows to propagate.

        Returns:
            ndarray: Shape (K+1, nrows*n), row k holding R^k flattened.
        """

        n = R.shape[0]
        stacked = np.empty((nterms + 1, nrows * n), dtype=float)

        # R^0 is the identity, of which only the leading nrows rows are kept
        current = np.zeros((nrows, n), dtype=float)
        current[np.arange(nrows), np.arange(nrows)] = 1.0

        for k in range(nterms + 1):
            stacked[k] = current.ravel()
            if k < nterms:
                current = current @ R

        return stacked

    def run(
        self,
        matrix: np.array,
        times: np.array,
        nrows: int = None,
    ) -> np.array:
        """
        Returns exp(A y) for every y, or its leading rows.

        Args:
            matrix (ndarray): The square matrix A.
            times (ndarray): The values of y, which must be non-negative.
            nrows (int, default=None): How many leading rows of exp(Ay) to
                return. The whole matrix is returned when this is None.

        Returns:
            ndarray: Shape (len(times), nrows, n), or None when the inputs
                cannot be used.
        """

        matrix = np.asarray(matrix, dtype=float)
        times = np.atleast_1d(np.asarray(times, dtype=float))

        if not self.__checkinputs(matrix, times):
            return None

        n = matrix.shape[0]

        if nrows is None:
            nrows = n
        elif not isinstance(nrows, (int, np.integer)) or nrows < 1 or nrows > n:
            print("Error: The number of rows has to be between 1 and the size of the matrix.")
            return None

        gamma, R = self.__uniformize(matrix)

        if R is None:
            # nothing to uniformize: exp(Ay) is the identity for every y
            self.gamma, self.rho, self.nterms, self.errorbound = 0.0, 1.0, 0, 0.0
            return np.repeat(np.eye(n)[None, :nrows, :], times.size, axis=0)

        rho = float(np.max(np.sum(np.abs(R), axis=1)))
        mus = gamma * times

        nterms, bound = self.__terms(float(np.max(mus)), rho)

        if not np.isfinite(bound):
            print("Error: The series needs more than maxterms terms to reach the tolerance. Raise maxterms, loosen the tolerance, or split the evaluation times.")
            return None

        self.gamma, self.rho, self.nterms, self.errorbound = gamma, rho, nterms, bound

        weights = self.__weights(mus, nterms)
        stacked = self.__powers(R, nterms, nrows)

        return np.matmul(weights, stacked).reshape(times.size, nrows, n)
