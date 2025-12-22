import numpy as np


class fitdph:
    """
    Fits discrete-time phase-type distributions using the
    EM algorithm from p. 675 Bladt and Nielsen (2017).
    
    References:
        Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in Applied Probability.
        Springer. https://doi.org/10.1007/978-1-4939-7049-0    
    """

    def __init__(
        self,
        obs: np.array = None,
        initpi: np.array = None,
        initphgen: np.array = None,
        initexitrates: np.array = None,
        randominit: bool = True,
        seed: int = None,
        tolerance: float = 1e-6,
        itermax: int = 1000000,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the discrete-time phase-type distribution fitter.

        Args:
            obs (np.array): Observed realizations of the phase-type distribution.
            initpi (np.array): Initial distribution vector.
            initphgen (np.array): Initial phase-type transition matrix.
            initexitrates (np.array): Initial exit probability vector.
            randominit (bool, default=True): Whether to randomize initial parameters.
            seed (int): Random seed for reproducibility.
            tolerance (float, default=1e-6): Convergence tolerance for the EM algorithm.
            itermax (int, default=1000000): Maximum number of EM iterations.
            verbose (bool, default=False): Whether to print intermediate fitting information.

        Returns:
            None
        """
        self.obs = obs  # observed realizations of the PH distribution
        self.initpi = initpi
        self.initphgen = initphgen
        self.initexitrates = initexitrates
        self.nphases = self.initphgen.shape[0]
        self.randominit = randominit
        self.seed = seed
        self.itermax = itermax
        self.tolerance = tolerance
        self.verbose = verbose
        self.__initialize()

    def fit(self) -> None:
        """
        Fits the discrete-time phase-type distribution using the EM algorithm.

        Args:
            None

        Returns:
            None
        """
        # fit the DPH distribution
        iter = 0
        eps = np.inf
        loglik0 = self.loglikelihood
        while iter < self.itermax and eps > self.tolerance:
            self.__estep()
            self.__mstep()
            self.__updatelikelihood()
            eps = self.loglikelihood - loglik0
            loglik0 = self.loglikelihood
            iter += 1
            if self.verbose and iter % 25 == 0:
                if isinstance(eps, float):
                    printeps = eps
                else:
                    printeps = eps.item()
                print(
                    "iter =",
                    iter,
                    "  eps =",
                    printeps,
                    "  mean =",
                    self.getmean(),
                    "  var =",
                    self.getvar(),
                )
        # self.__polish()

    def getinitdist(self) -> np.array:
        """
        Returns the initial distribution vector.

        Args:
            None

        Returns:
            np.array: The initial distribution.
        """
        return self.pi

    def getphasegen(self) -> np.array:
        """
        Returns the phase-type transition matrix.

        Args:
            None

        Returns:
            np.array: The phase-type generator matrix.
        """
        return self.phgen

    def getexitrates(self) -> np.array:
        """
        Returns the exit probability vector.

        Args:
            None

        Returns:
            np.array: The exit rates.
        """
        return self.exitrates

    def getmean(self) -> float:
        """
        Returns the mean of the discrete-time phase-type distribution.

        Args:
            None

        Returns:
            float: The mean of the distribution.
        """
        return np.sum(
            np.matmul(
                self.pi, np.linalg.inv(np.subtract(np.eye(self.nphases), self.phgen))
            )
        )

    def getvar(self) -> float:
        """
        Returns the variance of the discrete-time phase-type distribution.

        Args:
            None

        Returns:
            float: The variance of the distribution.
        """
        Tinv = np.linalg.inv(np.subtract(np.eye(self.nphases), self.phgen))
        return (
            np.sum(
                np.matmul(
                    np.matmul(self.pi, Tinv),
                    np.subtract((2 * Tinv), np.eye(self.nphases)),
                )
            )
            - self.getmean() ** 2
        )

    def getdensity(self, x: int) -> float:
        """
        Returns the probability mass function evaluated at x.

        Args:
            x (int): The discrete time point at which the density is evaluated.

        Returns:
            float: The computed probability mass.
        """
        if int(x) != x:
            print("Error: 'x' is not an integer.")
            return np.nan
        else:
            return np.matmul(
                self.pi,
                np.matmul(np.linalg.matrix_power(self.phgen, (x - 1)), self.exitrates),
            ).item()

    def getcumprob(self, x: int) -> float:
        """
        Returns the cumulative distribution function P(X ≤ x).

        Args:
            x (int): The discrete time point at which the cumulative probability is evaluated.

        Returns:
            float: The cumulative probability.
        """
        if int(x) != x:
            print("Error: 'x' is not an integer.")
            return np.nan
        else:
            return 1 - np.sum(np.matmul(self.pi, np.linalg.matrix_power(self.phgen, x)))

    def getloglik(self) -> float:
        """
        Returns the log-likelihood of the fitted model.

        Args:
            None

        Returns:
            float: The log-likelihood value.
        """
        return self.loglikelihood

    def getaic(self) -> float:
        """
        Returns Akaike's Information Criterion (AIC).

        Args:
            None

        Returns:
            float: The AIC value.
        """
        return -2 * self.loglikelihood + 2 * self.nparam

    def getbic(self) -> float:
        """
        Returns the Bayesian Information Criterion (BIC).

        Args:
            None

        Returns:
            float: The BIC value.
        """
        return -2 * self.loglikelihood + self.nparam * np.log(len(self.obs))

    def __initialize(self) -> None:
        """
        Initializes internal parameters and prepares the model for fitting.

        Args:
            None

        Returns:
            None
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        self.obs = self.obs.astype(int)
        self.obs = np.sort(self.obs)
        self.initpi = self.initpi.astype(float)
        self.initphgen = self.initphgen.astype(float)
        self.initexitrates = self.initexitrates.astype(float)
        self.identity = np.eye(self.nphases)

        self.pi = self.initpi
        self.phgen = self.initphgen
        self.exitrates = self.initexitrates

        if self.randominit:
            self.__initrandom()
        self.__updatelikelihood()
        self.__countParameters()

    def __initrandom(self) -> None:
        """
        Randomly initializes the parameters of the discrete-time phase-type distribution.

        Args:
            None

        Returns:
            None
        """
        nzidx = np.nonzero(self.pi)[1]
        u = np.random.uniform(low=0.0, high=1.0, size=len(nzidx))
        u = u / np.sum(u)
        self.pi[0, nzidx] = u

        nzidx = np.nonzero(self.exitrates)[0]
        u = np.random.uniform(low=0.0, high=1.0, size=len(nzidx))
        self.exitrates[nzidx, 0] = u

        for i in range(self.nphases):
            nzidx = np.nonzero(self.phgen[i, :])[1]
            u = np.random.uniform(low=0.0, high=1.0, size=len(nzidx))
            u = (u / np.sum(u)) * (1.0 - self.exitrates[i, 0])
            self.phgen[i, nzidx] = u

    def __estep(self) -> None:
        """
        Performs the expectation (E) step of the EM algorithm.

        Args:
            None

        Returns:
            None
        """
        self.bi = np.zeros(self.nphases)
        self.ni = np.zeros(self.nphases)
        self.nij = np.zeros((self.nphases, self.nphases))
        self.phgeninv = np.linalg.inv(self.phgen)

        y0 = -1
        for y in self.obs:
            if y != y0:
                if y == 1:
                    Tpow = self.identity
                    Ttprod = self.exitrates
                else:
                    Tpow = np.linalg.matrix_power(self.phgen, (y - 1))
                    Ttprod = np.matmul(Tpow, self.exitrates)
                piTtprod = np.matmul(self.pi, Ttprod)

                if y >= 2:
                    self.__Kmatrix(y)
            y0 = y

            if piTtprod != 0.0:
                for i in range(self.nphases):
                    self.bi[i] += self.pi[0, i] * Ttprod[i, 0] / piTtprod
                    piTpow = np.matmul(self.pi, Tpow)
                    self.ni[i] += piTpow[0, i] * self.exitrates[i, 0] / piTtprod

                    if y >= 2:
                        for j in range(self.nphases):
                            self.nij[i, j] += (
                                self.phgen[i, j] * self.Kmat[j, i] / piTtprod
                            )

    def __mstep(self) -> None:
        """
        Performs the maximization (M) step of the EM algorithm.

        Args:
            None

        Returns:
            None
        """
        for i in range(self.nphases):
            self.pi[0, i] = self.bi[i] / len(self.obs)

        for i in range(self.nphases):
            sm = self.ni[i]
            for j in range(self.nphases):
                sm += self.nij[i, j]
            self.exitrates[i, 0] = self.ni[i] / sm

        for i in range(self.nphases):
            for j in range(self.nphases):
                sm = self.ni[i]
                for k in range(self.nphases):
                    sm += self.nij[i, k]
                self.phgen[i, j] = self.nij[i, j] / sm

    def __updatelikelihood(self) -> None:
        """
        Updates the log-likelihood based on current model parameters.

        Args:
            None

        Returns:
            None
        """
        self.loglikelihood = 0.0
        for y in self.obs:
            self.loglikelihood += np.log(self.__getProbMass(y)).item()

    def __getProbMass(self, y: int) -> float:
        """
        Computes the probability mass at observation y.

        Args:
            y (int): Discrete observation value.

        Returns:
            float: The probability mass.
        """
        return np.matmul(
            self.pi, np.matmul(np.linalg.matrix_power(self.phgen, y), self.exitrates)
        )

    def __Kmatrix(self, y: int) -> None:
        """
        Computes the matrix-function K for a given observation.

        Args:
            y (int): Observation value.

        Returns:
            None
        """
        if y < 2:
            return

        self.Kmat = np.zeros((self.nphases, self.nphases))

        Ty = np.linalg.matrix_power(self.phgen, y - 2)
        exit_prod = np.matmul(Ty, self.exitrates)
        pi_mat = self.pi

        for k in range(y - 1):
            self.Kmat += np.outer(exit_prod, pi_mat)
            if k < y - 2:
                Ty = np.matmul(Ty, self.phgeninv)
                exit_prod = np.matmul(Ty, self.exitrates)
                pi_mat = np.matmul(pi_mat, self.phgen)

    def __countParameters(self) -> None:
        """
        Counts the number of independent model parameters.

        Args:
            None

        Returns:
            None
        """
        phg = 0
        for i in range(self.nphases):
            phg += (
                np.count_nonzero(self.phgen[i, :])
                + np.count_nonzero(self.exitrates[i, 0])
                - 1
            )
        self.nparam = phg + (np.count_nonzero(self.pi) - 1)

    def __polish(self) -> None:
        """
        Normalizes and enforces consistency constraints on model parameters.

        Args:
            None

        Returns:
            None
        """
        self.pi = self.pi / np.sum(self.pi)
        np.fill_diagonal(self.phgen, 0.0)
        v = np.subtract(
            np.ones((self.nphases, 1)),
            np.add(np.sum(self.phgen, axis=1), self.exitrates),
        )
        np.fill_diagonal(self.phgen, v)
