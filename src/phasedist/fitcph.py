import numpy as np
from scipy.linalg import expm


class fitcph:
    """
    Fits continuous-time phase-type distributions using the
    EM algorithm from p. 678 Bladt and Nielsen (2017).

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
        Initializes the continuous-time phase-type distribution fitter.

        Args:
            obs (array-like): Observed realizations of the phase-type distribution.
            initpi (ndarray): Initial distribution vector.
            initphgen (ndarray): Initial phase-type generator matrix.
            initexitrates (ndarray): Initial exit rate vector.
            randominit (bool, default=True): Whether to randomize initial parameters.
            seed (int): Random seed for reproducibility.
            tolerance (float, default=1e-6): Convergence tolerance for the EM algorithm.
            itermax (int, default=1000000): Maximum number of EM iterations.
            verbose (bool, default=False): Whether to print intermediate fitting information.
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
        Fits the continuous-time phase-type distribution using the EM algorithm.

        Args:
            None

        Returns:
            None
        """
        # fit the CPH distribution
        iter = 0
        eps = np.inf
        loglik0 = -np.inf
        while iter < self.itermax and eps > self.tolerance:
            self.__estep()
            self.__mstep()
            eps = self.loglikelihood - loglik0  # loglik is evaluated within the E-step
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
        self.__polish()
        self.__updatelikelihood()  # evaluate final loglik

    def getinitdist(self) -> np.array:
        """
        Returns the initial distribution vector.

        Args:
            None

        Returns:
            ndarray: The initial distribution.
        """
        return self.pi

    def getphasegen(self) -> np.array:
        """
        Returns the phase-type generator matrix.

        Args:
            None

        Returns:
            ndarray: The phase-type generator.
        """
        return self.phgen

    def getexitrates(self) -> np.array:
        """
        Returns the exit rate vector.

        Args:
            None

        Returns:
            ndarray: The exit rates.
        """
        return self.exitrates

    def getmean(self) -> float:
        """
        Returns the mean of the continuous-time phase-type distribution.

        Args:
            None

        Returns:
            float: The mean of the distribution.
        """
        return -np.sum(np.matmul(self.pi, np.linalg.inv(self.phgen)))

    def getvar(self) -> float:
        """
        Returns the variance of the continuous-time phase-type distribution.

        Args:
            None

        Returns:
            float: The variance of the distribution.
        """
        phinv = np.linalg.inv(self.phgen)
        return 2 * np.sum(
            np.matmul(self.pi, np.linalg.matrix_power(phinv, 2))
        ) - np.power(np.sum(np.matmul(self.pi, phinv)), 2)

    def getdensity(self, x: float) -> float:
        """
        Returns the distribution's density, f(x).

        Args:
            x (float): The distribution's density will be computed at time of x.

        Returns:
            float: The computed density.
        """
        return np.matmul(
            self.pi, np.matmul(expm(self.phgen * x), self.exitrates)
        ).item()

    def getcumprob(self, x: float) -> float:
        """
        Returns the cumulative distribution function P(X ≤ x).

        Args:
            x (float): The time at which the cumulative probability is evaluated.

        Returns:
            float: The cumulative probability.
        """
        return 1 - np.sum(np.matmul(self.pi, expm(self.phgen * x)))

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

        self.obs = self.obs.astype(float)
        self.initpi = self.initpi.astype(float)
        self.initphgen = self.initphgen.astype(float)
        self.initexitrates = self.initexitrates.astype(float)
        self.identity = np.eye(self.nphases)

        self.pi = self.initpi
        self.phgen = self.initphgen
        self.exitrates = self.initexitrates

        if self.randominit:
            self.__initrandom()

        self.__countParameters()

    def __initrandom(self) -> None:
        """
        Randomly initializes the parameters of the phase-type distribution.

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
        u = np.random.uniform(low=0.0, high=10.0, size=len(nzidx))
        self.exitrates[nzidx, 0] = u

        for i in range(self.nphases):
            nzidx = np.nonzero(self.phgen[i, :])[1]
            msk = nzidx != i
            nzidx = nzidx[msk]
            u = np.random.uniform(low=0.0, high=10.0, size=len(nzidx))
            self.phgen[i, nzidx] = u
            self.phgen[i, i] = -(np.sum(u) + self.exitrates[i, 0])

    def __estep(self) -> None:
        """
        Performs the expectation (E) step of the EM algorithm.

        Args:
            None

        Returns:
            None
        """
        self.bi = np.zeros(self.nphases)
        self.zi = np.zeros(self.nphases)
        self.ni = np.zeros(self.nphases)
        self.nij = np.zeros((self.nphases, self.nphases))
        self.loglikelihood = 0.0

        for y in self.obs:
            self.__Jmatrix(y)
            eTyt = np.matmul(self.eTy, self.exitrates)
            pieTy = np.matmul(self.pi, self.eTy)
            pieTyt = np.matmul(pieTy, self.exitrates)
            self.loglikelihood += np.log(pieTyt)
            for i in range(self.nphases):
                self.bi[i] += (self.pi[0, i] * eTyt[i, 0]) / pieTyt
                self.zi[i] += self.Jmat[i, i] / pieTyt
                for j in range(self.nphases):
                    if j != i:
                        self.nij[i, j] += (self.phgen[i, j] * self.Jmat[j, i]) / pieTyt
                self.ni[i] += (pieTy[0, i] * self.exitrates[i, 0]) / pieTyt

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
            self.exitrates[i, 0] = self.ni[i] / self.zi[i]

        for i in range(self.nphases):
            sm = self.exitrates[i, 0]
            for j in range(self.nphases):
                if j != i:
                    self.phgen[i, j] = self.nij[i, j] / self.zi[i]
                    sm += self.phgen[i, j]
            self.phgen[i, i] = -sm

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
            self.loglikelihood += np.log(self.getdensity(y))

    def __Jmatrix(self, y: float) -> None:
        """
        Computes the J matrix and matrix exponential exp(Ty).

        Args:
            y (float): Observation value.

        Returns:
            None
        """
        mat = expm(
            np.block(
                [
                    [self.phgen, np.matmul(self.exitrates, self.pi)],
                    [np.zeros((self.nphases, self.nphases)), self.phgen],
                ]
            )
            * y
        )

        self.eTy = mat[: self.nphases, : self.nphases]
        self.Jmat = mat[: self.nphases, self.nphases : 2 * self.nphases]

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
        v = np.add(np.sum(self.phgen, axis=1), self.exitrates)
        np.fill_diagonal(self.phgen, -v)
