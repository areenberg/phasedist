import warnings

import numpy as np
from scipy.linalg import expm
from phasedist.ecph import ecph
from phasedist.mcph import mcph
from phasedist.rndcph import rndcph

class _fitcph:
    """
    Fits continuous-time phase-type distributions using the
    EM algorithm from p. 678 Bladt and Nielsen (2017). Convergence is assessed
    by Aitken acceleration, see McLachlan and Krishnan (2008), Section 4.9.

    Warning: This class does not contain any input checks.

    References:
        Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in Applied Probability.
        Springer. https://doi.org/10.1007/978-1-4939-7049-0

        McLachlan, G. J., & Krishnan, T. (2008). The EM Algorithm and Extensions (2nd ed.).
        Wiley. https://doi.org/10.1002/9780470191613
    """

    def __init__(
        self,
        obs: np.array = None,
        censoring: np.array = None,
        initpi: np.array = None,
        initphgen: np.array = None,
        initexitrates: np.array = None,
        randominit: bool = True,
        seed: int = None,
        tolerance: float = 1e-6,
        itermax: int = 100000,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the continuous-time phase-type distribution fitter.

        Censored observations are specified with the (n_obs x 2) censoring array,
        in which each row corresponds to the observation in the same position of
        obs and the two columns say how that observation is censored:
        - [np.nan,np.nan] -> uncensored, the value in obs is used.
        - [np.nan,float] -> right-censored, it is known only that Y > that value.
        - [float,np.nan] -> left-censored, it is known only that Y <= that value.
        - [float,float] -> interval-censored, from the first value to the second.

        The value held in obs is ignored for a censored observation, since what
        is known about it is held in the censoring array instead.

        Args:
            obs (array-like): Observed realizations of the phase-type distribution.
            censoring (ndarray, optional): Specifies censored observations. If
                None, every observation is treated as uncensored.
            initpi (ndarray): Initial distribution vector.
            initphgen (ndarray): Initial phase-type generator matrix.
            initexitrates (ndarray): Initial exit rate vector.
            randominit (bool, default=True): Whether to randomize initial parameters.
            seed (int): Random seed for reproducibility.
            tolerance (float, default=1e-6): Convergence tolerance for the EM algorithm.
            itermax (int, default=100000): Maximum number of EM iterations.
            verbose (bool, default=False): Whether to print intermediate fitting information.
        """
        self.obs = obs  # observed realizations of the PH distribution
        self.censoring = censoring
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

        self.estep = ecph(nphases=self.nphases)
        self.mstep = mcph(nphases=self.nphases,
                          nobs=self.obs.size)

        iter = 0
        eps = np.inf
        loglik0 = -np.inf
        loglik1 = -np.inf  # the log-likelihood two iterations back
        while iter < self.itermax and eps > self.tolerance:

            #E-step
            self.bi,self.zi,self.ni,self.nij = self.estep.run(obs=self.obs,
                                                              initdist=self.pi,
                                                              phgen=self.phgen,
                                                              exitrates=self.exitrates,
                                                              censoring=self.censoring)
            self.loglikelihood = self.estep.loglikelihood

            #M-step
            self.pi,self.phgen,self.exitrates = self.mstep.run(bi=self.bi,
                                                               zi=self.zi,
                                                               ni=self.ni,
                                                               nij=self.nij)

            step = self.loglikelihood - loglik0  # loglik is evaluated within the E-step

            # Each EM iteration is guaranteed to increase the likelihood
            if step < 0.0:
                warnings.warn(
                    "The log-likelihood decreased by %.3e at iteration %d."
                    % (abs(float(step)), iter + 1),
                    RuntimeWarning,
                    stacklevel=2)

            # Aitken acceleration, McLachlan and Krishnan (2008), Section 4.9.
            eps = step
            denominator = loglik0 - loglik1
            if np.isfinite(denominator) and denominator > 0.0:
                rate = step / denominator
                if 0.0 < rate < 1.0:
                    eps = max(step, step * rate / (1.0 - rate))

            loglik1 = loglik0
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
        
        # a tolerance that is not finite means the caller asked for a fixed
        # number of iterations, so stopping at itermax is the intent
        if (iter >= self.itermax and eps > self.tolerance
                and np.isfinite(self.tolerance)):
            warnings.warn(
                "Algorithm terminated with iter==itermax. Results might be "
                "misleading. After %d iterations the estimated distance to the "
                "limit of the log-likelihood was still %.3e, against a "
                "tolerance of %.3e."
                % (iter, float(eps), float(self.tolerance)),
                RuntimeWarning,
                stacklevel=2)

        self.__polish()

        #self.__updatelikelihood()  # evaluate final loglik
        self.estep.run(obs=self.obs,
                       initdist=self.pi,
                       phgen=self.phgen,
                       exitrates=self.exitrates,
                       censoring=self.censoring)
        self.loglikelihood = self.estep.loglikelihood

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

        # The rows of the censoring array are matched to the observations by
        # position, so an array that does not have one row per observation
        # cannot be read at all.
        if self.censoring is not None:
            self.censoring = np.asarray(self.censoring, dtype=float)
            if self.censoring.shape != (self.obs.size, 2):
                print(
                    "Error: The censoring array must have one row per observation and two columns."
                )
                self.censoring = None

        self.initpi = self.initpi.astype(float)
        self.initphgen = self.initphgen.astype(float)
        self.initexitrates = self.initexitrates.astype(float)
        self.identity = np.eye(self.nphases)

        self.pi = np.array(self.initpi, dtype=float).flatten()
        self.phgen = self.initphgen
        self.exitrates = np.array(self.initexitrates, dtype=float).flatten()

        if self.randominit:
            self.__initrandom()

        self.__countParameters()

    def __initrandom(self) -> None:
        """
        Randomly initializes the parameters of the continuous phase-type distribution.

        Args:
            None

        Returns:
            None
        """

        rnd = rndcph(nphases=self.nphases,
                     initdist=self.pi,
                     phgen=self.phgen,
                     exitrates=self.exitrates)

        self.pi, self.phgen, self.exitrates = rnd.run()

        return None

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
                + np.count_nonzero(self.exitrates[i])
                - 1
            )
        self.nparam = phg + (np.count_nonzero(self.pi) - 1)

    def __polish(self) -> None:
        """
        Normalizes model parameters.

        Args:
            None

        Returns:
            None
        """
        self.pi = self.pi / np.sum(self.pi)
