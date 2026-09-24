import warnings

import numpy as np
from phasedist.edph import edph
from phasedist.mdph import mdph
from phasedist.rnddph import rnddph

class fitdph:
    """
    Fits discrete-time phase-type distributions using the
    EM algorithm from p. 675 Bladt and Nielsen (2017). Convergence is assessed
    by Aitken acceleration, see McLachlan and Krishnan (2008), Section 4.9.

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
        itermax: int = 1000000,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the discrete-time phase-type distribution fitter.

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
            obs (np.array): Observed realizations of the phase-type distribution.
            censoring (np.array): Specifies censored observations. If
                None, every observation is treated as uncensored.
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
        Fits the discrete-time phase-type distribution using the EM algorithm.

        Args:
            None

        Returns:
            None
        """
        # fit the DPH distribution

        self.estep = edph(nphases=self.nphases)
        self.mstep = mdph(nphases=self.nphases,
                          nobs=self.obs.size)

        iter = 0
        eps = np.inf
        loglik0 = -np.inf
        loglik1 = -np.inf  # the log-likelihood two iterations back
        while iter < self.itermax and eps > self.tolerance:
            
            #E-step
            self.bi,self.ni,self.nij = self.estep.run(obs=self.obs,
                                                      initdist=self.pi,
                                                      phgen=self.phgen,
                                                      exitrates=self.exitrates,
                                                      censoring=self.censoring)
            self.loglikelihood = self.estep.loglikelihood
            
            #M-step
            self.pi,self.phgen,self.exitrates = self.mstep.run(bi=self.bi,
                                                               ni=self.ni,
                                                               nij=self.nij)            
            
            #self.__updatelikelihood()
            step = self.loglikelihood - loglik0

            # Each EM iteration is guaranteed to increase the likelihood
            # (Bladt and Nielsen (2017), p. 675), so a decrease cannot happen in
            # exact arithmetic.
            if step < 0.0:
                warnings.warn(
                    "The log-likelihood decreased by %.3e at iteration %d. The "
                    "EM algorithm cannot decrease it, so this is either "
                    "round-off at a converged fit or an error in the E- or "
                    "M-step. The fit stopped here."
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
        self.__polish()
        # evaluate final loglik
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

        # The observations are sorted, so the censoring array has to be carried
        # along with them: Its rows are matched to the observations by position,
        # and sorting one without the other would silently attach each censoring
        # rule to the wrong observation.
        order = np.argsort(self.obs, kind="stable")
        self.obs = self.obs[order]

        if self.censoring is not None:
            self.censoring = np.asarray(self.censoring, dtype=float)
            if self.censoring.shape != (self.obs.size, 2):
                print(
                    "Error: The censoring array must have one row per observation and two columns."
                )
                self.censoring = None
            else:
                self.censoring = self.censoring[order, :]

        self.initpi = self.initpi.astype(float)
        self.initphgen = self.initphgen.astype(float)
        self.initexitrates = self.initexitrates.astype(float)
        self.identity = np.eye(self.nphases)

        self.pi = np.array(self.initpi, dtype=float).flatten()
        self.phgen = self.initphgen
        self.exitrates = np.array(self.initexitrates, dtype=float).flatten()

        if self.randominit:
            self.__initrandom()
        #self.__updatelikelihood()
        self.loglikelihood = 0.0
        self.__countParameters()

    def __initrandom(self) -> None:
        """
        Randomly initializes the parameters of the discrete-time phase-type distribution.

        Args:
            None

        Returns:
            None
        """

        rnd = rnddph(nphases=self.nphases,
                     initdist=self.pi,
                     phgen=self.phgen,
                     exitrates=self.exitrates)

        self.pi, self.phgen, self.exitrates = rnd.run()

        '''
        nzidx = np.nonzero(self.pi)
        u = np.random.uniform(low=0.0, high=1.0, size=len(nzidx))
        u = u / np.sum(u)
        self.pi[nzidx] = u

        nzidx = np.nonzero(self.exitrates)
        u = np.random.uniform(low=0.0, high=1.0, size=len(nzidx))
        self.exitrates[nzidx] = u

        for i in range(self.nphases):
            nzidx = np.nonzero(np.ravel(self.phgen[i, :]))
            u = np.random.uniform(low=0.0, high=1.0, size=nzidx[0].size)
            u = (u / np.sum(u)) * (1.0 - self.exitrates[i])
            self.phgen[i, nzidx] = u
        '''    


    '''
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
                    Ttprod = self.exitrates.flatten()
                else:
                    Tpow = np.linalg.matrix_power(self.phgen, y - 1)
                    Ttprod = np.ravel(np.matmul(Tpow, self.exitrates))

                piTtprod = np.matmul(self.pi, Ttprod)

                if y >= 2:
                    self.__Kmatrix(y)

            y0 = y

            if piTtprod != 0.0:
                piTpow = np.ravel(np.matmul(self.pi, Tpow))

                for i in range(self.nphases):
                    self.bi[i] += self.pi[i] * Ttprod[i] / piTtprod
                    self.ni[i] += piTpow[i] * self.exitrates[i] / piTtprod

                    if y >= 2:
                        for j in range(self.nphases):
                            self.nij[i, j] += (self.phgen[i, j] * self.Kmat[j, i]) / piTtprod
    '''

    '''
    def __mstep(self) -> None:
        """
        Performs the maximization (M) step of the EM algorithm.

        Args:
            None

        Returns:
            None
        """
        for i in range(self.nphases):
            self.pi[i] = self.bi[i] / self.obs.size

        for i in range(self.nphases):
            sm = self.ni[i]
            for j in range(self.nphases):
                sm += self.nij[i, j]
            self.exitrates[i] = self.ni[i] / sm

        for i in range(self.nphases):
            for j in range(self.nphases):
                sm = self.ni[i]
                for k in range(self.nphases):
                    sm += self.nij[i, k]
                self.phgen[i, j] = self.nij[i, j] / sm
    '''        

    '''
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
    '''
            
    def __getProbMass(self, y: int) -> float:
        """
        Computes the probability mass at observation y.

        Args:
            y (int): Discrete observation value.

        Returns:
            float: The probability mass.
        """
        t = self.exitrates[:, None]
        return np.matmul(
            self.pi, np.matmul(np.linalg.matrix_power(self.phgen, y), t)
        )

    '''
    def __Kmatrix(self, y: int) -> None:
        """
        Computes the matrix-function K for a given observation.

        Args:
            y (int): Observation value.

        Returns:
            None
        """
        if y < 2:
            self.Kmat = np.zeros((self.nphases, self.nphases))
            return

        self.Kmat = np.zeros((self.nphases, self.nphases))

        Ty = np.linalg.matrix_power(self.phgen, y - 2)

        exit_prod = np.matmul(Ty, self.exitrates).flatten()
        pi_mat = self.pi.flatten()

        for k in range(y - 1):
            self.Kmat += np.outer(exit_prod, pi_mat)

            if k < y - 2:
                Ty = np.matmul(Ty, self.phgeninv)
                exit_prod = np.matmul(Ty, self.exitrates).flatten()
                pi_mat = np.matmul(pi_mat, self.phgen).flatten()        
    '''

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
