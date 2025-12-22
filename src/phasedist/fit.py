import sys
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from phasedist.fitcph import fitcph
from phasedist.fitdph import fitdph
from phasedist.dist import dist


class fit:
    """
    Fit continuous or discrete-time phase-type (PH) distributions to data.

    This class provides a unified interface for estimating the parameters of
    continuous (CPH) or discrete (DPH) phase-type distributions using the
    Expectation-Maximization (EM) algorithm. Users may specify initial
    parameters, impose specific structural constraints (e.g., Coxian,
    hyper-exponential), or rely on random initialization.

    The fitted distribution is returned as a `phasedist.dist.dist`
    object that supports evaluation of densities, CDFs, quantiles, and random
    sampling.
    """
    
    def __init__(
        self,
        obs: np.array = None,
        nphases: int = 2,
        dtype: str = "general",
        discrete: bool = False,
        initdist: np.array = None,
        initphgen: np.array = None,
        initexitrates: np.array = None,
        randominit: bool = True,
        seed: int = None,
        tolerance: float = 1e-6,
        itermax: int = 1000000,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the fitting procedure for a phase-type distribution.

        Args:
            obs (np.array):
                Array of observed data points.
            nphases (int, default=2):
                Number of phases in the PH distribution.
            dtype (str, default="general"):
                Distribution type (e.g., "general", "hyperexp").
            discrete (bool, default=False):
                Whether a discrete PH distribution is assumed.
            initdist (np.array, optional):
                Initial distribution vector for custom structure or initialization.
            initphgen (np.array, optional):
                Initial phase-type generator matrix for custom structure or initialization.
            initexitrates (np.array, optional):
                Initial exit rate vector for custom structure or initialization.
            randominit (bool, default=True):
                Whether to use random initialization instead of provided initial parameters.
            seed (int, optional):
                Random seed.
            tolerance (float, default=1e-6):
                Convergence tolerance for the EM algorithm.
            itermax (int, default=1000000):
                Maximum number of iterations.
            verbose (bool, default=False):
                If True, prints progress output during fitting.
                
        Notes
        -----        
        Input validation is performed automatically. If validation fails,
        program execution terminates.        
        """

        # set parameters
        self.obs = obs
        self.nphases = nphases
        self.dtype = dtype
        self.discrete = discrete
        self.initdist = initdist
        self.initphgen = initphgen
        self.initexitrates = initexitrates
        self.randominit = randominit
        self.seed = seed
        self.tolerance = tolerance
        self.itermax = itermax
        self.verbose = verbose

        # checking and fitting
        if self.__checkinputs():
            self.__fitdist()  # fit the parameters
        else:
            sys.exit(1)  # terminate the program

    # ----------------------------------------------------------------------
    #   PUBLIC METHODS
    # ----------------------------------------------------------------------

    def getinitdist(self) -> np.array:
        """
        Return the fitted initial distribution vector.

        Returns:
            np.array: Row vector representing the initial phase probabilities.
        """        
        return self.dist.getinitdist()

    def getphasegen(self) -> np.array:
        """
        Return the fitted phase-type generator matrix.

        Returns:
            np.array: Sub-intensity matrix (CPH) or sub-transition matrix (DPH).
        """
        return self.dist.getphasegen()

    def getexitrates(self) -> np.array:
        """
        Return the fitted exit rate vector.

        Returns:
            np.array: Column vector of exit rates from each phase.
        """
        return self.dist.getexitrates()

    def getmean(self) -> float:
        """
        Return the mean of the fitted PH distribution.

        Returns:
            float: The mean of the distribution.
        """
        return self.dist.getmean()

    def getvar(self) -> float:
        """
        Return the variance of the fitted PH distribution.

        Returns:
            float: The variance of the distribution.
        """
        return self.dist.getvar()

    def getdensity(self, x: float) -> float:
        """
        Evaluate the fitted distribution's density at point x.

        Args:
            x (float): Evaluation point.

        Returns:
            float: Probability density (CPH) or probability mass (DPH).
        """
        return self.dist.getdensity(x)

    def getcumprob(self, x: float) -> float:
        """
        Evaluate the fitted cumulative distribution function at point x.

        Args:
            x (float): Evaluation point.

        Returns:
            float: Cumulative distribution value P(X<=x).
        """
        return self.dist.getcumprob(x)

    def getquantile(self, p: float, tolerance: float = 1e-6) -> int | float:
        """
        Compute the p-quantile of the fitted PH distribution.

        Args:
            p (float): Cumulative probability in the interval (0, 1).
            tolerance (float, default=1e-6): Numerical tolerance for the quantile search.

        Returns:
            int | float: The p-quantile.
        """
        return self.dist.getquantile(p, tolerance)

    def getloglik(self) -> float:
        """
        Return the log-likelihood for the fitted PH model.

        Returns:
            float: Log-likelihood evaluated at the fitted parameters.
        """        
        return self.d.getloglik()

    def getaic(self) -> float:
        """
        Compute Akaike's Information Criterion (AIC).

        Returns:
            float: The AIC value of the fitted model.
        """
        return self.d.getaic()

    def getbic(self) -> float:
        """
        Compute Bayesian Information Criterion (BIC).

        Returns:
            float: The BIC value of the fitted model.
        """
        return self.d.getbic()

    def getdist(self) -> dist:
        """
        Return the fitted PH distribution as a dist object.

        Returns:
            dist: Fully constructed phase-type distribution object.
        """
        return self.dist

    def plot(
        self,
        confint: bool = True,
        confidence: float = 0.95,
        xlabel: str = "x",
        ylabel: str = "CDF",
        title: str = "Empirical and Fitted CDFs",
        labelfitted: str = "Fitted CDF",
        labelempirical: str = "Empirical CDF",
    ) -> None:
        """
        Plot empirical and fitted cumulative distribution functions (CDFs).

        Args:
            confint (bool, default=True): Whether to include confidence intervals for the empirical CDF.
            confidence (float, default=0.95): Confidence level for the intervals.
            xlabel (str, default="x"): Label for the x-axis.
            ylabel (str, default="CDF"): Label for the y-axis.
            title (str, default="Empirical and Fitted CDFs"): Plot title.
            labelfitted (str, default="Fitted CDF"): Legend label for the fitted CDF.
            labelempirical (str, default="Empirical CDF"): Legend label for the empirical CDF.

        Returns:
            None: Displays the plot.
        """

        obssorted = np.sort(self.obs)

        if not self.discrete:
            empcdf = np.arange(1, self.obs.size + 1)
            empcdf_upper = np.zeros(empcdf.size)
            empcdf_lower = np.zeros(empcdf.size)
            for i in range(empcdf.size):
                lims = proportion_confint(
                    empcdf[i], self.obs.size, alpha=(1 - confidence), method="wilson"
                )
                empcdf_upper[i] = lims[1]
                empcdf_lower[i] = lims[0]
            empcdf = empcdf.astype(float)
            empcdf /= float(self.obs.size)
            res = 1000
            theocdf = np.zeros(res)
            x = np.linspace(np.min(obssorted), np.max(obssorted), res)
            for i in range(res):
                theocdf[i] = self.getcumprob(x[i])
        else:
            x = np.arange(np.min(obssorted), np.max(obssorted) + 1)
            empcdf = np.zeros(x.size)
            empcdf_upper = np.zeros(x.size)
            empcdf_lower = np.zeros(x.size)
            theocdf = np.zeros(x.size)
            emp_old = 0
            for i in range(x.size):
                theocdf[i] = self.getcumprob(x[i])
                a = obssorted[obssorted == (i + np.min(obssorted))]
                empcdf[i] = a.size + emp_old
                emp_old = empcdf[i]
                lims = proportion_confint(
                    empcdf[i],
                    self.obs.size,
                    alpha=(1 - confidence),
                    method="binom_test",
                )
                empcdf_upper[i] = lims[1]
                empcdf_lower[i] = lims[0]
            empcdf = empcdf.astype(float)
            empcdf /= float(self.obs.size)

        plt.figure(figsize=(8, 6))

        if not self.discrete:
            plt.plot(x, theocdf, label=labelfitted, lw=1, linestyle="-", color="blue")
            plt.plot(
                obssorted,
                empcdf,
                label=labelempirical,
                lw=1,
                linestyle="-",
                color="red",
            )
            if confint:
                plt.plot(
                    obssorted,
                    empcdf_upper,
                    label="Upper conf. int.",
                    lw=1,
                    linestyle="--",
                    color="red",
                )
                plt.plot(
                    obssorted,
                    empcdf_lower,
                    label="Lower conf. int.",
                    lw=1,
                    linestyle="--",
                    color="red",
                )
        else:
            plt.scatter(x, empcdf, label=labelempirical, lw=1, marker="x", color="red")
            if confint:
                plt.scatter(
                    x,
                    empcdf_upper,
                    label="Upper conf. int.",
                    marker="_",
                    s=100,
                    color="red",
                )
                plt.scatter(
                    x,
                    empcdf_lower,
                    label="Lower conf. int.",
                    marker="_",
                    s=100,
                    color="red",
                )
            plt.scatter(x, theocdf, label=labelfitted, lw=1, marker="x", color="blue")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS
    # ----------------------------------------------------------------------

    def __checkinputs(self) -> bool:
        """
        Validate input parameters prior to fitting.

        Returns:
            bool: True if all inputs are valid; otherwise False.
        """

        # check data types and convert if necesarry
        if isinstance(self.obs, list):
            self.obs = np.array(self.obs)
        elif not isinstance(self.obs, np.ndarray):
            print(
                "Error: Observations can only be specified as a list or a NumPy array."
            )
            return False
        if not isinstance(self.nphases, int) or self.nphases < 1:
            print(
                "Error: The number of phases can only be specified as an integer larger than 0."
            )
            return False
        if not isinstance(self.dtype, str):
            print("Error: The distribution type can only be specified as a string.")
        if not isinstance(self.discrete, bool):
            print("Error: The argument 'discrete' needs to be of type 'bool'.")
        if self.initdist is not None and (
            isinstance(self.initdist, np.ndarray) or isinstance(self.initdist, list)
        ):
            self.initdist = np.matrix(self.initdist)
        elif self.initdist is not None and not isinstance(self.initdist, np.matrix):
            print(
                "Error: The initial distribution can only be specified as a list, NumPy array, or a NumPy matrix."
            )
            return False
        if self.initphgen is not None and (
            isinstance(self.initphgen, np.ndarray) or isinstance(self.initphgen, list)
        ):
            self.initphgen = np.matrix(self.initphgen)
        elif self.initphgen is not None and not isinstance(self.initphgen, np.matrix):
            print(
                "Error: The PH generator can only be specified as a list or a NumPy matrix."
            )
            return False
        if self.initexitrates is not None and (
            isinstance(self.initexitrates, np.ndarray)
            or isinstance(self.initexitrates, list)
        ):
            self.initexitrates = np.transpose(np.matrix(self.initexitrates))
        elif self.initexitrates is not None and not isinstance(
            self.initexitrates, np.matrix
        ):
            print(
                "Error: The exit rate vector can only be specified as a list, NumPy array, or a NumPy matrix."
            )
            return False
        if not isinstance(self.randominit, bool):
            print("Error: The argument 'randominit' needs to be of type 'bool'.")
            return False
        if self.seed is not None and not isinstance(self.seed, int):
            print("Error: The seed can only be specified as an integer.")
            return False
        if not isinstance(self.tolerance, float):
            print("Error: The argument 'tolerance' needs to be of type 'float'.")
            return False
        if not isinstance(self.itermax, float) and not isinstance(self.itermax, int):
            print("Error: The argument 'itermax' needs to be of type 'float' or 'int'.")
            return False
        if not isinstance(self.verbose, bool):
            print("Error: The argument 'verbose' needs to be of type 'bool'.")
            return False
        if self.discrete and self.dtype == "general":
            print(
                "Error: The 'general' option is currently inactivated for discrete PH distributions."
            )
            return False

        # check observations are equal to or greather than zero
        if np.any(self.obs < 0):
            print("Error: The array of observations contains negative values.")
            return False

        # check PH generator and exit rates in case of no random initialization
        if self.dtype == "custom" or not self.randominit:
            self.nphases = self.initphgen.shape[0]
        if not self.randominit:
            if not self.__correctphgen(
                self.initphgen, self.initexitrates
            ) or not self.__correctinitdist(self.initdist):
                return False

        return True

    def __fitdist(self) -> int:
        """
        Fit the PH distribution parameters using the chosen model type and EM algorithm.

        Returns:
            int: 0 if successful, 1 otherwise.
        """

        # set distribution type
        if self.dtype == "general":
            self.__general()
        elif self.dtype == "generlang":
            self.__generlang()
        elif self.dtype == "hyperexp":
            self.__hyperexp()
        elif self.dtype == "coxian":
            self.__coxian()
        elif self.dtype == "gencoxian":
            self.__gencoxian()
        elif self.dtype != "custom":
            print("Error: Unknown distribution type.")
            return 1

        # check for zeros in observations
        obsnonzero, fraczero = self.__checkzeros()

        # fit parameters
        if self.discrete:
            self.d = fitdph(
                obs=obsnonzero,
                initpi=self.initdist,
                initphgen=self.initphgen,
                initexitrates=self.initexitrates,
                randominit=self.randominit,
                seed=self.seed,
                tolerance=self.tolerance,
                itermax=self.itermax,
                verbose=self.verbose,
            )
        else:

            self.d = fitcph(
                obs=obsnonzero,
                initpi=self.initdist,
                initphgen=self.initphgen,
                initexitrates=self.initexitrates,
                randominit=self.randominit,
                seed=self.seed,
                tolerance=self.tolerance,
                itermax=self.itermax,
                verbose=self.verbose,
            )
        self.d.fit()

        # check fitted parameters
        self.__checkfit()

        # adjust for zeros in observations
        self.d.initpi = self.d.initpi * (1 - fraczero)

        # create object for output PH distribution
        self.dist = dist(
            discrete=self.discrete,
            initdist=self.d.getinitdist(),
            phgen=self.d.getphasegen(),
            seed=self.seed,
        )

        if self.fitaccepted:
            return 0
        else:
            print(
                "The PH distribution might contain infeasible or inaccurate parameters."
            )
            return 1

    def __general(self) -> None:
        """
        Initialize parameters for a fully general PH distribution.
        
        Returns:
            None: Initialized parameters.
        """
        
        self.initdist = np.matrix(np.ones((1, self.nphases)))
        self.initphgen = np.matrix(np.ones((self.nphases, self.nphases)))
        self.initexitrates = np.matrix(np.ones((self.nphases, 1)))

    def __generlang(self) -> None:
        """
        Initialize parameters for a generalized Erlang distribution.
        
        Returns:
            None: Initialized parameters.
        """
        
        self.initdist = np.matrix(np.zeros((1, self.nphases)))
        self.initdist[0, 0] = 1

        self.initexitrates = np.matrix(np.zeros((self.nphases, 1)))
        self.initexitrates[self.nphases - 1, 0] = 1

        self.initphgen = np.matrix(np.zeros((self.nphases, self.nphases)))
        for i in range(self.nphases):
            self.initphgen[i, i] = 1
            if i < (self.nphases - 1):
                self.initphgen[i, i + 1] = 1

    def __hyperexp(self) -> None:
        """
        Initialize parameters for a hyper-exponential distribution.
        
        Returns:
            None: Initialized parameters.
        """
        self.initdist = np.matrix(np.ones((1, self.nphases)))

        self.initphgen = np.matrix(np.zeros((self.nphases, self.nphases)))
        self.initexitrates = np.matrix(np.ones((self.nphases, 1)))
        for i in range(self.nphases):
            self.initphgen[i, i] = 1

    def __coxian(self) -> None:
        """
        Initialize parameters for a Coxian distribution.
        
        Returns:
            None: Initialized parameters.
        """
        self.initdist = np.matrix(np.zeros((1, self.nphases)))
        self.initdist[0, 0] = 1

        self.initexitrates = np.matrix(np.ones((self.nphases, 1)))

        self.initphgen = np.matrix(np.zeros((self.nphases, self.nphases)))
        for i in range(self.nphases):
            self.initphgen[i, i] = 1
            if i < (self.nphases - 1):
                self.initphgen[i, i + 1] = 1

    def __gencoxian(self) -> None:
        """
        Initialize parameters for a generalized Coxian distribution.
        
        Returns:
            None: Initialized parameters.
        """
        self.initdist = np.matrix(np.ones((1, self.nphases)))
        self.initexitrates = np.matrix(np.ones((self.nphases, 1)))

        self.initphgen = np.matrix(np.zeros((self.nphases, self.nphases)))
        for i in range(self.nphases):
            self.initphgen[i, i] = 1
            if i < (self.nphases - 1):
                self.initphgen[i, i + 1] = 1

    def __checkzeros(self):
        """
        Detect zero observations and compute their empirical proportion.

        Returns:
            tuple: (nonzero observations, proportion of zeros)
        """
        nz = np.nonzero(self.obs)[0]
        return self.obs[nz], 1 - (nz.size / self.obs.size)

    def __checkfit(self) -> None:
        """
        Check feasibility of the fitted PH parameters.
        
        Returns:
            None: Feasibility check.
        """
        self.fitaccepted = True
        if self.__correctphgen(
            self.d.getphasegen(), self.d.getexitrates()
        ) and self.__correctinitdist(self.d.getinitdist()):
            self.fitaccepted = True
        else:
            self.fitaccepted = False

    def __correctphgen(self, phasegen: np.array, exitrates: np.array) -> bool:
        """
        Validate a PH generator matrix and exit rate vector.

        Args:
        phasegen (np.array): Candidate PH generator matrix.
        exitrates (np.array): Candidate exit rate vector.

        Returns:
            bool: True if the input is feasible.
        """

        # check PH generator
        if phasegen.shape[0] != phasegen.shape[1] or phasegen.shape[0] != self.nphases:
            print(
                "Error: The dimensions of the PH generator does not match the number of phases."
            )
            return False
        if (
            np.where(np.isnan(phasegen))[0].size > 0
            or np.where(np.isinf(phasegen))[0].size > 0
            or np.where(np.isneginf(phasegen))[0].size > 0
        ):
            print("Error: The PH generator contains NaN or/and infinity values.")
            return False
        if np.any(phasegen[~np.eye(self.nphases, dtype=bool)] < 0):
            print("Error: The PH generator contains negative off-diagonal values.")
            return False
        if not self.discrete and np.any(phasegen[np.eye(self.nphases, dtype=bool)] > 0):
            print("Error: The PH generator contains positive diagonal values.")
            return False
        if (
            not self.discrete
            and np.max(abs(np.add(np.sum(phasegen, axis=1), exitrates))) > 1e-6
        ):
            print(
                "Warning: An element of the exit rate vector deviates at least 1e-6 from the absolute row sum of the PH generator."
            )
            return True
        # check exit rates
        if self.nphases != exitrates.size:
            print(
                "Error: The size of the exit rate vector does not match the number of phases."
            )
            return False
        if (
            np.where(np.isnan(exitrates))[0].size > 0
            or np.where(np.isinf(exitrates))[0].size > 0
            or np.where(np.isneginf(exitrates))[0].size > 0
        ):
            print("Error: The exit rate vector contains NaN or/and infinity values.")
            return False
        if np.any(exitrates < 0):
            print("Error: The exit rate vector contains negative values.")
            return False
        return True

    def __correctinitdist(self, initdist: np.array) -> bool:
        """
        Validate an initial distribution vector.

        Args:
            initdist (np.array): Candidate initial distribution.

        Returns:
            bool: True if valid.
        """

        if initdist.size != self.nphases:
            print(
                "Error: The size of the initial distribution does not match the number of phases."
            )
            return False
        if (
            np.where(np.isnan(initdist))[0].size > 0
            or np.where(np.isinf(initdist))[0].size > 0
            or np.where(np.isneginf(initdist))[0].size > 0
        ):
            print(
                "Error: The initial distribution contains NaN or/and infinity values."
            )
            return False
        if np.any(initdist < 0.0):
            print("Error: The initial distribution contains negative values.")
            return False
        if np.abs(np.sum(initdist) - 1.0) > 1e-14:
            print(
                "Warning: Prior to adjusting for zeros in the observations the initial distribution summed to "
                + str(np.sum(initdist))
            )
            return False
        return True
