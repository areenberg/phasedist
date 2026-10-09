import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from phasedist.fitcph import _fitcph
from phasedist.fitdph import _fitdph
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
        itermax: int = 100000,
        fixediter: int = None,
        verbose: bool = False,
        censoring: np.array = None,
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
            itermax (int, default=100000):
                Maximum number of iterations.
            fixediter (int, optional):
                Fixed number of iterations for the EM algorithm.    
            verbose (bool, default=False):
                If True, prints progress output during fitting.
            censoring (np.array, optional):
                An (n_obs x 2) array marking how each observation is censored,
                read as in fitcph and fitdph:
                [nan,nan] uncensored, [nan,t] right-censored, [s,nan]
                left-censored, [s,t] interval-censored. The value in obs is
                ignored for a censored observation.

        Notes
        -----        
        Input validation is performed automatically. If validation fails,
        program execution terminates.        
        """

        # set parameters
        self.obs = obs
        self.censoring = censoring
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
        self.fixediter = fixediter
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
            float: Log-likelihood evaluated at the fitted parameters, over
                every observation including any at zero.
        """
        return self.d.getloglik() + self.atomloglik

    def getaic(self) -> float:
        """
        Compute Akaike's Information Criterion (AIC).

        Returns:
            float: The AIC value of the fitted model.
        """
        return -2.0 * self.getloglik() + 2.0 * self.nparam

    def getbic(self) -> float:
        """
        Compute Bayesian Information Criterion (BIC).

        Returns:
            float: The BIC value of the fitted model.
        """
        return -2.0 * self.getloglik() + self.nparam * np.log(self.obs.size)

    def getdist(self) -> dist:
        """
        Return the fitted PH distribution as a dist object.

        Returns:
            dist: Fully constructed phase-type distribution object.
        """
        return self.dist

    def plot(
        self,
        confint: bool = False,
        confidence: float = 0.95,
        xlabel: str = "x",
        ylabel: str = "CDF",
        title: str = "Empirical and Fitted CDFs",
        labelfitted: str = "Fitted CDF",
        labelempirical: str = "Empirical CDF",
        filename: str = "CDFcheck.png"
    ) -> None:
        """
        Plot empirical and fitted cumulative distribution functions (CDFs).

        The empirical CDF is left out when any observation is censored, the
        values held in obs being placeholders rather than data. Only the fitted
        CDF is then drawn, over the range spanned by the uncensored
        observations and the censoring limits, and a warning says so.

        Args:
            confint (bool, default=False): Whether to add Clopper-Pearson confidence intervals to the empirical CDF.
            confidence (float, default=0.95): Confidence level for the intervals.
            xlabel (str, default="x"): Label for the x-axis.
            ylabel (str, default="CDF"): Label for the y-axis.
            title (str, default="Empirical and Fitted CDFs"): Plot title.
            labelfitted (str, default="Fitted CDF"): Legend label for the fitted CDF.
            labelempirical (str, default="Empirical CDF"): Legend label for the empirical CDF.
            filename (str, default="CDFcheck.png"): Filename of the generated graph.

        Returns:
            None: Writes the figure to filename.
        """

        uncensored = (np.ones(self.obs.size, dtype=bool) if self.censoring is None
                      else np.all(np.isnan(self.censoring), axis=1))
        censored = not np.all(uncensored)

        if censored:
            warnings.warn(
                "Some observations are censored, so the empirical CDF is left out "
                "and only the fitted CDF is drawn.",
                UserWarning,
                stacklevel=2,
            )

            # the range over which something is known about the sample
            known = np.concatenate(
                [self.obs[uncensored], self.censoring[np.isfinite(self.censoring)]]
            )
            lower, upper = float(np.min(known)), float(np.max(known))
        else:
            obssorted = np.sort(self.obs)
            lower, upper = float(obssorted[0]), float(obssorted[-1])

        if not self.discrete:
            x = np.linspace(lower, upper, 1000)
        else:
            x = np.arange(int(lower), int(upper) + 1)

        theocdf = np.zeros(x.size)
        for i in range(x.size):
            theocdf[i] = self.getcumprob(x[i])

        if not censored:
            if self.discrete:
                counts = np.cumsum(
                    [np.count_nonzero(obssorted == value) for value in x]
                )
            else:
                counts = np.arange(1, self.obs.size + 1)

            empcdf = counts / float(self.obs.size)

            if confint:
                empcdf_lower, empcdf_upper = self.__clopperpearson(counts, confidence)

        plt.figure(figsize=(8, 6))

        if not self.discrete:
            plt.plot(x, theocdf, label=labelfitted, lw=1, linestyle="-", color="blue")
            if not censored:
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
            if not censored:
                plt.scatter(
                    x, empcdf, label=labelempirical, lw=1, marker="x", color="red"
                )
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
        plt.savefig(filename)

        # the figure is only written to file, so it is closed rather than left
        # open for a later call to accumulate
        plt.close()

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS
    # ----------------------------------------------------------------------

    def __clopperpearson(self, counts: np.array, confidence: float) -> tuple:
        """
        Returns the Clopper-Pearson interval for every count at once, the beta
        quantiles being vectorized. The ends are set directly, the beta
        distribution having no parameters for them.

        Args:
            counts (ndarray): Number of observations at or below each point.
            confidence (float): Confidence level.

        Returns:
            tuple: The lower and the upper bound (ndarray).
        """

        alpha = 1.0 - confidence
        total = self.obs.size

        lower = beta.ppf(alpha / 2.0, counts, total - counts + 1)
        upper = beta.ppf(1.0 - alpha / 2.0, counts + 1, total - counts)

        lower[counts == 0] = 0.0
        upper[counts == total] = 1.0

        return lower, upper

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
        if self.censoring is not None:
            if isinstance(self.censoring, list):
                self.censoring = np.array(self.censoring, dtype=float)
            elif not isinstance(self.censoring, np.ndarray):
                print(
                    "Error: The censoring array can only be specified as a list or a NumPy array."
                )
                return False
            self.censoring = np.asarray(self.censoring, dtype=float)
            if self.censoring.shape != (self.obs.size, 2):
                print(
                    "Error: The censoring array must have one row per observation and two columns."
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
            self.initdist = np.asarray(self.initdist, dtype=float)
        elif self.initdist is not None:
            print(
                "Error: The initial distribution can only be specified as a list, NumPy array, or a NumPy matrix."
            )
            return False
        if self.initphgen is not None and (
            isinstance(self.initphgen, np.ndarray) or isinstance(self.initphgen, list)
        ):
            self.initphgen = np.asarray(self.initphgen, dtype=float)
        elif self.initphgen is not None:
            print(
                "Error: The PH generator can only be specified as a list or a NumPy matrix."
            )
            return False
        if self.initexitrates is not None and (
            isinstance(self.initexitrates, np.ndarray)
            or isinstance(self.initexitrates, list)
        ):
            self.initexitrates = np.asarray(
                self.initexitrates, dtype=float
            ).reshape(-1, 1)
        elif self.initexitrates is not None:
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
        if self.fixediter is not None and not isinstance(self.fixediter, int):
            print("Error: The argument 'fixediter' needs to be of type 'int'.")
            return False        
        if not isinstance(self.verbose, bool):
            print("Error: The argument 'verbose' needs to be of type 'bool'.")
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

        # check the distribution type is one that is known
        if self.dtype not in ("general", "generlang", "hyperexp", "coxian",
                              "gencoxian", "custom"):
            print("Error: Unknown distribution type.")
            return 1

        # The types below set the structure, that is which elements are to be
        # non-zero, which is what the random initialization needs. They are not
        # starting values, the generator they build having a positive diagonal,
        # so they must not replace a start the user has supplied.
        if self.randominit:
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

        # check if fixed iterations requested
        if self.fixediter is not None and self.fixediter>0:
            self.tolerance=-np.inf
            self.itermax=self.fixediter

        # check for zeros in observations
        obsnonzero, censoringnonzero, fraczero = self.__checkzeros()

        # fit parameters
        if self.discrete:
            self.d = _fitdph(
                obs=obsnonzero,
                censoring=censoringnonzero,
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

            self.d = _fitcph(
                obs=obsnonzero,
                censoring=censoringnonzero,
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

        # adjust for zeros in observations, leaving an atom of that size at
        # zero. The fitted vector is pi; initpi is the start and is spent
        self.d.pi = self.d.pi * (1 - fraczero)

        # The atom makes the model a mixture: an observation is zero with
        # probability p, and otherwise phase-type. Its log-likelihood is
        # therefore the phase-type one over the kept observations, which
        # carry a factor 1-p each, plus the zeros' own term. The estimate of
        # p is the observed proportion, which is one more parameter.
        self.nparam = self.d.nparam
        self.atomloglik = 0.0

        nkept = obsnonzero.size
        nzero = self.obs.size - nkept

        if nzero > 0:
            self.atomloglik = (nzero * np.log(fraczero)
                               + nkept * np.log(1.0 - fraczero))
            self.nparam += 1

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
        
        self.initdist = np.ones((1, self.nphases))
        self.initphgen = np.ones((self.nphases, self.nphases))
        self.initexitrates = np.ones((self.nphases, 1))

    def __generlang(self) -> None:
        """
        Initialize parameters for a generalized Erlang distribution.
        
        Returns:
            None: Initialized parameters.
        """
        
        self.initdist = np.zeros((1, self.nphases))
        self.initdist[0, 0] = 1

        self.initexitrates = np.zeros((self.nphases, 1))
        self.initexitrates[self.nphases - 1, 0] = 1

        self.initphgen = np.zeros((self.nphases, self.nphases))
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
        self.initdist = np.ones((1, self.nphases))

        self.initphgen = np.zeros((self.nphases, self.nphases))
        self.initexitrates = np.ones((self.nphases, 1))
        for i in range(self.nphases):
            self.initphgen[i, i] = 1

    def __coxian(self) -> None:
        """
        Initialize parameters for a Coxian distribution.
        
        Returns:
            None: Initialized parameters.
        """
        self.initdist = np.zeros((1, self.nphases))
        self.initdist[0, 0] = 1

        self.initexitrates = np.ones((self.nphases, 1))

        self.initphgen = np.zeros((self.nphases, self.nphases))
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
        self.initdist = np.ones((1, self.nphases))
        self.initexitrates = np.ones((self.nphases, 1))

        self.initphgen = np.zeros((self.nphases, self.nphases))
        for i in range(self.nphases):
            self.initphgen[i, i] = 1
            if i < (self.nphases - 1):
                self.initphgen[i, i + 1] = 1

    def __checkzeros(self):
        """
        Drop the observations at zero and compute their empirical proportion.
        A censored row is kept whatever its value in obs, that value being
        ignored, and the censoring array is filtered alongside so the rows stay
        matched.

        Returns:
            tuple: (observations, censoring or None, proportion dropped)
        """
        keep = self.obs != 0

        if self.censoring is not None:
            keep = keep | ~np.all(np.isnan(self.censoring), axis=1)
            return (self.obs[keep], self.censoring[keep],
                    1 - (np.count_nonzero(keep) / self.obs.size))

        return self.obs[keep], None, 1 - (np.count_nonzero(keep) / self.obs.size)

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
