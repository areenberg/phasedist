import sys
import warnings

import numpy as np
from scipy.stats import lognorm, norm, gamma, weibull_min, chi2
from scipy.special import gammaincc, gamma as gammafunction
import matplotlib.pyplot as plt
from phasedist.dist import dist
from phasedist.unif import _unif
from phasedist.rndcph import rndcph

class fitcph2dist:
    """
    Fit a continuous-time phase-type distribution to
    a distribution with a continuous density using
    the EM algorithm from p. 681 Bladt and Nielsen (2017).

    The target is discretised onto a grid and the expectation step is evaluated
    in closed form by the block identity of Van Loan (1978). Convergence is
    assessed by Aitken acceleration, see McLachlan and Krishnan (2008), Section
    4.9, and the algorithm itself is accelerated by SQUAREM, see Varadhan and
    Roland (2008).

    References:
        Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in Applied Probability.
        Springer. https://doi.org/10.1007/978-1-4939-7049-0

        McLachlan, G. J., & Krishnan, T. (2008). The EM Algorithm and Extensions (2nd ed.).
        Wiley. https://doi.org/10.1002/9780470191613

        Van Loan, C. (1978). Computing integrals involving the matrix exponential.
        IEEE Transactions on Automatic Control, 23(3), 395-404.
        https://doi.org/10.1109/TAC.1978.1101743

        Varadhan, R., & Roland, C. (2008). Simple and Globally Convergent Methods for
        Accelerating the Convergence of Any EM Algorithm. Scandinavian Journal of
        Statistics, 35(2), 335-353. https://doi.org/10.1111/j.1467-9469.2007.00585.x
    """

    def __init__(
        self,
        nphases: int = 2,
        dtype: str = "general",
        initdist: np.array = None,
        initphgen: np.array = None,
        initexitrates: np.array = None,
        randominit: bool = True,
        seed: int = None,
        tolerance: float = 1e-5,
        truncation: float = 0.99,
        steps: int = 50,
        itermax: int = 100000,
        accelerate: bool = True,
        restarts: int = 3,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the phase-type distribution fitting object.
        
        Args:
            nphases (int): Number of phases.
            dtype (str): Distribution structure type.
            initdist (np.array): Initial distribution vector.
            initphgen (np.array): Initial generator matrix.
            initexitrates (np.array): Initial exit rates.
            randominit (bool): Whether to randomly initialize parameters.
            seed (int): Random seed.
            tolerance (float): Convergence tolerance.
            truncation (float): Truncation level.
            steps (int): Number of integration steps.
            itermax (int): Maximum number of iterations.
            restarts (int): Number of random starting points to screen, the
                best of them then fitted properly. Ignored for a supplied start.
            accelerate (bool): Whether to accelerate the EM algorithm by SQUAREM.
            verbose (bool): Verbosity flag.
            
        Returns:
            None
        """
        
        self.nphases = nphases
        self.dtype = dtype
        self.initdist = initdist
        self.initphgen = initphgen
        self.initexitrates = initexitrates
        self.randominit = randominit
        self.seed = seed
        self.itermax = itermax
        self.accelerate = accelerate
        self.restarts = restarts
        self.tolerance = tolerance
        self.truncation = truncation
        self.disttype = None
        self.verbose = verbose
        self.dist = None
        self.steps = steps  # number of steps in the numerical integration

        # uniformization stays accurate when the rates are nearly equal
        self.__unif = _unif(tolerance=1e-14)

        # checking and fitting
        if self.__checkinputs():
            self.__makedist()  # fit the parameters
        else:
            sys.exit(1)  # terminate the program

    def __exponential(self, matrix: np.array, x: float) -> np.array:
        """
        Returns exp(matrix * x) by uniformization.

        Args:
            matrix (ndarray): The matrix to exponentiate.
            x (float): The time to exponentiate over.

        Returns:
            ndarray: The matrix exponential.
        """

        return self.__unif.run(matrix, float(x))[0]

    # ----------------------------------------------------------------------
    #   PUBLIC METHODS
    # ----------------------------------------------------------------------

    def lognorm(self, mu: float = None, sigma: float = None, mean: float = None, var: float = None) -> None:
        """
        Configures approximation to a log-normal distribution.
        
        Args:
            mu (float): Mean of underlying normal distribution.
            sigma (float): Standard deviation of underlying normal distribution.
            mean (float): Mean of the log-normal distribution.
            var (float): Variance of the log-normal distribution.
        
        Returns:
            None
        """     
        if mu is None and mean is not None:
            if mean <= 0:
                print("Error: 'mean<=0' is infeasible for the lognormal distribution.")
                sys.exit(1)
            self.param1 = np.log(np.power(mean, 2) / np.sqrt(np.power(mean, 2) + var))
            self.param2 = np.log(1 + var / np.power(mean, 2))
        elif mu is not None and mean is None:
            self.param1 = mu
            self.param2 = np.power(sigma, 2)
        self.disttype = "lognorm"
        self.__initialize()

    def norm(self, mu: float = None, sigma: float = None) -> None:
        """
        Configures approximation to a truncated normal distribution.
        
        Args:
            mu (float): Mean of the normal distribution.
            sigma (float): Standard deviation of the normal distribution.
        
        Returns:
            None
        """
        self.param1 = mu
        self.param2 = sigma
        self.disttype = "norm"
        self.__initialize()

    def gamma(self, shape: float = None, scale: float = None, rate: float = None) -> None:
        """
        Configures approximation to a gamma distribution.
        
        Args:
            shape (float): Shape parameter.
            scale (float): Scale parameter.
            rate (float): Rate parameter.
        
        Returns:
            None
        """
        self.param1 = shape
        if scale is None:
            self.param2 = 1 / rate
        elif rate is None:
            self.param2 = scale
        self.disttype = "gamma"
        self.__initialize()

    def chisq(self, df: int = None) -> None:
        """
        Configures approximation to a chi-squared distribution.
        
        Args:
            df (int): Degrees of freedom.
        
        Returns:
            None
        """
        self.param1 = df
        self.disttype = "chisq"
        self.__initialize()

    def weibull(self, shape: float = None, scale: float = None) -> None:
        """
        Configures approximation to a Weibull distribution.
        
        Args:
            shape (float): Shape parameter.
            scale (float): Scale parameter.
        
        Returns:
            None
        """
        self.param1 = shape
        self.param2 = scale
        self.disttype = "weibull"
        self.__initialize()

    def phasedist(self, initdist: np.array, phgen: np.array) -> None:
        """
        Configures approximation to an existing phase-type distribution.
        
        Args:
            initdist (np.array): Initial distribution vector.
            phgen (np.array): Generator matrix.
        
        Returns:
            None
        """
        self.param1 = initdist
        self.param2 = phgen
        self.disttype = "ph"
        self.__initialize()

    def percentiles(self, cumprobs: np.array = None, x: np.array = None) -> None:
        """
        Configures approximation using (empirical) cumulative probabilities.
        
        Args:
            cumprobs (np.array): Cumulative probabilities.
            x (np.array): Corresponding values.
        
        Returns:
            None
        """
        self.param1 = cumprobs
        self.param2 = x
        self.disttype = "per"
        self.__initialize()

    def fit(self) -> int:
        """
        Fits the phase-type distribution using the EM algorithm.
        
        Args:
            None
        
        Returns:
            int: Status code.
        """
        if self.disttype is None:
            print("Error: Select a distribution for the approximation.")
            return 1
        # the presets start unnormalized, which the first M-step would
        # otherwise correct as an apparent drop in the log-likelihood
        self.pi = self.pi / np.sum(self.pi)

        # a supplied start is the one the caller asked for, and a tolerance
        # that is not finite asks for a fixed number of iterations and no more
        if (self.restarts > 1 and self.randominit
                and np.isfinite(self.tolerance)):
            self.__screen()

        iter = self.__emloop(self.tolerance, self.itermax)

        # a tolerance that is not finite means the caller asked for a fixed
        # number of iterations, so stopping at itermax is the intent
        if (iter >= self.itermax and self.eps > self.tolerance
                and np.isfinite(self.tolerance)):
            warnings.warn(
                "Algorithm terminated with iter==itermax. Results might be "
                "misleading. After %d iterations the estimated distance to the "
                "limit of the log-likelihood was still %.3e, against a "
                "tolerance of %.3e."
                % (iter, float(self.eps), float(self.tolerance)),
                RuntimeWarning,
                stacklevel=2)
        # the loop reads it where each iteration starts, so once more here
        self.__estep()

        # create object for output PH distribution
        self.dist = dist(
            discrete=False, initdist=self.pi, phgen=self.phgen, seed=self.seed
        )

        return 0

    def plot(
        self,
        filename: str = "compare.png"
    ) -> None:
        """
        Plots the fitted distribution and target distribution.
        
        Args:
            filename (str, default="compare.png"): Filename of the generated graph.
        
        Returns:
            None
        """
        if self.dist is None:
            print("Error: Fit the distribution before plotting it.")
            return None

        x = np.linspace(0.0, self.y.max(), 500)
        dist_pdf = np.zeros(len(x))
        ph_pdf = np.zeros(len(x))
        for i in range(len(x)):
            if self.disttype == "lognorm":
                # param2 is the variance, scipy wants the standard deviation
                dist_pdf[i] = lognorm.pdf(
                    x[i], np.sqrt(self.param2), scale=np.exp(self.param1)
                )
            elif self.disttype == "gamma":
                dist_pdf[i] = gamma.pdf(x[i], self.param1, scale=self.param2)
            elif self.disttype == "weibull":
                dist_pdf[i] = weibull_min.pdf(x[i], self.param1, scale=self.param2)
            elif self.disttype == "chisq":
                dist_pdf[i] = chi2.pdf(x[i], self.param1)
            elif self.disttype == "ph":
                d = dist(discrete=False, initdist=self.param1, phgen=self.param2)
                dist_pdf[i] = d.getdensity(x[i])
            elif self.disttype == "norm":
                dist_pdf[i] = self.__normtruncdensity(x[i])
            elif self.disttype == "per":
                dist_pdf[i] = self.__perdensity(x[i])
            ph_pdf[i] = self.getdensity(x[i])

        # make plot
        plt.figure(figsize=(10, 6))
        plt.plot(x, dist_pdf, label="True density", color="blue")
        plt.plot(x, ph_pdf, label="Approx. density", color="red", linestyle="--")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.title("Approximation validation")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

        return None

    def __normtruncdensity(self, x: float) -> float:
        """
        Density of the normal target, which is truncated to the positive half
        line, so it is rescaled by the mass the truncation leaves.

        Args:
            x (float): Evaluation point.

        Returns:
            float: The density at x.
        """

        if x < 0.0:
            return 0.0

        return float(norm.pdf((x - self.param1) / self.param2) / self.param2
                     / norm.sf(-self.param1 / self.param2))

    def __perdensity(self, x: float) -> float:
        """
        Returns the density of the percentile target at x, which is constant
        between the supplied points and zero outside them.

        Args:
            x (float): Evaluation point.

        Returns:
            float: The density at x.
        """

        if x > self.param2[-1] or x < 0.0:
            return 0.0

        idx = int(np.min(np.where(self.param2 >= x)))

        if idx == 0:
            return float(self.param1[0] / self.param2[0])

        return float((self.param1[idx] - self.param1[idx - 1])
                     / (self.param2[idx] - self.param2[idx - 1]))

    def getinitdist(self) -> np.array:
        """
        Returns the initial distribution vector.
        
        Args:
            None
        
        Returns:
            np.array: Initial distribution.
        """
        if self.dist is not None:
            return self.dist.getinitdist()
        else:
            return np.nan

    def getphasegen(self) -> np.array:
        """
        Returns the phase generator matrix.
        
        Args:
            None
        
        Returns:
            np.array: Generator matrix.
        """
        if self.dist is not None:
            return self.dist.getphasegen()
        else:
            return np.nan

    def getexitrates(self) -> np.array:
        """
        Returns the exit rate vector.
        
        Args:
            None
        
        Returns:
            np.array: Exit rates.
        """
        if self.dist is not None:
            return self.dist.getexitrates()
        else:
            return np.nan

    def getmean(self) -> float:
        """
        Returns the mean of the fitted distribution.
        
        Args:
            None
        
        Returns:
            float: Mean.
        """
        if self.dist is not None:
            return self.dist.getmean()
        else:
            return np.nan

    def getvar(self) -> float:
        """
        Returns the variance of the fitted distribution.
        
        Args:
            None
        
        Returns:
            float: Variance.
        """
        if self.dist is not None:
            return self.dist.getvar()
        else:
            return np.nan

    def getdensity(self, x: float) -> float:
        """
        Returns the distribution's density, f(x).
        
        Args:
            x (float): The distribution's density will be computed at time of x.
        
        Returns:
            float: The computed density.
        """
        if self.dist is not None:
            return self.dist.getdensity(x)
        else:
            return np.nan

    def getcumprob(self, x: float) -> float:
        """
        Returns the cumulative distribution value at x.
        
        Args:
            x (float): Evaluation point.
        
        Returns:
            float: Cumulative probability.
        """
        if self.dist is not None:
            return self.dist.getcumprob(x)
        else:
            return np.nan

    def getquantile(self, p: float, tolerance: float = 1e-9) -> float:
        """
        Returns the quantile corresponding to probability p.
        
        Args:
            p (float): Probability level.
            tolerance (float): Numerical tolerance.
        
        Returns:
            float: Quantile value.
        """
        if self.dist is not None:
            return self.dist.getquantile(p, tolerance)
        else:
            return np.nan

    def getdist(self) -> dist:
        """
        Returns the fitted phase-type distribution object.
        
        Args:
            None
        
        Returns:
            dist: Phase-type distribution.
        """
        return self.dist

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS
    # ----------------------------------------------------------------------

    def __checkinputs(self) -> bool | int:
        """
        Checks feasibility and validity of all input parameters.
        
        Args:
            None
        
        Returns:
            bool | int: True if inputs are valid, False or 0 otherwise.
        """
        if not isinstance(self.nphases, int) or self.nphases < 1:
            print(
                "Error: The number of phases can only be specified as an integer larger than 0."
            )
            return 0
        if not isinstance(self.dtype, str):
            print("Error: The distribution type can only be specified as a string.")
            return False
        if self.dtype not in ("general", "generlang", "hyperexp", "coxian",
                              "gencoxian", "custom"):
            print("Error: Unknown distribution type '%s'." % self.dtype)
            return False
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
        if (not isinstance(self.restarts, (int, np.integer))
                or isinstance(self.restarts, bool) or self.restarts < 1):
            print("Error: The number of restarts can only be specified as an integer larger than 0.")
            return False
        if not isinstance(self.accelerate, bool):
            print("Error: The argument 'accelerate' needs to be of type 'bool'.")
            return False
        if not isinstance(self.verbose, bool):
            print("Error: The argument 'verbose' needs to be of type 'bool'.")
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

    def __correctphgen(self, phasegen: np.array, exitrates: np.array) -> bool:
        """
        Checks feasibility of the PH generator and exit rate vector.
        
        Args:
            phasegen (np.array): Phase-type generator matrix.
            exitrates (np.array): Exit rate vector.
        
        Returns:
            bool: True if feasible, False otherwise.
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
        if np.any(phasegen[np.eye(self.nphases, dtype=bool)] > 0):
            print("Error: The PH generator contains positive diagonal values.")
            return False
        if np.max(abs(np.add(np.sum(phasegen, axis=1), exitrates))) > 1e-6:
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
        Checks feasibility of the initial distribution.
        
        Args:
            initdist (np.array): Initial distribution vector.
        
        Returns:
            bool: True if feasible, False otherwise.
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

    def __makedist(self) -> None | int:
        """
        Initializes distribution structure based on the specified type.
        
        Args:
            None
        
        Returns:
            None | int: None if successful, error code otherwise.
        """
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

    def __general(self) -> None:
        """
        Initializes parameters for a general phase-type distribution.
        
        Args:
            None
        
        Returns:
            None
        """
        self.initdist = np.ones((1, self.nphases))
        self.initphgen = np.ones((self.nphases, self.nphases))
        self.initexitrates = np.ones((self.nphases, 1))

    def __generlang(self) -> None:
        """
        Initializes parameters for a generalized Erlang distribution.
        
        Args:
            None
        
        Returns:
            None
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
        Initializes parameters for a hyper-exponential distribution.
        
        Args:
            None
        
        Returns:
            None
        """
        self.initdist = np.ones((1, self.nphases))

        self.initphgen = np.zeros((self.nphases, self.nphases))
        self.initexitrates = np.ones((self.nphases, 1))
        for i in range(self.nphases):
            self.initphgen[i, i] = 1

    def __coxian(self) -> None:
        """
        Initializes parameters for a Coxian distribution.
        
        Args:
            None
        
        Returns:
            None
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
        Initializes parameters for a generalized Coxian distribution.
        
        Args:
            None
        
        Returns:
            None
        """
        self.initdist = np.ones((1, self.nphases))
        self.initexitrates = np.ones((self.nphases, 1))

        self.initphgen = np.zeros((self.nphases, self.nphases))
        for i in range(self.nphases):
            self.initphgen[i, i] = 1
            if i < (self.nphases - 1):
                self.initphgen[i, i + 1] = 1

    def __initialize(self) -> None:
        """
        Initializes parameters, random values, and numerical integration grid.
        
        Args:
            None
        
        Returns:
            None
        """        
        if self.seed is not None:
            np.random.seed(self.seed)

        # convert data types
        self.steps = int(self.steps)
        self.initpi = self.initdist.astype(float)
        self.initphgen = self.initphgen.astype(float)
        self.initexitrates = self.initexitrates.astype(float)

        # copied because the EM writes in place, which would otherwise spend
        # the stored structure
        self.pi = np.array(self.initdist, dtype=float).flatten()
        self.phgen = np.array(self.initphgen, dtype=float)
        self.exitrates = np.array(self.initexitrates, dtype=float).flatten()

        # initialize with random parameters
        # accounting for the specified structure
        if self.randominit:
            try:
                rnd = rndcph(
                    nphases=self.nphases,
                    initdist=self.pi,
                    phgen=self.phgen,
                    exitrates=self.exitrates,
                )
                self.pi, self.phgen, self.exitrates = rnd.run()
            except ValueError as error:
                print("Error: %s" % error)
                sys.exit(1)

        # create the cumulated probabilities and evaluation points
        self.__computeyvector()

    def __estep(self) -> None:
        """
        Performs the expectation step of the EM algorithm. The integral over
        the time of the jump is the upper right block of exp(By) for
        B = [[T, t pi], [0, T]], see Van Loan (1978).

        Args:
            None

        Returns:
            None
        """

        nphases = self.nphases

        block = np.zeros((2 * nphases, 2 * nphases))
        block[:nphases, :nphases] = self.phgen
        block[nphases:, nphases:] = self.phgen
        block[:nphases, nphases:] = np.outer(self.exitrates, self.pi)

        expblock = self.__unif.run(block, np.asarray(self.y, dtype=float))

        eTy = expblock[:, :nphases, :nphases]
        # transposed so the phase the process leaves is on the first axis
        integral = np.transpose(expblock[:, :nphases, nphases:], (0, 2, 1))

        eTyt = np.matmul(eTy, self.exitrates)
        pieTy = np.matmul(self.pi, eTy)

        # the fitted density at each cell, which normalizes every term
        density = np.sum(pieTy * self.exitrates, axis=1)
        weights = self.hy / density

        # the discretized log-likelihood, the quantity the EM maximizes
        self.loglikelihood = float(np.sum(self.hy * np.log(density)))

        self.bi = self.pi * np.matmul(weights, eTyt)
        self.ni = self.exitrates * np.matmul(weights, pieTy)

        expected = np.einsum("k,kij->ij", weights, integral)

        self.zi = np.diag(expected).copy()
        self.nij = self.phgen * expected
        np.fill_diagonal(self.nij, 0.0)

    def __emloop(self, tolerance: float, itermax: int) -> int:
        """
        Runs the EM algorithm until the estimated distance to the limit of the
        log-likelihood falls below a tolerance.

        Args:
            tolerance (float): The distance to stop at.
            itermax (int): Largest number of iterations to spend.

        Returns:
            int: The iterations spent.
        """

        iter = 0
        self.eps = np.inf
        loglik0 = -np.inf
        loglik1 = -np.inf

        while iter < itermax and self.eps > tolerance:
            if self.accelerate:
                # an accelerated iteration spends two of the plain ones
                loglikelihood = self.__squaremstep()
                spent = 2
            else:
                loglikelihood = self.__emstep()
                spent = 1

            step = loglikelihood - loglik0

            # Each EM iteration is guaranteed to increase the likelihood
            if step < 0.0:
                warnings.warn(
                    "The log-likelihood decreased by %.3e at iteration %d."
                    % (abs(float(step)), iter + 1),
                    RuntimeWarning,
                    stacklevel=2)

            # the max() keeps a transiently small rate, before the EM reaches
            # its linear regime, from stopping the fit short
            self.eps = step
            denominator = loglik0 - loglik1
            if np.isfinite(denominator) and denominator > 0.0:
                rate = step / denominator
                if 0.0 < rate < 1.0:
                    self.eps = max(step, step * rate / (1.0 - rate))

            loglik1 = loglik0
            loglik0 = loglikelihood
            iter += spent
            if self.verbose and iter % 10 == 0:
                d = dist(discrete=False, initdist=self.pi, phgen=self.phgen)
                print(
                    "iter =",
                    iter,
                    "  eps =",
                    self.eps,
                    "  mean =",
                    d.getmean(),
                    "  var =",
                    d.getvar(),
                )

        return iter

    def __screen(self) -> None:
        """
        Runs several random starts to a loose tolerance and keeps the one
        reaching the highest log-likelihood, which the caller fits properly.

        Args:
            None

        Returns:
            None
        """

        tolerance = max(self.tolerance, 1e-4)

        best = None
        bestloglik = -np.inf

        for attempt in range(self.restarts):
            if attempt > 0:
                # continuing the stream, since reseeding repeats the first start
                rnd = rndcph(nphases=self.nphases,
                             initdist=np.array(self.initdist, dtype=float).flatten(),
                             phgen=np.array(self.initphgen, dtype=float),
                             exitrates=np.array(self.initexitrates, dtype=float).flatten())
                self.pi, self.phgen, self.exitrates = rnd.run()
                self.pi = self.pi / np.sum(self.pi)

            self.__emloop(tolerance, self.itermax)
            self.__estep()

            if self.loglikelihood > bestloglik:
                bestloglik = self.loglikelihood
                best = self.__pack()

        self.__unpack(best)

        return None

    def __pack(self) -> np.array:
        """
        Returns the parameters as one vector, the generator's diagonal left out
        since it is minus the rest of its row.

        Args:
            None

        Returns:
            ndarray: The initial distribution, the exit rates and the
                off-diagonal generator entries, in that order.
        """

        offdiagonal = ~np.eye(self.nphases, dtype=bool)

        return np.concatenate((self.pi, self.exitrates,
                               self.phgen[offdiagonal]))

    def __unpack(self, values: np.array) -> None:
        """
        Sets the parameters from the vector __pack returns, rebuilding the
        diagonal and normalizing the initial distribution.

        Args:
            values (ndarray): The parameters as one vector.

        Returns:
            None
        """

        nphases = self.nphases
        offdiagonal = ~np.eye(nphases, dtype=bool)

        self.pi = values[:nphases] / np.sum(values[:nphases])
        self.exitrates = np.copy(values[nphases:2 * nphases])

        self.phgen = np.zeros((nphases, nphases))
        self.phgen[offdiagonal] = values[2 * nphases:]
        np.fill_diagonal(self.phgen,
                         -(self.phgen.sum(axis=1) + self.exitrates))

        return None

    def __isfeasible(self, values: np.array) -> bool:
        """
        Returns whether a vector of parameters describes a phase-type
        distribution.

        Args:
            values (ndarray): The parameters as one vector.

        Returns:
            bool: True when the parameters are usable.
        """

        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            return False

        nphases = self.nphases

        if np.sum(values[:nphases]) <= 0.0:
            return False

        # every phase must be possible to leave, or the Green matrix does not exist
        rates = np.copy(values[nphases:2 * nphases])
        offdiagonal = np.zeros((nphases, nphases))
        offdiagonal[~np.eye(nphases, dtype=bool)] = values[2 * nphases:]

        return bool(np.all(rates + offdiagonal.sum(axis=1) > 0.0))

    def __emstep(self) -> float:
        """
        Performs one EM iteration.

        Args:
            None

        Returns:
            float: The log-likelihood at the parameters it started from.
        """

        self.__estep()
        loglikelihood = self.loglikelihood
        self.__mstep()

        return loglikelihood

    def __loglikat(self, values: np.array) -> float:
        """
        Returns the log-likelihood at a vector of parameters, leaving the
        object holding them.

        Args:
            values (ndarray): The parameters as one vector.

        Returns:
            float: The log-likelihood.
        """

        self.__unpack(values)
        self.__estep()

        return self.loglikelihood

    def __squaremstep(self) -> float:
        """
        Performs one accelerated iteration, moving the parameters to
        theta - 2 alpha r + alpha^2 v for the step r of two EM iterations and
        its change v, see Varadhan and Roland (2008). At alpha = -1 this is the
        second EM iterate exactly, so backtracking towards it cannot lower the
        log-likelihood.

        Args:
            None

        Returns:
            float: The log-likelihood at the parameters it started from.
        """

        start = self.__pack()
        loglikelihood = self.__emstep()
        first = self.__pack()
        self.__emstep()
        second = self.__pack()

        r = first - start
        v = second - first - r
        lengthv = float(np.linalg.norm(v))

        if lengthv > 0.0:
            alpha = -float(np.linalg.norm(r)) / lengthv
            plain = self.__loglikat(second)

            while alpha < -1.001:
                candidate = start - 2.0 * alpha * r + alpha * alpha * v
                if (self.__isfeasible(candidate)
                        and self.__loglikat(candidate) >= plain):
                    # one more iteration, whose E-step __loglikat has just run
                    self.__mstep()
                    return loglikelihood
                alpha = (alpha - 1.0) / 2.0

        self.__unpack(second)

        return loglikelihood

    def __mstep(self) -> None:
        """
        Performs the maximization step of the EM algorithm.
        
        Args:
            None
        
        Returns:
            None
        """

        # update the initial distribution
        for i in range(self.nphases):
            self.pi[i] = self.bi[i]
        self.pi = self.pi / np.sum(self.pi)

        # update the exit rates
        for i in range(self.nphases):
            self.exitrates[i] = self.ni[i] / self.zi[i]

        # update the PH generator
        for i in range(self.nphases):
            sm = self.exitrates[i]
            for j in range(self.nphases):
                if j != i:
                    self.phgen[i, j] = self.nij[i, j] / self.zi[i]
                    sm += self.phgen[i, j]
            self.phgen[i, i] = -sm

    def __computeyvector(self) -> None:
        """
        Computes evaluation points and probability weights for numerical integration.
        
        Args:
            None
        
        Returns:
            None
        """

        # make evaluation points
        if self.disttype == "lognorm":
            # param2 is the variance, scipy wants the standard deviation
            self.y = np.linspace(
                0,
                lognorm.ppf(self.truncation, np.sqrt(self.param2),
                            scale=np.exp(self.param1)),
                self.steps + 1,
            )
        elif self.disttype == "gamma":
            self.y = np.linspace(
                0,
                gamma.ppf(self.truncation, self.param1, scale=self.param2),
                self.steps + 1,
            )
        elif self.disttype == "norm":
            self.y = np.linspace(
                0, self.__normtruncquantfun(self.param1, self.param2), self.steps + 1
            )
        elif self.disttype == "weibull":
            self.y = np.linspace(
                0,
                weibull_min.ppf(self.truncation, self.param1, scale=self.param2),
                self.steps + 1,
            )
        elif self.disttype == "chisq":
            self.y = np.linspace(
                0, chi2.ppf(self.truncation, self.param1), self.steps + 1
            )
        elif self.disttype == "ph":
            d = dist(discrete=False, initdist=self.param1, phgen=self.param2)
            self.y = np.linspace(0, d.getquantile(self.truncation), self.steps + 1)
        elif self.disttype == "per":
            self.y = np.linspace(0, np.max(self.param2), self.steps + 1)

        # compute cumulated probability segments
        self.hy = np.zeros(self.steps)
        for i in range(self.steps):
            if self.disttype == "lognorm":
                self.hy[i] = self.__lognorm_dcdf(self.y[i], self.y[i + 1])
            elif self.disttype == "gamma":
                self.hy[i] = self.__gamma_dcdf(self.y[i], self.y[i + 1])
            elif self.disttype == "norm":
                self.hy[i] = self.__normtrunc_dcdf(self.y[i], self.y[i + 1])
            elif self.disttype == "weibull":
                self.hy[i] = self.__weib_dcdf(self.y[i], self.y[i + 1])
            elif self.disttype == "chisq":
                self.hy[i] = self.__chisq_dcdf(self.y[i], self.y[i + 1])
            elif self.disttype == "ph":
                self.hy[i] = self.__ph_dcdf(self.y[i], self.y[i + 1])
            elif self.disttype == "per":
                self.hy[i] = self.__per_dcdf(self.y[i], self.y[i + 1])

        # each segment sits at its own conditional mean, which makes the mean
        # of the discretised target exact, and the tail above the truncation
        # point is one further cell at the same quantity
        edges = self.y
        truncationpoint = edges[-1]

        cellmeans = self.__cellmeans(edges)
        self.y = (0.5 * (edges[:-1] + edges[1:]) if cellmeans is None
                  else cellmeans)

        tailmean = self.__tailmean(truncationpoint)
        tailmass = 1.0 - np.sum(self.hy)

        if tailmean is not None and tailmass > 0.0:
            self.y = np.append(self.y, tailmean)
            self.hy = np.append(self.hy, tailmass)

        # the segments plus the tail cell, which is not the same as steps
        self.ncells = self.y.size

    def __cellmeans(self, edges: np.array) -> np.array:
        """
        Returns E[X | a < X <= b] for every segment of the grid, taken as the
        difference of the partial expectations above its two ends.

        Args:
            edges (ndarray): The segment boundaries, one more than the cells.

        Returns:
            ndarray: One conditional mean per segment, or None when the target
                says nothing about where the mass sits inside a segment.
        """

        if self.__tailmean(edges[0]) is None:
            return None

        # the mass above each boundary, taken from the segment masses so that
        # it agrees with them exactly
        above = 1.0 - np.concatenate(([0.0], np.cumsum(self.hy)))
        above = np.maximum(above, 0.0)

        partial = np.array([self.__tailmean(edge) for edge in edges]) * above

        with np.errstate(invalid="ignore", divide="ignore"):
            means = -np.diff(partial) / self.hy

        # a segment carrying no mass has no conditional mean, and one in the
        # far tail can lose its leading digits to the subtraction
        midpoints = 0.5 * (edges[:-1] + edges[1:])
        usable = (np.isfinite(means) & (self.hy > 0.0)
                  & (means >= edges[:-1]) & (means <= edges[1:]))

        return np.where(usable, means, midpoints)

    def __tailmean(self, q: float) -> float:
        """
        Returns E[X | X > q], where the mass left by the truncation is placed.

        Args:
            q (float): The truncation point.

        Returns:
            float: E[X | X > q], or None.
        """

        if self.disttype == "lognorm":
            if q <= 0.0:
                # nothing is cut off, so the conditional mean is the mean
                return float(np.exp(self.param1 + self.param2 / 2.0))
            sigma = np.sqrt(self.param2)
            upper = (np.log(q) - self.param1) / sigma
            return float(np.exp(self.param1 + self.param2 / 2.0)
                         * norm.sf(upper - sigma) / norm.sf(upper))

        if self.disttype in ("gamma", "chisq"):
            if self.disttype == "gamma":
                shape, scale = self.param1, self.param2
            else:
                shape, scale = self.param1 / 2.0, 2.0
            return float(shape * scale * gammaincc(shape + 1.0, q / scale)
                         / gammaincc(shape, q / scale))

        if self.disttype == "weibull":
            scaled = np.power(q / self.param2, self.param1)
            order = 1.0 + 1.0 / self.param1
            return float(self.param2 * gammafunction(order)
                         * gammaincc(order, scaled) / np.exp(-scaled))

        if self.disttype == "norm":
            upper = (q - self.param1) / self.param2
            return float(self.param1
                         + self.param2 * norm.pdf(upper) / norm.sf(upper))

        if self.disttype == "ph":
            # the life left beyond q is phase-type with the same generator,
            # started from where the process is at q
            alive = np.matmul(np.asarray(self.param1, dtype=float).ravel(),
                              self.__exponential(self.param2, q))
            green = np.linalg.inv(-np.asarray(self.param2, dtype=float))
            return float(q + np.sum(np.matmul(alive, green)) / np.sum(alive))

        return None

    def __lognorm_dcdf(self, x0: float, x1: float) -> float:
        """
        Computes cumulative probability between two points for a log-normal distribution.
        
        Args:
            x0 (float): Lower bound.
            x1 (float): Upper bound.
        
        Returns:
            float: Cumulative probability.
        """
        if x0 == 0:
            return norm.cdf((np.log(x1) - self.param1) / np.sqrt(self.param2))
        else:
            return norm.cdf(
                (np.log(x1) - self.param1) / np.sqrt(self.param2)
            ) - norm.cdf((np.log(x0) - self.param1) / np.sqrt(self.param2))

    def __gamma_dcdf(self, x0: float, x1: float) -> float:
        """
        Computes cumulative probability between two points for a gamma distribution.
        
        Args:
            x0 (float): Lower bound.
            x1 (float): Upper bound.
        
        Returns:
            float: Cumulative probability.
        """
        if x0 == 0:
            return gamma.cdf(x1, self.param1, scale=self.param2)
        else:
            return gamma.cdf(x1, self.param1, scale=self.param2) - gamma.cdf(
                x0, self.param1, scale=self.param2
            )

    def __normtrunc_dcdf(self, x0: float, x1: float) -> float:
        """
        Computes cumulative probability between two points for a truncated normal distribution.
        
        Args:
            x0 (float): Lower bound.
            x1 (float): Upper bound.
        
        Returns:
            float: Cumulative probability.
        """
        return (
            norm.cdf((x1 - self.param1) / self.param2)
            - norm.cdf((x0 - self.param1) / self.param2)
        ) / (1 - norm.cdf((-self.param1) / self.param2))

    def __weib_dcdf(self, x0: float, x1: float) -> float:
        """
        Computes cumulative probability between two points for a Weibull distribution.
        
        Args:
            x0 (float): Lower bound.
            x1 (float): Upper bound.
        
        Returns:
            float: Cumulative probability.
        """
        if x0 == 0:
            return weibull_min.cdf(x1, self.param1, scale=self.param2)
        else:
            return weibull_min.cdf(
                x1, self.param1, scale=self.param2
            ) - weibull_min.cdf(x0, self.param1, scale=self.param2)

    def __chisq_dcdf(self, x0: float, x1: float) -> float:
        """
        Computes cumulative probability between two points for a chi-square distribution.
        
        Args:
            x0 (float): Lower bound.
            x1 (float): Upper bound.
        
        Returns:
            float: Cumulative probability.
        """
        if x0 == 0:
            return chi2.cdf(x1, self.param1)
        else:
            return chi2.cdf(x1, self.param1) - chi2.cdf(x0, self.param1)

    def __ph_dcdf(self, x0: float, x1: float) -> float:
        """
        Computes cumulative probability between two points for a PH distribution.
        
        Args:
            x0 (float): Lower bound.
            x1 (float): Upper bound.
        
        Returns:
            float: Cumulative probability.
        """
        if x0 == 0:
            return 1 - np.sum(np.matmul(self.param1, self.__exponential(self.param2, x1)))
        else:
            return np.sum(np.matmul(self.param1, self.__exponential(self.param2, x0))) - np.sum(
                np.matmul(self.param1, self.__exponential(self.param2, x1))
            )

    def __per_dcdf(self, x0: float, x1: float) -> float:
        """
        Computes cumulative probability between two points using empirical percentiles.
        
        Args:
            x0 (float): Lower bound.
            x1 (float): Upper bound.
        
        Returns:
            float: Cumulative probability.
        """
        if x0 == 0:
            return self.__per_cdf(x1)
        else:
            return self.__per_cdf(x1) - self.__per_cdf(x0)

    def __per_cdf(self, x: float) -> float:
        """
        Computes cumulative probability at a point using empirical percentiles.
        
        Args:
            x (float): Evaluation point.
        
        Returns:
            float: Cumulative probability.
        """
        idx1 = np.min(np.where(self.param2 >= x))
        if idx1 > 0:
            idx0 = idx1 - 1
            return self.param1[idx0] + (self.param1[idx1] - self.param1[idx0]) * (
                (x - self.param2[idx0]) / (self.param2[idx1] - self.param2[idx0])
            )
        else:
            return self.param1[idx1] * (x / self.param2[idx1])

    def __normtruncquantfun(self, mu: float, sigma: float) -> float:
        """
        Computes numerical quantile for a truncated normal distribution.
        
        Args:
            mu (float): Mean of the normal distribution.
            sigma (float): Standard deviation.
        
        Returns:
            float: Quantile value.
        """
        cmp = 1 - norm.cdf(-mu / sigma)
        x = np.max(np.array([sigma * self.tolerance, mu]))
        trc = (norm.cdf((x - mu) / sigma) - norm.cdf(-mu / sigma)) / cmp
        dd = sigma * self.tolerance
        iter = 0
        while np.abs(trc - self.truncation) > self.tolerance and iter < self.itermax:
            x1 = x + dd
            f1 = (norm.cdf((x1 - mu) / sigma) - norm.cdf(-mu / sigma)) / cmp
            grad = (f1 - trc) / dd
            x = x - (trc - self.truncation) / grad
            trc = (norm.cdf((x - mu) / sigma) - norm.cdf(-mu / sigma)) / cmp
            iter += 1
        return x
