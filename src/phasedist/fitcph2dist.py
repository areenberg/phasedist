import sys
import numpy as np
from scipy.linalg import expm
from scipy.stats import lognorm, norm, gamma, weibull_min, chi2
import matplotlib.pyplot as plt
from phasedist.dist import dist


class fitcph2dist:
    """
    Fit a continuous-time phase-type distribution to
    a distribution with a continuous density using
    the EM algorithm from p. 681 Bladt and Nielsen (2017).

    References:
        Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in Applied Probability.
        Springer. https://doi.org/10.1007/978-1-4939-7049-0
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
        tolerance: float = 1e-3,
        truncation: float = 0.99,
        steps: int = 50,
        itermax: int = 1000000000,
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
        self.tolerance = tolerance
        self.truncation = truncation
        self.disttype = None
        self.verbose = verbose
        self.dist = None
        self.steps = steps  # number of steps in the numerical integration

        # checking and fitting
        if self.__checkinputs():
            self.__makedist()  # fit the parameters
        else:
            sys.exit(1)  # terminate the program

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
        iter = 0
        self.eps = np.inf
        self.pi0 = np.copy(self.pi)
        self.phgen0 = np.copy(self.phgen)
        while iter < self.itermax and self.eps > self.tolerance:
            self.__estep()
            self.__mstep()
            self.__updateEpsilon()
            self.pi0 = np.copy(self.pi)
            self.phgen0 = np.copy(self.phgen)
            iter += 1
            if self.verbose and iter % 5 == 0:
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

        # create object for output PH distribution
        self.dist = dist(
            discrete=False, initdist=self.pi, phgen=self.phgen, seed=self.seed
        )

        return 0

    def plot(self) -> None:
        """
        Plots the fitted distribution and target distribution.
        
        Args:
            None
        
        Returns:
            None
        """
        x = np.linspace(0.0, self.y.max(), 500)
        dist_pdf = np.zeros(len(x))
        ph_pdf = np.zeros(len(x))
        for i in range(len(x)):
            if self.disttype == "lognorm":
                dist_pdf[i] = lognorm.pdf(x[i], self.param2, scale=np.exp(self.param1))
            elif self.disttype == "gamma":
                dist_pdf[i] = gamma.pdf(x[i], self.param1, scale=self.param2)
            elif self.disttype == "weibull":
                dist_pdf[i] = weibull_min.pdf(x[i], self.param1, scale=self.param2)
            elif self.disttype == "chisq":
                dist_pdf[i] = chi2.pdf(x[i], self.param1)
            elif self.disttype == "ph":
                d = dist(discrete=False, initdist=self.param1, phgen=self.param2)
                dist_pdf[i] = d.getdensity(x[i])
            ph_pdf[i] = self.getdensity(x[i])

        # make plot
        plt.figure(figsize=(10, 6))
        if not self.disttype == "norm" and not self.disttype == "per":
            plt.plot(x, dist_pdf, label="True density", color="blue")
        plt.plot(x, ph_pdf, label="Approx. density", color="red", linestyle="--")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.title("Approximation validation")
        plt.legend()
        plt.grid(True)
        plt.show()

        return None

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

        # check PH generator and exit rates in case of no random initialization
        if self.dtype == "custom" or not self.randominit:
            self.nphases = self.initphgen.shape[0]
        if not self.randominit:
            if not self.__correctphgen(
                self.initphgen, self.initexitrates
            ) or not self.__correctinitdist(self.d.getinitdist):
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
        self.initdist = np.matrix(np.ones((1, self.nphases)))
        self.initphgen = np.matrix(np.ones((self.nphases, self.nphases)))
        self.initexitrates = np.matrix(np.ones((self.nphases, 1)))

    def __generlang(self) -> None:
        """
        Initializes parameters for a generalized Erlang distribution.
        
        Args:
            None
        
        Returns:
            None
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
        Initializes parameters for a hyper-exponential distribution.
        
        Args:
            None
        
        Returns:
            None
        """
        self.initdist = np.matrix(np.ones((1, self.nphases)))

        self.initphgen = np.matrix(np.zeros((self.nphases, self.nphases)))
        self.initexitrates = np.matrix(np.ones((self.nphases, 1)))
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
        Initializes parameters for a generalized Coxian distribution.
        
        Args:
            None
        
        Returns:
            None
        """
        self.initdist = np.matrix(np.ones((1, self.nphases)))
        self.initexitrates = np.matrix(np.ones((self.nphases, 1)))

        self.initphgen = np.matrix(np.zeros((self.nphases, self.nphases)))
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

        # copy to output parameters
        self.pi = self.initdist
        self.phgen = self.initphgen
        self.exitrates = self.initexitrates

        # initialize with random parameters
        # accounting for the specified structure
        if self.randominit:
            self.__initrandom()

        # create the cumulated probabilities and evaluation points
        self.__computeyvector()

    def __initrandom(self) -> None:
        """
        Randomly initializes parameters of a continuous-time PH distribution.
        
        Args:
            None
        
        Returns:
            None
        """                
        
        # make a random initial distribution (pi)
        nzidx = np.nonzero(self.pi)[1]
        u = np.random.uniform(low=0.0, high=1.0, size=len(nzidx))
        u = u / np.sum(u)
        self.pi[0, nzidx] = u

        # make a random exit vector
        nzidx = np.nonzero(self.exitrates)[0]
        u = np.random.uniform(low=0.0, high=10.0, size=len(nzidx))
        self.exitrates[nzidx, 0] = u

        # make a random PH generator
        for i in range(self.nphases):
            nzidx = np.nonzero(self.phgen[i, :])[1]
            msk = nzidx != i
            nzidx = nzidx[msk]
            u = np.random.uniform(low=0.0, high=10.0, size=len(nzidx))
            self.phgen[i, nzidx] = u
            self.phgen[i, i] = -(np.sum(u) + self.exitrates[i, 0])

    def __estep(self) -> None:
        """
        Performs the expectation step of the EM algorithm.
        
        Args:
            None
        
        Returns:
            None
        """
        self.bi = np.zeros(self.nphases)
        self.zi = np.zeros(self.nphases)
        self.ni = np.zeros(self.nphases)
        self.nij = np.zeros((self.nphases, self.nphases))

        # pre-compute eTy and pieTyt
        self.eTy = [None] * self.steps
        self.eTyut = [[None] * (self.steps + 1) for _ in range(self.steps)]
        self.pieTu = [[None] * (self.steps + 1) for _ in range(self.steps)]
        self.pieTyt = np.zeros(self.steps)
        for k in range(self.steps):
            self.eTy[k] = expm(self.phgen * self.y[k])
            self.pieTyt[k] = np.matmul(self.pi, np.matmul(self.eTy[k], self.exitrates))

        for i in range(self.nphases):

            # bi
            for k in range(self.steps):
                eTyt = np.matmul(self.eTy[k], self.exitrates)
                Gy = (self.pi[0, i] * eTyt[i, 0]) / self.pieTyt[k]
                self.bi[i] += Gy * self.hy[k]

            # zi (denominator used in calculation of PH generator and exit rates)
            for k in range(self.steps):
                # inner integral
                innerint = 0.0
                if self.y[k] > 0:
                    u = np.linspace(0, self.y[k], self.steps + 1)
                    for l in range(0, len(u) - 2, 2):
                        # inner fa
                        self.pieTu[k][l] = np.matmul(self.pi, expm(self.phgen * u[l]))
                        self.eTyut[k][l] = np.matmul(
                            expm(self.phgen * (self.y[k] - u[l])), self.exitrates
                        )
                        inner_fa = self.pieTu[k][l][0, i] * self.eTyut[k][l][i, 0]
                        # inner fmid
                        self.pieTu[k][l + 1] = np.matmul(
                            self.pi, expm(self.phgen * u[l + 1])
                        )
                        self.eTyut[k][l + 1] = np.matmul(
                            expm(self.phgen * (self.y[k] - u[l + 1])), self.exitrates
                        )
                        inner_fmid = (
                            self.pieTu[k][l + 1][0, i] * self.eTyut[k][l + 1][i, 0]
                        )
                        # inner fb
                        self.pieTu[k][l + 2] = np.matmul(
                            self.pi, expm(self.phgen * u[l + 2])
                        )
                        self.eTyut[k][l + 2] = np.matmul(
                            expm(self.phgen * (self.y[k] - u[l + 2])), self.exitrates
                        )
                        inner_fb = (
                            self.pieTu[k][l + 2][0, i] * self.eTyut[k][l + 2][i, 0]
                        )

                        innerint += self.__simpsonsrule(
                            u[l], u[l + 2], inner_fa, inner_fmid, inner_fb
                        )
                # outer integral
                Gy = innerint / self.pieTyt[k]
                self.zi[i] += Gy * self.hy[k]

            # ni (numerator used in calculation of exit rates)
            for k in range(self.steps):
                pieTy = np.matmul(self.pi, self.eTy[k])
                Gy = (pieTy[0, i] / self.pieTyt[k]) * self.exitrates[i, 0]
                self.ni[i] += Gy * self.hy[k]

            for j in range(self.nphases):
                # nij (numerator used in calculation of PH generator)
                if j != i and self.phgen[i, j] > 0.0:
                    for k in range(self.steps):
                        # inner integral
                        innerint = 0.0
                        if self.y[k] > 0:
                            u = np.linspace(0, self.y[k], self.steps + 1)
                            for l in range(0, len(u) - 2, 2):
                                # inner fa
                                inner_fa = (
                                    self.pieTu[k][l][0, i] * self.eTyut[k][l][j, 0]
                                )
                                # inner fmid
                                inner_fmid = (
                                    self.pieTu[k][l + 1][0, i]
                                    * self.eTyut[k][l + 1][j, 0]
                                )
                                # inner fb
                                inner_fb = (
                                    self.pieTu[k][l + 2][0, i]
                                    * self.eTyut[k][l + 2][j, 0]
                                )

                                innerint += self.__simpsonsrule(
                                    u[l], u[l + 2], inner_fa, inner_fmid, inner_fb
                                )
                        # outer integral
                        Gy = (self.phgen[i, j] / self.pieTyt[k]) * innerint
                        self.nij[i, j] += Gy * self.hy[k]

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
            self.pi[0, i] = self.bi[i]
        self.pi = self.pi / np.sum(self.pi)

        # update the exit rates
        for i in range(self.nphases):
            self.exitrates[i, 0] = self.ni[i] / self.zi[i]

        # update the PH generator
        for i in range(self.nphases):
            sm = self.exitrates[i, 0]
            for j in range(self.nphases):
                if j != i:
                    self.phgen[i, j] = self.nij[i, j] / self.zi[i]
                    sm += self.phgen[i, j]
            self.phgen[i, i] = -sm

    def __updateEpsilon(self):
        """
        Computes the maximum absolute change in parameters.
        
        Args:
            None
        
        Returns:
            None
        """        
        # eps1 = np.max(np.abs(np.divide(np.subtract(self.pi[np.nonzero(self.pi0)],self.pi0[np.nonzero(self.pi0)]),self.pi0[np.nonzero(self.pi0)])))
        # eps2 = np.max(np.abs(np.divide(np.subtract(self.phgen[np.nonzero(self.phgen0)],self.phgen0[np.nonzero(self.phgen0)]),self.phgen0[np.nonzero(self.phgen0)])))
        eps1 = np.max(
            np.abs(
                np.subtract(
                    self.pi[np.nonzero(self.pi0)], self.pi0[np.nonzero(self.pi0)]
                )
            )
        )
        eps2 = np.max(
            np.abs(
                np.subtract(
                    self.phgen[np.nonzero(self.phgen0)],
                    self.phgen0[np.nonzero(self.phgen0)],
                )
            )
        )
        self.eps = np.max(np.array([eps1, eps2]))

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
            self.y = np.linspace(
                0,
                lognorm.ppf(self.truncation, self.param2, scale=np.exp(self.param1)),
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

        # adjust y's for the E-step
        self.y = self.y[1:]

    def __simpsonsrule(self, a: float, b: float, fa: float, fmid: float, fb: float) -> float:
        """
        Computes Simpson’s 1/3 rule approximation.
        
        Args:
            a (float): Lower bound.
            b (float): Upper bound.
            fa (float): Function value at a.
            fmid (float): Function value at midpoint.
            fb (float): Function value at b.
        
        Returns:
            float: Approximated integral value.
        """
        # note: fmid = f((a+b)/2)
        return ((b - a) / 6) * (fa + 4 * fmid + fb)

    def lognormdensity(self, x: float) -> float:
        """
        Probability density function of the log-normal distribution.
        
        Args:
            x (float): Compute the density at x.
        
        Returns:
            float: Density.
        """        
        return (1 / (x * np.sqrt(self.param2) * np.sqrt(2 * np.pi))) * np.exp(
            -(np.power(np.log(x) - self.param1, 2) / (2 * self.param2))
        )

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
            return 1 - np.sum(np.matmul(self.param1, expm(self.param2 * x1)))
        else:
            return np.sum(np.matmul(self.param1, expm(self.param2 * x0))) - np.sum(
                np.matmul(self.param1, expm(self.param2 * x1))
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
