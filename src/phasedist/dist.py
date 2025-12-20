import sys
import numpy as np
from scipy.linalg import expm
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from typing import Union


class dist:
    """
    Phase-type distribution object that sets up a discrete or continuous phase-type (PH)
    distribution based on a provided initial distribution vector and
    phase-type generator matrix. The object supports evaluation of
    density, cumulative distribution, quantiles, and random sampling.
    """

    def __init__(
        self,
        discrete: bool = False,
        initdist: np.array = None,
        phgen: np.array = None,
        seed: int = None,
    ) -> None:
        """
        Initialize a phase-type distribution object.

        Args:
            discrete (bool, default=False):
                If True, the distribution is treated as a discrete
                phase-type (DPH) distribution. If False, it is treated as a
                continuous phase-type (CPH) distribution.
            initdist (np.array):
                The initial distribution vector (row vector). Must have
                length equal to the number of phases. If None, no explicit
                initial distribution is enforced.
            phgen (np.array):
                The PH generator matrix. For CPH this must be a sub-intensity
                matrix; for DPH, a sub-transition matrix. Its dimension
                defines the number of phases.
            seed (int):
                Random seed used for reproducible sampling.

        Notes
        -----
        Input validation is performed automatically. If validation fails,
        program execution terminates.
        """

        self.discrete = discrete
        self.initdist = initdist
        self.phgen = phgen
        self.nphases = self.phgen.shape[0]
        self.seed = seed

        if self.__checkinputs():  # check inputs
            self.__initialize()
        else:
            sys.exit(1)  # terminate the program

    def getinitdist(self) -> np.array:
        """
        Returns the initial distribution.

        Returns:
            np.array: The initial distribution (vector).
        """

        return self.initdist

    def getphasegen(self) -> np.array:
        """
        Returns the phase-type generator.

        Returns:
            np.array: The phase-type generator (matrix).
        """

        return self.phgen

    def getexitrates(self) -> np.array:
        """
        Returns the exitrate vector.

        Returns:
            np.array: The exit rate vector (vector).
        """

        return self.exitrates

    def getmean(self) -> float:
        """
        Returns the mean.

        Returns:
            float: The computed mean.
        """

        if self.discrete:
            return np.sum(
                np.matmul(
                    self.initdist,
                    np.linalg.inv(np.subtract(np.eye(self.nphases), self.phgen)),
                )
            )
        else:
            return -np.sum(np.matmul(self.initdist, np.linalg.inv(self.phgen)))

    def getvar(self) -> float:
        """
        Returns the variance.

        Returns:
            float: The computed variance.
        """

        if self.discrete:
            Tinv = np.linalg.inv(np.subtract(np.eye(self.nphases), self.phgen))
            return (
                np.sum(
                    np.matmul(
                        np.matmul(self.initdist, Tinv),
                        np.subtract((2 * Tinv), np.eye(self.nphases)),
                    )
                )
                - self.getmean() ** 2
            )
        else:
            phinv = np.linalg.inv(self.phgen)
            return 2 * np.sum(
                np.matmul(self.initdist, np.linalg.matrix_power(phinv, 2))
            ) - np.power(np.sum(np.matmul(self.initdist, phinv)), 2)

    def getdensity(self, x: float) -> float:
        """
        Returns the distribution's density, f(x).

        Args:
            x (float): The distribution's density will be computed at time of x.

        Returns:
            float: The computed density.
        """

        if isinstance(x, np.ndarray):
            y = np.zeros(x.size)
            for i in range(x.size):
                y[i] = self.__computedensity(x[i])
            return y
        else:
            return self.__computedensity(x)

    def getcumprob(self, x: float) -> float:
        """
        Returns the cumulated probability, P(X<=x).

        Args:
            x (float): The cumulated probability will be computed for the interval between 0 and x, i.e. P(X<=x).

        Returns:
            float: The computed probability.
        """

        if isinstance(x, np.ndarray):
            y = np.zeros(x.size)
            for i in range(x.size):
                y[i] = self.__computecumprob(x[i])
            return y
        else:
            return self.__computecumprob(x)

    def getquantile(self, p: float, tolerance: float = 1e-9) -> int | float:
        """
        Returns the x that ensures P(X<=x)=p, i.e. the quantile function of the
        PH distribution.

        Args:
            p (float): The cumulated probability.
            tolerance (float, default=1e-9): Tolerance for the numerical algorithm.

        Returns:
            int or float: The numerically evaluated quantile.
        """

        if isinstance(p, np.ndarray):
            y = np.zeros(p.size)
            for i in range(p.size):
                y[i] = self.__computequantile(p[i], tolerance)
            return y
        else:
            return self.__computequantile(p, tolerance)

    def getrandom(self, size: int = 1) -> int | float:
        """
        Samples a pseudo-random number from the distribution.

        Args:
            size (int, default=1): The sample size.

        Returns:
            int | float: The generated random number.
        """

        if size == 1:
            if self.discrete:
                return self.__dphsample()
            else:
                return self.__cphsample()
        elif size < 1:
            return np.nan
        else:
            obs = np.zeros(size)
            for i in range(size):
                if self.discrete:
                    obs[i] = self.__dphsample()
                else:
                    obs[i] = self.__cphsample()
            return obs

    def plot(self, type: str = "pdf") -> None:
        """
        Plots the density or cumulated distribution function.

        Args:
            type (str, default="pdf"): The plotted function ("pdf" or "cdf").

        Returns:
            None
        """

        # compute densities for approximate and true distributions
        x = np.linspace(0.0, self.getquantile(p=0.999), 500)
        val = np.zeros(len(x))
        for i in range(len(x)):
            if type == "pdf":
                val[i] = self.getdensity(x[i])
            elif type == "cdf":
                val[i] = self.getcumprob(x[i])

        if type == "pdf":
            ylbl = "Density"
            tl = "Probability Density Function"
        elif type == "cdf":
            ylbl = "Probability"
            tl = "Cumulative Distribution Function"

        # make plot
        plt.figure(figsize=(10, 6))
        plt.plot(x, val, label=ylbl, color="blue", linestyle="-")
        plt.xlabel("x")
        plt.ylabel(ylbl)
        plt.title(tl)
        plt.legend()
        plt.grid(True)
        plt.show()

        return None

    def __initialize(self) -> None:
        """
        Overall initialization.

        Returns:
            None
        """

        # compute exit rate vector
        if self.discrete:
            self.exitrates = 1 - np.sum(self.phgen, axis=1)
        else:
            self.exitrates = abs(np.sum(self.phgen, axis=1))

        # set a pre-defined seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # prepare vectors for random sampling
        self.flatinitdist = np.asarray(self.initdist).ravel()
        self.a = []
        for s in range(self.nphases):
            if self.discrete:
                self.a.append(
                    np.append(
                        np.asarray(self.phgen[s, :]).ravel(), self.exitrates[s].item()
                    )
                )
            else:
                a = np.asarray(self.phgen[s, :] / (-self.phgen[s, s])).ravel()
                a[a < 0] = 0
                self.a.append(
                    np.append(a, (self.exitrates[s].item() / -self.phgen[s, s]))
                )

    def __checkinputs(self) -> bool:
        """
        Checks the feasibility of all input parameters.

        Returns:
            bool: Returns True if feasible; otherwise False.
        """

        # check data types and convert if necesarry
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
        if self.phgen is not None and (
            isinstance(self.phgen, np.ndarray) or isinstance(self.phgen, list)
        ):
            self.phgen = np.matrix(self.phgen)
        elif self.phgen is not None and not isinstance(self.phgen, np.matrix):
            print(
                "Error: The PH generator can only be specified as a list or a NumPy matrix."
            )
            return False
        if self.seed is not None and not isinstance(self.seed, int):
            print("Error: The seed can only be specified as an integer.")
            return False
        return True  # if all correct

    def __computedensity(self, x: float) -> float:
        """
        Returns the distribution's density function.

        Args:
            x (float): Computes the density function for the scalar x.

        Returns:
            float: The computed density.
        """

        if self.discrete:
            if int(x) != x:
                print("Error: 'x' is not an integer.")
                return np.nan
            else:
                return np.matmul(
                    self.initdist,
                    np.matmul(
                        np.linalg.matrix_power(self.phgen, (x - 1)), self.exitrates
                    ),
                ).item()
        else:
            return np.matmul(
                self.initdist, np.matmul(expm(self.phgen * x), self.exitrates)
            ).item()

    def __computecumprob(self, x: float) -> float:
        """
        Returns the distribution function.

        Args:
            x (float): Computes the distribution function for the scalar x.

        Returns:
            float: The computed cumulated probability.
        """

        if self.discrete:
            if int(x) != x:
                print("Error: 'x' is not an integer.")
                return np.nan
            else:
                return 1 - np.sum(
                    np.matmul(self.initdist, np.linalg.matrix_power(self.phgen, int(x)))
                )
        else:
            return 1 - np.sum(np.matmul(self.initdist, expm(self.phgen * x)))

    def __computequantile(self, p: float, tolerance: float = 1e-9) -> int | float:
        """
        The quantile function of the PH distribution.

        Args:
            p (float): The cumulated probability.
            tolerance (float, default=1e-9): The tolerance used in the numerical algorithm.

        Returns:
            int | float: The computed quantile.
        """

        if p == 1.0:
            return np.inf
        elif p < 0.0 or p > 1.0:
            return np.nan
        elif p == 0.0:
            return 0.0
        elif self.discrete:
            return self.__dphquantfun(prob=p, itermax=1000000)
        else:
            return self.__cphquantfun(prob=p, tol=tolerance, itermax=1000000)

    def __cphsample(self) -> float:
        """
        Samples a pseudo-random value from a continuous phase-type (CPH) distribution.

        Returns:
            float: The sampled value.
        """

        t = 0.0
        s = np.random.choice(self.nphases, size=1, p=self.flatinitdist)[0]
        while True:
            t += np.random.exponential(scale=1 / (-self.phgen[s, s]))
            a = self.a[s]
            s = np.random.choice((self.nphases + 1), size=1, p=a)[0]
            if s == self.nphases:
                return t

    def __dphsample(self) -> int:
        """
        Samples a pseudo-random value from a discrete phase-type (DPH) distribution.

        Returns:
            int: The sampled value.
        """

        t = 0  # Time in discrete steps
        s = np.random.choice(self.nphases, size=1, p=self.flatinitdist)[0]
        while True:
            t += 1
            a = self.a[s]
            s = np.random.choice((self.nphases + 1), size=1, p=a)[0]
            if s == self.nphases:
                return t

    def __cphquantfun(
        self, prob: float, tol: float = 1e-9, itermax: int = 1000000
    ) -> float:
        """
        Numerically evaluates the quantile function of the CPH distribution.

        Args:
            prob (float): The cumulated probability, P(X<=x).
            tol (float, default=1e-9): The tolerance for the numerical algorithm.
            itermax (int, default=1000000): The maximum number of iterations.

        Returns:
            float: The numerically evaluated quantile.
        """

        # some initial preparation
        phinv = np.linalg.inv(self.phgen)
        var = 2 * np.sum(
            np.matmul(self.initdist, np.linalg.matrix_power(phinv, 2))
        ) - np.power(np.sum(np.matmul(self.initdist, phinv)), 2)
        mean = -np.sum(np.matmul(self.initdist, phinv))

        # generate initial guess
        param1 = np.log(np.power(mean, 2) / np.sqrt(np.power(mean, 2) + var))
        param2 = np.log(1 + var / np.power(mean, 2))
        x = lognorm.ppf(prob, param2, scale=np.exp(param1))

        # improve x until convergence
        trc = 1 - np.sum(np.matmul(self.initdist, expm(self.phgen * x)))
        iter = 0
        while np.abs(trc - prob) > tol and iter < itermax:
            grad = np.matmul(
                self.initdist, np.matmul(expm(self.phgen * x), self.exitrates)
            ).item()
            x = x - (trc - prob) / grad
            trc = 1 - np.sum(np.matmul(self.initdist, expm(self.phgen * x)))
            iter += 1
        if iter == itermax:
            print(
                "Warning: Algorithm terminated with iter==itermax. Results might be misleading."
            )
        return np.round(x, int(-np.log10(tol)) - 1)

    def __dphquantfun(self, prob: float, itermax: int = 1000000) -> int:
        """
        Numerical quantile function for the DPH distribution.

        Args:
            prob (float): The cumulated probability, P(X<=x).
            itermax (int, default=1000000): The maximum number of iterations.

        Returns:
            int: The computed quantile.
        """

        # initialization
        x = int(-1)
        trc = 0.0
        iter = 0

        # improve x until convergence
        while trc < prob and iter < itermax:
            x += 1
            trc = 1 - np.sum(
                np.matmul(self.initdist, np.linalg.matrix_power(self.phgen, x))
            )
            iter += 1
        if iter == itermax:
            print(
                "Warning: Algorithm terminated with iter==itermax. Results might be misleading."
            )
        return x