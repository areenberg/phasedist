import numpy as np


class mcph:
    """
    Performs the M-step of the EM algorithm for a Continuous-Time Phase Type (CPH) distribution
    from p. 678 Bladt and Nielsen (2017).

    References:
        Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in Applied Probability.
        Springer. https://doi.org/10.1007/978-1-4939-7049-0
    """

    def __init__(
        self,
        nphases: int,
        nobs: int
    ) -> None:
        """
        Initializes the M-step class.

        Args:
            nphases (int): Number of phases in the CPH distribution.
            nobs (int): Number of observations.

        Raises:
            ValueError: If either argument is not a positive integer.
        """
        if not isinstance(nphases, (int, np.integer)) or nphases < 1:
            raise ValueError("The number of phases must be an integer larger than zero.")

        # nobs divides the b_i, so it cannot be zero
        if not isinstance(nobs, (int, np.integer)) or nobs < 1:
            raise ValueError("The number of observations must be an integer larger than zero.")

        self.nphases = nphases
        self.nobs = nobs

        return None

    def run(self,
            bi: np.array,
            zi: np.array,
            ni: np.array,
            nij: np.array
        ) -> None:    
        """
        Performs the calculations of the M-step.

        Args:
            bi (ndarray): Number of processes initiating in state i.
            zi (ndarray): Total time spent in state i.
            ni (ndarray): Number of processes exiting to absorbing state from state i.
            nij (ndarray): Number of processes jumping from state i to state j.

        Returns:
            ndarray: Initial distribution vector.
            ndarray: Phase-type generator.
            ndarray: Exit-rate vector.

        Raises:
            ValueError: If a statistic is infeasible.
        """

        self.bi = bi
        self.zi = zi
        self.ni = ni
        self.nij = nij

        # the arguments are sufficient statistics rather than parameters, so
        # only the shapes and the divisor z_i are checked
        if (np.shape(bi) != (self.nphases,) or np.shape(zi) != (self.nphases,)
                or np.shape(ni) != (self.nphases,)
                or np.shape(nij) != (self.nphases, self.nphases)):
            raise ValueError("The sufficient statistics must be of dimension %d." % self.nphases)

        if not np.min(zi) > 0.0:
            raise ValueError("Every phase must hold a positive total time z_i.")

        self.initdist = self.bi / self.nobs

        self.exitrates = self.ni / self.zi

        self.phgen = np.zeros((self.nphases, self.nphases))

        for i in range(self.nphases):
            for j in range(self.nphases):
                if j != i:
                    self.phgen[i, j] = self.nij[i, j] / self.zi[i]
            off_diag_sum = np.sum([self.phgen[i, j] for j in range(self.nphases) if j != i])
            self.phgen[i, i] = -(off_diag_sum + self.exitrates[i])

        return self.initdist,self.phgen,self.exitrates

    