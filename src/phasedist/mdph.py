import numpy as np


class mdph:
    """
    Performs the M-step of the EM algorithm for a Discrete-Time Phase Type (DPH) distribution
    from p. 675 Bladt and Nielsen (2017).

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
            nphases (int): Number of phases in the DPH distribution.
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
            ni: np.array,
            nij: np.array
        ) -> None:    
        """
        Performs the calculations of the M-step.

        Args:
            bi (ndarray): Number of processes initiating in state i.
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
        self.ni = ni
        self.nij = nij

        # the arguments are sufficient statistics rather than parameters, so
        # only the shapes and the divisor n_i + sum_k n_ik are checked
        if (np.shape(bi) != (self.nphases,) or np.shape(ni) != (self.nphases,)
                or np.shape(nij) != (self.nphases, self.nphases)):
            raise ValueError("The sufficient statistics must be of dimension %d." % self.nphases)

        if not np.min(np.add(ni, np.sum(nij, axis=1))) > 0.0:
            raise ValueError("Every phase must record a positive number of transitions out of it.")

        self.initdist = self.bi / self.nobs

        self.exitrates = np.zeros(self.nphases)
        
        for i in range(self.nphases):
            sm = self.ni[i]
            for j in range(self.nphases):
                sm += self.nij[i, j]
            self.exitrates[i] = self.ni[i] / sm

        self.phgen = np.zeros((self.nphases, self.nphases))

        for i in range(self.nphases):
            for j in range(self.nphases):
                sm = self.ni[i]
                for k in range(self.nphases):
                    sm += self.nij[i, k]
                self.phgen[i, j] = self.nij[i, j] / sm

        return self.initdist,self.phgen,self.exitrates