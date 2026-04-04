import numpy as np
from scipy.linalg import expm


class ecph:
    """
    Performs the E-step of the EM algorithm for a Continuous-Time Phase Type (CPH) distribution 
    from p. 678 Bladt and Nielsen (2017). Note: This class has no input checks.

    References:
        Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in Applied Probability.
        Springer. https://doi.org/10.1007/978-1-4939-7049-0
    """

    def __init__(
        self,
        nphases: int
    ) -> None:
        """
        Initializes the E-step class.

        Args:
            nphases (int): Number of phases in the CPH distribution. 
        """
        
        self.nphases=nphases

        return None

    def __Jmatrix(self, y: float) -> None:
        """
        Computes the J-matrix and matrix exponential exp(Ty).

        Args:
            y (float): Observation value.

        Returns:
            None
        """
        t = self.exitrates[:, None]
        pi = self.initdist[None, :]

        mat = expm(
            np.block([
                [self.phgen, np.matmul(t,pi)],
                [np.zeros((self.nphases, self.nphases)), self.phgen],
            ]) * y
        )

        self.eTy = mat[: self.nphases, : self.nphases]
        self.Jmat = mat[: self.nphases, self.nphases : 2 * self.nphases]

    def run(
            self,
            obs: np.array,
            initdist: np.array,
            phgen: np.array,
            exitrates: np.array
        ) -> np.array | np.array | np.array | np.array:
        """
        Performs the calculations of the E-step.

        Args:
            obs (ndarray): Array of observations.
            initdist (ndarray): Initial distribution vector.
            phgen (ndarray): Phase-type generator.
            exitrates (ndarray): Exit-rate vector.

        Returns:
            ndarray: Number of processes that initiated in state i (b_i).
            ndarray: Total time spent in state i (z_i).
            ndarray: Number of processes that exited to the absorbing state from state i (n_i).
            ndarray: Number of processes that jumped from state i to state j (n_ij).
        """
        self.initdist = initdist
        self.phgen = phgen 
        self.exitrates = exitrates
        self.bi = np.zeros(self.nphases)
        self.zi = np.zeros(self.nphases)
        self.ni = np.zeros(self.nphases)
        self.nij = np.zeros((self.nphases, self.nphases))
        self.loglikelihood = 0.0

        for y in obs:
            self.__Jmatrix(y)
            eTyt = np.matmul(self.eTy, self.exitrates)
            pieTy = np.matmul(self.initdist, self.eTy)
            pieTyt = np.matmul(pieTy, self.exitrates)
            self.loglikelihood += np.log(pieTyt)
            for i in range(self.nphases):
                self.bi[i] += (self.initdist[i] * eTyt[i]) / pieTyt
                self.zi[i] += self.Jmat[i, i] / pieTyt
                for j in range(self.nphases):
                    if j != i:
                        self.nij[i, j] += (self.phgen[i, j] * self.Jmat[j, i]) / pieTyt
                self.ni[i] += (pieTy[i] * self.exitrates[i]) / pieTyt


        return self.bi,self.zi,self.ni,self.nij

    def getLogLikelihood(self) -> float:
        """
        Returns the log-likelihood computed during the E-step.

        Args:
            None.

        Returns:
            float: The log-likelihood.
        """
        return self.loglikelihood