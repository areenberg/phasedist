import numpy as np
from scipy.linalg import expm


class edph:
    """
    Performs the E-step of the EM algorithm for a Discrete-Time Phase Type (DPH) distribution 
    from p. 675 Bladt and Nielsen (2017). Note: This class has no input checks.

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
            nphases (int): Number of phases in the DPH distribution. 
        """
        
        self.nphases=nphases

        return None

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
        pi_mat = self.initdist.flatten()

        for k in range(y - 1):
            self.Kmat += np.outer(exit_prod, pi_mat)

            if k < y - 2:
                Ty = np.matmul(Ty, self.phgeninv)
                exit_prod = np.matmul(Ty, self.exitrates).flatten()
                pi_mat = np.matmul(pi_mat, self.phgen).flatten()        

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
            ndarray: Number of processes that exited to the absorbing state from state i (n_i).
            ndarray: Number of processes that jumped from state i to state j (n_ij).
        """

        self.initdist = initdist
        self.phgen = phgen 
        self.exitrates = exitrates        
        self.bi = np.zeros(self.nphases)
        self.ni = np.zeros(self.nphases)
        self.nij = np.zeros((self.nphases, self.nphases))
        self.phgeninv = np.linalg.inv(self.phgen)

        y0 = -1
        for y in obs:
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
                piTpow = np.ravel(np.matmul(self.initdist, Tpow))

                for i in range(self.nphases):
                    self.bi[i] += self.pi[i] * Ttprod[i] / piTtprod
                    self.ni[i] += piTpow[i] * self.exitrates[i] / piTtprod

                    if y >= 2:
                        for j in range(self.nphases):
                            self.nij[i, j] += (self.phgen[i, j] * self.Kmat[j, i]) / piTtprod

        return self.bi,self.ni,self.nij