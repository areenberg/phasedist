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
        self.bi = np.zeros(self.nphases)
        self.zi = np.zeros(self.nphases)
        self.ni = np.zeros(self.nphases)
        self.nij = np.zeros((self.nphases, self.nphases))

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

    def __uncensored(self,y):
        """
        Updates b_i, z_i, n_i, n_ij with the contribution
        from a single fully observed (uncensored) observation y.

        Args:
            y (float): Observation value.

        Returns:
            None
        """

        self.__Jmatrix(y)

        eTyt = np.matmul(self.eTy, self.exitrates)
        pieTy = np.matmul(self.initdist, self.eTy)
        self.pieTyt = np.matmul(pieTy, self.exitrates)

        for i in range(self.nphases):
            self.bi[i] += (self.initdist[i] * eTyt[i]) / self.pieTyt
            self.zi[i] += self.Jmat[i, i] / self.pieTyt
            for j in range(self.nphases):
                if j != i:
                    self.nij[i, j] += (self.phgen[i, j] * self.Jmat[j, i]) / self.pieTyt
            self.ni[i] += (pieTy[i] * self.exitrates[i]) / self.pieTyt

    def __rightcensored(self,right):
        # CODE HERE
        return 0

    def __leftcensored(self,left):
        # CODE HERE
        return 0

    def __intervalcensored(self,left,right):
        # CODE HERE
        return 0

    def __storefundamental(
                            self,
                            initdist: np.array,
                            phgen: np.array,
                            exitrates: np.array
                        ):
        """
        Stores the CPH distribution's fundamental parameters as instance attributes so they
        can be accessed by the other methods during the E-step calculations.

        Args:
            initdist (ndarray): Initial distribution vector.
            phgen (ndarray): Phase-type generator.
            exitrates (ndarray): Exit-rate vector.

        Returns:
            None
        """

        self.initdist = initdist
        self.phgen = phgen 
        self.exitrates = exitrates

    def run(
            self,
            obs: np.array,
            initdist: np.array,
            phgen: np.array,
            exitrates: np.array,
            censoring: np.array = None
        ):
        """
        Performs the calculations of the E-step.

        Censored observations can be specified using the (n_obs x 2) censoring ndarray where each
        row corresponds to an observation and the two columns determines if an observation is censored:
        - [np.nan,np.nan] -> uncensored.
        - [np.nan,float] -> right-censored.
        - [float,np.nan] -> left-censored.
        - [float,float] -> interval-censored.

        Args:
            obs (ndarray): Array of observations.
            initdist (ndarray): Initial distribution vector.
            phgen (ndarray): Phase-type generator.
            exitrates (ndarray): Exit-rate vector.
            censoring (ndarray): Specifies censored observations.

        Returns:
            ndarray: Number of processes that initiated in state i (b_i).
            ndarray: Total time spent in state i (z_i).
            ndarray: Number of processes that exited to the absorbing state from state i (n_i).
            ndarray: Number of processes that jumped from state i to state j (n_ij).
        """

        self.__storefundamental(initdist,phgen,exitrates)

        self.bi.fill(0)
        self.zi.fill(0)
        self.ni.fill(0)
        self.nij.fill(0)

        self.loglikelihood = 0.0

        if censoring is not None:
            for idx,y in enumerate(obs):

                if np.isnan(censoring[idx,0]) and np.isnan(censoring[idx,1]):
                    self.__uncensored(y) #uncensored observation
                elif np.isnan(censoring[idx,0]) and not np.isnan(censoring[idx,1]):
                    self.__rightcensored(censoring[idx,1]) #Right-censored observation
                elif not np.isnan(censoring[idx,0]) and np.isnan(censoring[idx,1]):
                    self.__leftcensored(censoring[idx,0]) #Left-censored observation
                elif not np.isnan(censoring[idx,0]) and not np.isnan(censoring[idx,1]):
                    self.__intervalcensored(censoring[idx,0],censoring[idx,1]) #Interval-censored observation

                self.loglikelihood += np.log(self.pieTyt) #<<<--- check if needs to be dependent on censoring type    

        else:
            for y in obs:
                self.__uncensored(y)
                self.loglikelihood += np.log(self.pieTyt)

        return self.bi,self.zi,self.ni,self.nij