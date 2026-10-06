import numpy as np
from phasedist.unif import _unif

class ecph:
    """
    Performs the E-step of the EM algorithm for a Continuous-Time Phase Type (CPH) distribution
    from p. 678 Bladt and Nielsen (2017).

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

        Raises:
            ValueError: If the number of phases is not a positive integer.
        """

        if not isinstance(nphases, (int, np.integer)) or nphases < 1:
            raise ValueError("The number of phases must be an integer larger than zero.")

        self.nphases=nphases
        self.bi = np.zeros(self.nphases)
        self.zi = np.zeros(self.nphases)
        self.ni = np.zeros(self.nphases)
        self.nij = np.zeros((self.nphases, self.nphases))
        self.loglikelihood = 0.0

        # the data last seen by the input checks
        self.checkedobs = None
        self.checkedcensoring = None

        # the matrix exponentials below are evaluated by uniformization
        self.unif = _unif(tolerance=1e-14)

        return None

    def __blockJ(self) -> np.array:
        """
        Returns the block matrix behind the J-matrix. Exponentiated over a time
        y, the block matrix [[T, t*pi],[0, T]] has top-left block exp(Ty) and
        top-right block J(y), by Theorem A.2.1 (Van Loan).

        Args:
            None

        Returns:
            np.array: The block matrix, of twice the number of phases.
        """

        t = self.exitrates[:, None]
        pi = self.initdist[None, :]

        return np.block([
            [self.phgen, np.matmul(t, pi)],
            [np.zeros((self.nphases, self.nphases)), self.phgen],
        ])

    def __blockM(self) -> np.array:
        """
        Returns the block matrix behind the M-matrix, where
        M(y) = int_0^y exp(Tu) du. Exponentiated over a time y, the block matrix
        [[T, I],[0, 0]] has top-left block exp(Ty) and top-right block M(y).
        Note the zero block bottom right, where the discrete analogue carries
        the identity instead.

        Args:
            None

        Returns:
            np.array: The block matrix, of twice the number of phases.
        """

        return np.block([
            [self.phgen, np.eye(self.nphases)],
            [np.zeros((self.nphases, self.nphases)), np.zeros((self.nphases, self.nphases))],
        ])

    def __blockK(self) -> np.array:
        """
        Returns the block matrix behind the K-matrix, where
        K(y) = int_0^y exp(T(y-u)) e pi exp(Tu) du, with e the column vector of
        ones (i.e. the same construction as __blockJ, but with e in place of
        the exit-rate vector t).

        Args:
            None

        Returns:
            np.array: The block matrix, of twice the number of phases.
        """

        e = np.ones((self.nphases, 1))
        pi = self.initdist[None, :]

        return np.block([
            [self.phgen, np.matmul(e, pi)],
            [np.zeros((self.nphases, self.nphases)), self.phgen],
        ])

    def __uncensored(self, values: np.array, counts: np.array) -> None:
        """
        Updates b_i, z_i, n_i, n_ij and the log-likelihood with the contribution
        from the fully observed (uncensored) observations. Each distinct value
        is computed once and weighted by how often it occurs.

        Args:
            values (ndarray): Distinct observation values.
            counts (ndarray): How often each of those values occurs.

        Returns:
            None
        """

        n = self.nphases
        block = self.__blockJ()

        # every distinct value in one call: the powers of the uniformized matrix
        # do not depend on the value, so they are built once and shared
        mats = self.unif.run(block, values, nrows=n)

        for index, count in enumerate(counts):

            eTy = mats[index][:, :n]
            Jmat = mats[index][:, n:]

            eTyt = np.matmul(eTy, self.exitrates)
            pieTy = np.matmul(self.initdist, eTy)
            den = np.matmul(pieTy, self.exitrates)
            self.pieTyt = den

            self.bi += count * (self.initdist * eTyt) / den
            self.zi += count * np.diag(Jmat) / den
            self.ni += count * (pieTy * self.exitrates) / den
            # element i,j of the update is phgen[i,j] * Jmat[j,i], i.e. the
            # generator against the TRANSPOSE of the J-matrix. The generator is
            # taken without its diagonal, since the sum runs over j != i only.
            self.nij += count * (self.offdiagonal * Jmat.T) / den
            self.loglikelihood += count * np.log(den)

    def __rightcensored(self, limits: np.array, counts: np.array) -> None:
        """
        Updates b_i, z_i, n_i, n_ij and the log-likelihood with the contribution
        from the right-censored observations, known only to satisfy Y > limit.

        Args:
            limits (ndarray): Distinct censoring limits.
            counts (ndarray): How often each of those limits occurs.

        Returns:
            None
        """

        n = self.nphases
        block = self.__blockK()

        mats = self.unif.run(block, limits, nrows=n)

        for index, count in enumerate(counts):

            eTs = mats[index][:, :n]
            Ks = mats[index][:, n:]

            pieTs = np.matmul(self.initdist, eTs)
            den = np.sum(pieTs)  # P(Y > limit)
            self.pieTyt = den

            self.bi += count * (self.initdist * np.sum(eTs, axis=1)) / den
            self.zi += count * np.diag(Ks) / den
            self.nij += count * (self.offdiagonal * Ks.T) / den
            # N_i(limit) = 0 identically when Y > limit, so ni is untouched
            self.loglikelihood += count * np.log(den)

    def __leftcensored(self, limits: np.array, counts: np.array) -> None:
        """
        Updates b_i, z_i, n_i, n_ij and the log-likelihood with the contribution
        from the left-censored observations, known only to satisfy Y <= limit.

        Equivalent to __intervalcensored with lower limit zero.

        Args:
            limits (ndarray): Distinct censoring limits.
            counts (ndarray): How often each of those limits occurs.

        Returns:
            None
        """

        n = self.nphases
        blockM = self.__blockM()
        blockK = self.__blockK()

        matsM = self.unif.run(blockM, limits, nrows=n)
        matsK = self.unif.run(blockK, limits, nrows=n)

        for index, count in enumerate(counts):

            eTt = matsM[index][:, :n]
            Mt = matsM[index][:, n:]

            Kt = matsK[index][:, n:]

            pieTt = np.matmul(self.initdist, eTt)
            den = 1.0 - np.sum(pieTt)  # P(Y <= limit)
            self.pieTyt = den

            piM = np.matmul(self.initdist, Mt)

            self.bi += count * (self.initdist
                                * (1.0 - np.sum(eTt, axis=1))) / den
            self.zi += count * (piM - np.diag(Kt)) / den
            self.nij += count * (self.offdiagonal
                                 * (piM[:, None] - Kt.T)) / den
            self.ni += count * (self.exitrates * piM) / den
            self.loglikelihood += count * np.log(den)

    def __intervalcensored(
            self,
            lefts: np.array,
            rights: np.array,
            counts: np.array
        ) -> None:
        """
        Updates b_i, z_i, n_i, n_ij and the log-likelihood with the contribution
        from the interval-censored observations, known only to lie in
        (left, right].

        Both ends of an interval are needed at once, so unlike the other three
        kinds this one first works out the matrices at every distinct endpoint
        and then visits the distinct intervals.

        Args:
            lefts (ndarray): Lower limits of the distinct intervals.
            rights (ndarray): Upper limits of the distinct intervals.
            counts (ndarray): How often each of those intervals occurs.

        Returns:
            None
        """

        n = self.nphases
        blockM = self.__blockM()
        blockK = self.__blockK()

        endpoints = np.unique(np.concatenate([lefts, rights]))

        # the exponential of each block matrix at each distinct endpoint, so
        # that an endpoint shared by several intervals is computed once
        eTy = np.empty((endpoints.size, n, n))
        Mmat = np.empty((endpoints.size, n, n))
        Kmat = np.empty((endpoints.size, n, n))

        matsM = self.unif.run(blockM, endpoints, nrows=n)
        matsK = self.unif.run(blockK, endpoints, nrows=n)

        for k in range(endpoints.size):
            eTy[k] = matsM[k][:, :n]
            Mmat[k] = matsM[k][:, n:]
            Kmat[k] = matsK[k][:, n:]

        # where each interval's ends sit among the distinct endpoints
        leftindex = np.searchsorted(endpoints, lefts)
        rightindex = np.searchsorted(endpoints, rights)

        for count, low, high in zip(counts, leftindex, rightindex):

            eTs, eTt = eTy[low], eTy[high]

            pieTs = np.matmul(self.initdist, eTs)
            pieTt = np.matmul(self.initdist, eTt)
            den = np.sum(pieTs) - np.sum(pieTt)  # P(left < Y <= right)
            self.pieTyt = den

            piM = np.matmul(self.initdist, Mmat[high] - Mmat[low])
            KtmKs = Kmat[high] - Kmat[low]

            self.bi += count * (self.initdist
                                * (np.sum(eTs, axis=1)
                                   - np.sum(eTt, axis=1))) / den
            self.zi += count * (piM - np.diag(KtmKs)) / den
            self.nij += count * (self.offdiagonal
                                 * (piM[:, None] - KtmKs.T)) / den
            self.ni += count * (self.exitrates * piM) / den
            self.loglikelihood += count * np.log(den)

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

        self.initdist = np.asarray(initdist).reshape(-1)
        self.phgen = np.asarray(phgen)
        self.exitrates = np.asarray(exitrates).reshape(-1)

        # the generator without its diagonal, since the n_ij sums run over
        # j != i only; the diagonal of the generator is not a transition
        self.offdiagonal = np.array(self.phgen, dtype=float)
        np.fill_diagonal(self.offdiagonal, 0.0)

    def __checkinputs(
            self,
            obs: np.array,
            initdist: np.array,
            phgen: np.array,
            exitrates: np.array,
            censoring: np.array
        ) -> None:
        """
        Checks the inputs of run(). Every comparison is written so that a NaN
        fails it. The checks over the whole sample are made once per array,
        recognized by identity, since only the parameters change between
        iterations.

        Args:
            obs (ndarray): Array of observations.
            initdist (ndarray): Initial distribution vector.
            phgen (ndarray): Phase-type generator.
            exitrates (ndarray): Exit-rate vector.
            censoring (ndarray): Specifies censored observations, or None.

        Raises:
            ValueError: If an input is infeasible.
        """

        if (initdist.size != self.nphases or exitrates.size != self.nphases
                or phgen.shape != (self.nphases, self.nphases)):
            raise ValueError("The initial distribution, the generator and the exit rates must all be of dimension %d." % self.nphases)

        if not (initdist.min() >= 0.0 and initdist.sum() <= 1.0 + 1e-9):
            raise ValueError("The initial distribution must be non-negative and sum to at most one.")

        if not (exitrates.min() >= 0.0 and exitrates.max() < np.inf):
            raise ValueError("The exit rates must be non-negative and finite.")

        # the row sums are formed by cancellation, so they are judged
        # against the largest rate
        diagonal = np.diag(phgen)
        offdiagonal = phgen - np.diag(diagonal)

        if not (diagonal.max() < 0.0 and offdiagonal.min() >= 0.0
                and phgen.sum(axis=1).max() <= 1e-9 * (1.0 - diagonal.min())):
            raise ValueError("The phase-type generator is not a sub-intensity matrix.")

        if obs is self.checkedobs and censoring is self.checkedcensoring:
            return None

        if obs.ndim != 1 or obs.size == 0:
            raise ValueError("The observations must be a non-empty one-dimensional array.")

        if not (obs.min() > 0.0 and obs.max() < np.inf):
            raise ValueError("The observations must be positive and finite.")

        if censoring is not None:
            if (censoring.ndim != 2 or censoring.shape[1] != 2
                    or censoring.shape[0] < obs.size):
                raise ValueError("The censoring array must have two columns and at least one row per observation.")

            # zero is admissible: it is the lower limit of a left-censored
            # interval
            known = censoring[np.isfinite(censoring)]

            if known.size and not known.min() >= 0.0:
                raise ValueError("The censoring limits must be non-negative.")

        self.checkedobs = obs
        self.checkedcensoring = censoring

        return None

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
        - [np.nan,float] -> right-censored (larger or equal to t).
        - [float,np.nan] -> left-censored (less or equal to s).
        - [float,float] -> interval-censored (from s to t).

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

        Raises:
            ValueError: If an input is infeasible.
        """

        obs = np.asarray(obs, dtype=float)
        initdist = np.asarray(initdist, dtype=float).reshape(-1)
        phgen = np.asarray(phgen, dtype=float)
        exitrates = np.asarray(exitrates, dtype=float).reshape(-1)

        if censoring is not None:
            censoring = np.asarray(censoring, dtype=float)

        self.__checkinputs(obs, initdist, phgen, exitrates, censoring)

        self.__storefundamental(initdist,phgen,exitrates)

        self.bi.fill(0)
        self.zi.fill(0)
        self.ni.fill(0)
        self.nij.fill(0)

        self.loglikelihood = 0.0

        if censoring is None:
            values, counts = np.unique(obs, return_counts=True)
            if values.size:
                self.__uncensored(values, counts)
            return self.bi, self.zi, self.ni, self.nij

        censoring = censoring[:obs.size]

        # the four kinds, read off the censoring array in one go rather than
        # one observation at a time
        lowermissing = np.isnan(censoring[:, 0])
        uppermissing = np.isnan(censoring[:, 1])

        isuncensored = lowermissing & uppermissing
        isright = lowermissing & ~uppermissing
        isleft = ~lowermissing & uppermissing
        isinterval = ~lowermissing & ~uppermissing

        if np.any(isuncensored):
            values, counts = np.unique(obs[isuncensored], return_counts=True)
            self.__uncensored(values, counts)

        if np.any(isright):
            limits, counts = np.unique(censoring[isright, 1],
                                       return_counts=True)
            self.__rightcensored(limits, counts)

        if np.any(isleft):
            limits, counts = np.unique(censoring[isleft, 0],
                                       return_counts=True)
            self.__leftcensored(limits, counts)

        if np.any(isinterval):
            intervals, counts = np.unique(censoring[isinterval, :], axis=0,
                                          return_counts=True)
            self.__intervalcensored(intervals[:, 0], intervals[:, 1], counts)

        return self.bi, self.zi, self.ni, self.nij
