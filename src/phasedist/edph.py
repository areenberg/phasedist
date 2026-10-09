import numpy as np


class edph:
    """
    Performs the E-step of the EM algorithm for a Discrete-Time Phase Type (DPH) distribution
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
            nphases (int): Number of phases in the DPH distribution.

        Raises:
            ValueError: If the number of phases is not a positive integer.
        """

        if not isinstance(nphases, (int, np.integer)) or nphases < 1:
            raise ValueError("The number of phases must be an integer larger than zero.")

        self.nphases = nphases
        self.bi = np.zeros(self.nphases)
        self.ni = np.zeros(self.nphases)
        self.nij = np.zeros((self.nphases, self.nphases))
        self.loglikelihood = 0.0

        # the data last seen by the input checks
        self.checkedobs = None
        self.checkedcensoring = None

        return None

    def __blockJ(self) -> np.array:
        """
        Returns the block matrix behind the J-matrix, where
        J(y;pi,T) = sum_{k=0}^{y-2} T^(y-2-k) t pi T^k,
        via the discrete analogue of Theorem A.2.1 (Van Loan) for matrix powers
        (Theorem A.2.2, p. 714): for the block matrix [[T, t*pi],[0, T]], its
        (y-1)-th power has top-left block T^(y-1) and top-right block J(y;pi,T).
        Used in the uncensored N_ij formula (13.14). Requires y >= 1 (for y=1
        this correctly reduces to the zero matrix, since J(1;pi,T) is an empty sum).

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
        Returns the block matrix behind the M-matrix, where M(y) = sum_{k=0}^{y-1} T^k.

        Via Theorem A.2.2 with block matrix [[T, I],[0, I]]: its y-th power has
        top-left block T^y and top-right block M(y) (since the (2,2) block must
        be the identity -- not the zero matrix as in the continuous case --
        so that its k-th power stays I for every k, giving a constant factor of
        I at each term of the sum, matching M(y) = sum T^k * I * I^k).

        Args:
            None

        Returns:
            np.array: The block matrix, of twice the number of phases.
        """

        return np.block([
            [self.phgen, np.eye(self.nphases)],
            [np.zeros((self.nphases, self.nphases)), np.eye(self.nphases)],
        ])

    def __blockK(self) -> np.array:
        """
        Returns the block matrix behind the K-matrix, where
        K(y) = sum_{k=0}^{y-1} T^(y-1-k) e pi T^k, with e the column vector of
        ones (i.e. the same construction as __blockJ, but with e in place of
        the exit-rate vector t, and summed over one more term -- k=0..y-1
        rather than k=0..y-2 -- since here every one of the y steps may be a
        transient-to-transient transition).

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

    def __powers(self, block: np.array, exponents: np.array):
        """
        Yields block raised to each of the given exponents, which must be sorted
        in increasing order.

        Each power is reached from the one before it, so a run of consecutive
        exponents costs a single matrix multiplication each instead of a fresh
        exponentiation. Where the exponents jump, the gap is crossed by binary
        exponentiation, which is what computing the power directly would have
        cost anyway, so this is never the slower way round.

        Args:
            block (ndarray): The matrix to raise to the powers.
            exponents (ndarray): Exponents, sorted in increasing order.

        Yields:
            ndarray: The block matrix raised to each exponent in turn.
        """

        current = np.eye(block.shape[0])
        reached = 0

        for exponent in exponents:
            step = int(exponent) - reached
            if step == 1:
                current = np.matmul(current, block)
            elif step > 1:
                current = np.matmul(current, np.linalg.matrix_power(block, step))
            reached = int(exponent)
            yield current

    def __uncensored(self, values: np.array, counts: np.array) -> None:
        """
        Updates b_i, n_i, n_ij and the log-likelihood with the contribution from
        the fully observed (uncensored) observations, using (13.13), (13.14),
        (13.15). Each distinct value is computed once and weighted by how often
        it occurs.

        Args:
            values (ndarray): Distinct observation values (y >= 1), increasing.
            counts (ndarray): How often each of those values occurs.

        Returns:
            None
        """

        n = self.nphases

        for count, mat in zip(counts, self.__powers(self.__blockJ(), values - 1)):

            Tpow = mat[:n, :n]
            Jmat = mat[:n, n:]

            Ttprod = np.matmul(Tpow, self.exitrates)
            piTpow = np.matmul(self.initdist, Tpow)
            den = np.matmul(piTpow, self.exitrates)  # P(Y=y)
            self.piTyt = den

            self.bi += count * (self.initdist * Ttprod) / den
            self.ni += count * (piTpow * self.exitrates) / den
            # element i,j of the update is phgen[i,j] * Jmat[j,i], i.e. the
            # generator against the TRANSPOSE of the J-matrix. Unlike the
            # continuous case, i==j (self-transition, "stay another discrete
            # step") is a real, estimable DPH parameter and is included here.
            self.nij += count * (self.phgen * Jmat.T) / den
            self.loglikelihood += count * np.log(den)

    def __rightcensored(self, limits: np.array, counts: np.array) -> None:
        """
        Updates b_i, n_i, n_ij and the log-likelihood with the contribution from
        the right-censored observations, known only to satisfy Y > limit.

        Args:
            limits (ndarray): Distinct censoring limits, increasing.
            counts (ndarray): How often each of those limits occurs.

        Returns:
            None
        """

        n = self.nphases

        for count, mat in zip(counts, self.__powers(self.__blockK(), limits)):

            Tpow = mat[:n, :n]
            Kmat = mat[:n, n:]

            piTpow = np.matmul(self.initdist, Tpow)
            den = np.sum(piTpow)  # pi T^limit e = P(Y > limit)
            self.piTyt = den

            self.bi += count * (self.initdist * np.sum(Tpow, axis=1)) / den
            self.nij += count * (self.phgen * Kmat.T) / den
            # N_i(limit) = 0 identically when Y > limit, so ni is untouched
            self.loglikelihood += count * np.log(den)

    def __leftcensored(self, limits: np.array, counts: np.array) -> None:
        """
        Updates b_i, n_i, n_ij and the log-likelihood with the contribution from
        the left-censored observations, known only to satisfy Y <= limit.

        Equivalent to __intervalcensored with lower limit zero (M(0)=0, K(0)=0,
        T^0=I in closed form), written directly to avoid the wasted
        zero-argument work.

        Args:
            limits (ndarray): Distinct censoring limits, increasing.
            counts (ndarray): How often each of those limits occurs.

        Returns:
            None
        """

        n = self.nphases

        # both block matrices are wanted at the same exponents, so the two
        # sequences of powers are walked side by side
        walkM = self.__powers(self.__blockM(), limits)
        walkK = self.__powers(self.__blockK(), limits)

        for count, matM, matK in zip(counts, walkM, walkK):

            Tpow = matM[:n, :n]
            Mmat = matM[:n, n:]
            Kmat = matK[:n, n:]

            piTpow = np.matmul(self.initdist, Tpow)
            den = 1.0 - np.sum(piTpow)  # P(Y <= limit)
            self.piTyt = den

            piM = np.matmul(self.initdist, Mmat)

            self.bi += count * (self.initdist * (1.0 - np.sum(Tpow, axis=1))) / den
            self.nij += count * (self.phgen * (piM[:, None] - Kmat.T)) / den
            self.ni += count * (self.exitrates * piM) / den
            self.loglikelihood += count * np.log(den)

    def __intervalcensored(
            self,
            lefts: np.array,
            rights: np.array,
            counts: np.array
        ) -> None:
        """
        Updates b_i, n_i, n_ij and the log-likelihood with the contribution from
        the interval-censored observations, known only to lie in (left, right].

        Both ends of an interval are needed at once, so unlike the other three
        kinds this one first collects the powers at every distinct endpoint and
        then visits the distinct intervals.

        Args:
            lefts (ndarray): Lower limits of the distinct intervals.
            rights (ndarray): Upper limits of the distinct intervals.
            counts (ndarray): How often each of those intervals occurs.

        Returns:
            None
        """

        n = self.nphases

        endpoints = np.unique(np.concatenate([lefts, rights]))

        walkM = self.__powers(self.__blockM(), endpoints)
        walkK = self.__powers(self.__blockK(), endpoints)

        # the blocks are copied out rather than kept as views, so that only the
        # three n by n blocks of each endpoint are held rather than the whole
        # block matrix
        atendpoint = {}
        for endpoint, matM, matK in zip(endpoints, walkM, walkK):
            atendpoint[int(endpoint)] = (
                np.array(matM[:n, :n]),
                np.array(matM[:n, n:]),
                np.array(matK[:n, n:]),
            )

        for count, left, right in zip(counts, lefts, rights):

            TpowL, MmatL, KmatL = atendpoint[int(left)]
            TpowR, MmatR, KmatR = atendpoint[int(right)]

            piTpowL = np.matmul(self.initdist, TpowL)
            piTpowR = np.matmul(self.initdist, TpowR)
            den = np.sum(piTpowL) - np.sum(piTpowR)  # P(left < Y <= right)
            self.piTyt = den

            piM = np.matmul(self.initdist, MmatR - MmatL)
            KmatDiff = KmatR - KmatL

            self.bi += count * (self.initdist
                                * (np.sum(TpowL, axis=1)
                                   - np.sum(TpowR, axis=1))) / den
            self.nij += count * (self.phgen * (piM[:, None] - KmatDiff.T)) / den
            self.ni += count * (self.exitrates * piM) / den
            self.loglikelihood += count * np.log(den)

    def __storefundamental(
                            self,
                            initdist: np.array,
                            phgen: np.array,
                            exitrates: np.array
                        ):
        """
        Stores the DPH distribution's fundamental parameters as instance attributes so they
        can be accessed by the other methods during the E-step calculations.

        Args:
            initdist (ndarray): Initial distribution vector.
            phgen (ndarray): Phase-type generator matrix.
            exitrates (ndarray): Exit-probability vector.

        Returns:
            None
        """

        self.initdist = np.asarray(initdist).reshape(-1)
        self.phgen = np.asarray(phgen)
        self.exitrates = np.asarray(exitrates).reshape(-1)

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

        Observations are checked to be whole numbers because run() casts them
        to integers, which would otherwise truncate a fraction in silence.

        Args:
            obs (ndarray): Array of observations.
            initdist (ndarray): Initial distribution vector.
            phgen (ndarray): Phase-type generator matrix.
            exitrates (ndarray): Exit-probability vector.
            censoring (ndarray): Specifies censored observations, or None.

        Raises:
            ValueError: If an input is infeasible.
        """

        if (initdist.size != self.nphases or exitrates.size != self.nphases
                or phgen.shape != (self.nphases, self.nphases)):
            raise ValueError("The initial distribution, the generator and the exit probabilities must all be of dimension %d." % self.nphases)

        if not (initdist.min() >= 0.0 and initdist.sum() <= 1.0 + 1e-9):
            raise ValueError("The initial distribution must be non-negative and sum to at most one.")

        # the bound on the row sums also keeps the exit probabilities
        # themselves below one
        if not (phgen.min() >= 0.0 and exitrates.min() >= 0.0
                and (phgen.sum(axis=1) + exitrates).max() <= 1.0 + 1e-9):
            raise ValueError("The generator and the exit probabilities do not form a sub-transition matrix.")

        if obs is self.checkedobs and censoring is self.checkedcensoring:
            return None

        if obs.ndim != 1 or obs.size == 0:
            raise ValueError("The observations must be a non-empty one-dimensional array.")

        if censoring is None:
            uncensored = obs
        else:
            if (censoring.ndim != 2 or censoring.shape[1] != 2
                    or censoring.shape[0] < obs.size):
                raise ValueError("The censoring array must have two columns and at least one row per observation.")

            # zero is admissible: it is the lower limit of a left-censored
            # interval
            known = censoring[np.isfinite(censoring)]

            if known.size and (not known.min() >= 0.0
                               or np.any(known != np.floor(known))):
                raise ValueError("The censoring limits must be non-negative whole numbers.")

            # the value in obs is ignored for a censored observation
            uncensored = obs[np.all(np.isnan(censoring[:obs.size]), axis=1)]

        if uncensored.size and (
                not uncensored.min() >= 1
                or (not np.issubdtype(uncensored.dtype, np.integer)
                    and np.any(uncensored != np.floor(uncensored)))):
            raise ValueError("Every uncensored observation must be a whole number of at least one.")

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
            phgen (ndarray): Phase-type generator matrix.
            exitrates (ndarray): Exit-probability vector.
            censoring (ndarray): Specifies censored observations.

        Returns:
            ndarray: Number of processes that initiated in state i (b_i).
            ndarray: Number of processes that exited to the absorbing state from state i (n_i).
            ndarray: Number of processes that jumped from state i to state j (n_ij).

        Raises:
            ValueError: If an input is infeasible.
        """

        # left in the type they arrive in, since converting would rebuild the
        # sample at every iteration and defeat the caching in __checkinputs
        obs = np.asarray(obs)
        initdist = np.asarray(initdist, dtype=float).reshape(-1)
        phgen = np.asarray(phgen, dtype=float)
        exitrates = np.asarray(exitrates, dtype=float).reshape(-1)

        if censoring is not None:
            censoring = np.asarray(censoring, dtype=float)

        self.__checkinputs(obs, initdist, phgen, exitrates, censoring)

        self.__storefundamental(initdist, phgen, exitrates)

        self.bi.fill(0)
        self.ni.fill(0)
        self.nij.fill(0)

        self.loglikelihood = 0.0

        if censoring is None:
            values, counts = np.unique(obs.astype(np.int64),
                                       return_counts=True)
            if values.size:
                self.__uncensored(values, counts)
            return self.bi, self.ni, self.nij

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
            values, counts = np.unique(obs[isuncensored].astype(np.int64),
                                       return_counts=True)
            self.__uncensored(values, counts)

        if np.any(isright):
            limits, counts = np.unique(censoring[isright, 1].astype(np.int64),
                                       return_counts=True)
            self.__rightcensored(limits, counts)

        if np.any(isleft):
            limits, counts = np.unique(censoring[isleft, 0].astype(np.int64),
                                       return_counts=True)
            self.__leftcensored(limits, counts)

        if np.any(isinterval):
            intervals, counts = np.unique(
                censoring[isinterval, :].astype(np.int64), axis=0,
                return_counts=True)
            self.__intervalcensored(intervals[:, 0], intervals[:, 1], counts)

        return self.bi, self.ni, self.nij
