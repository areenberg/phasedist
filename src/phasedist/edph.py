import numpy as np


class edph:
    """
    Performs the E-step of the EM algorithm for a Discrete-Time Phase Type (DPH) distribution
    from p. 675 Bladt and Nielsen (2017), extended to censored observations by analogy with
    the continuous case (Section 13.5, p. 685) -- the book itself does not derive the censored
    DPH formulas (see Problem 13.8.7, p. 701, posed as an open exercise). Note: This class has
    no input checks.

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

        self.nphases = nphases
        self.bi = np.zeros(self.nphases)
        self.ni = np.zeros(self.nphases)
        self.nij = np.zeros((self.nphases, self.nphases))
        self.loglikelihood = 0.0

        return None

    def __Jmatrix(self, y: int) -> None:
        """
        Computes the J-matrix and T^(y-1), where
        J(y;pi,T) = sum_{k=0}^{y-2} T^(y-2-k) t pi T^k,
        via the discrete analogue of Theorem A.2.1 (Van Loan) for matrix powers
        (Theorem A.2.2, p. 714): for a block matrix [[T, t*pi],[0, T]], its
        (y-1)-th power has top-left block T^(y-1) and top-right block J(y;pi,T).
        Used in the uncensored N_ij formula (13.14). Requires y >= 1 (for y=1
        this correctly reduces to the zero matrix, since J(1;pi,T) is an empty sum).

        Args:
            y (int): Observation value (y >= 1).

        Returns:
            None
        """
        t = self.exitrates[:, None]
        pi = self.initdist[None, :]

        mat = np.linalg.matrix_power(
            np.block([
                [self.phgen, np.matmul(t, pi)],
                [np.zeros((self.nphases, self.nphases)), self.phgen],
            ]),
            y - 1,
        )

        self.Tpow = mat[: self.nphases, : self.nphases]
        self.Jmat = mat[: self.nphases, self.nphases : 2 * self.nphases]

    def __Mmatrix(self, y: int) -> None:
        """
        Computes the M-matrix and T^y, where M(y) = sum_{k=0}^{y-1} T^k.

        Via Theorem A.2.2 with block matrix [[T, I],[0, I]]: its y-th power has
        top-left block T^y and top-right block M(y) (since the (2,2) block must
        be the identity -- not the zero matrix as in the continuous __Mmatrix --
        so that its k-th power stays I for every k, giving a constant factor of
        I at each term of the sum, matching M(y) = sum T^k * I * I^k).

        Args:
            y (int): Observation value (y >= 0).

        Returns:
            None
        """
        mat = np.linalg.matrix_power(
            np.block([
                [self.phgen, np.eye(self.nphases)],
                [np.zeros((self.nphases, self.nphases)), np.eye(self.nphases)],
            ]),
            y,
        )

        self.Tpow = mat[: self.nphases, : self.nphases]
        self.Mmat = mat[: self.nphases, self.nphases : 2 * self.nphases]

    def __Kmatrix(self, y: int) -> None:
        """
        Computes the K-matrix and T^y, where
        K(y) = sum_{k=0}^{y-1} T^(y-1-k) e pi T^k, with e the column vector of
        ones (i.e. the same construction as __Jmatrix, but with e in place of
        the exit-rate vector t, and summed over one more term -- k=0..y-1
        rather than k=0..y-2 -- since here every one of the y steps may be a
        transient-to-transient transition).

        Args:
            y (int): Observation value (y >= 0).

        Returns:
            None
        """
        e = np.ones((self.nphases, 1))
        pi = self.initdist[None, :]

        mat = np.linalg.matrix_power(
            np.block([
                [self.phgen, np.matmul(e, pi)],
                [np.zeros((self.nphases, self.nphases)), self.phgen],
            ]),
            y,
        )

        self.Tpow = mat[: self.nphases, : self.nphases]
        self.Kmat = mat[: self.nphases, self.nphases : 2 * self.nphases]

    def __uncensored(self, y: int) -> None:
        """
        Updates b_i, n_i, n_ij with the contribution from a single fully
        observed (uncensored) observation y, using (13.13), (13.14), (13.15).

        Args:
            y (int): Observation value (y >= 1).

        Returns:
            None
        """

        self.__Jmatrix(y)  # computes self.Tpow (= T^(y-1)) and self.Jmat

        Ttprod = np.matmul(self.Tpow, self.exitrates)
        piTpow = np.matmul(self.initdist, self.Tpow)
        piTtprod = np.matmul(piTpow, self.exitrates)
        self.piTyt = piTtprod  # P(Y=y); likelihood contribution for this observation

        for i in range(self.nphases):
            self.bi[i] += (self.initdist[i] * Ttprod[i]) / piTtprod
            self.ni[i] += (piTpow[i] * self.exitrates[i]) / piTtprod
            for j in range(self.nphases):
                # unlike the continuous case, i==j (self-transition, "stay
                # another discrete step") is a real, estimable DPH parameter
                # (phgen[i,i]) and must be included here, not skipped.
                self.nij[i, j] += (self.phgen[i, j] * self.Jmat[j, i]) / piTtprod

    def __rightcensored(self, right: int) -> None:
        """
        Updates b_i, n_i, n_ij with the contribution from a single
        right-censored observation known only to satisfy Y > right.

        Derived by analogy with the continuous case's (13.38)/(13.40): K(y) is
        built from the rank-one matrix e*pi, so its (i,i) and (j,i) entries
        already equal the needed sums directly -- no separate construction
        is required. N_i(right) = 0 identically, since absorption hasn't
        happened yet, so ni is left untouched (verified by simulation).

        Args:
            right (int): It is known that Y > right.

        Returns:
            None
        """

        self.__Kmatrix(right)  # computes self.Tpow (= T^right) and self.Kmat (= K(right))

        piTpow = np.matmul(self.initdist, self.Tpow)
        den = np.sum(piTpow)  # pi T^right e = P(Y > right)
        self.piTyt = den

        Tpow_rowsum = np.sum(self.Tpow, axis=1)  # e_i' T^right e for each i

        for i in range(self.nphases):
            self.bi[i] += (self.initdist[i] * Tpow_rowsum[i]) / den
            for j in range(self.nphases):
                # i==j (self-transition) included -- see note in __uncensored.
                self.nij[i, j] += (self.phgen[i, j] * self.Kmat[j, i]) / den
            # N_i(right) = 0 identically when Y > right, so ni is untouched

    def __leftcensored(self, left: int) -> None:
        """
        Updates b_i, n_i, n_ij with the contribution from a single
        left-censored observation known only to satisfy Y <= left.

        Equivalent to __intervalcensored(0, left) (M(0)=0, K(0)=0, T^0=I in
        closed form), written directly to avoid the wasted zero-argument calls.

        Args:
            left (int): It is known that Y <= left.

        Returns:
            None
        """

        self.__Mmatrix(left)  # computes self.Tpow (= T^left) and self.Mmat (= M(left))
        Tpow = self.Tpow
        Mmat = self.Mmat

        self.__Kmatrix(left)  # computes self.Tpow (= T^left, again) and self.Kmat (= K(left))
        Kmat = self.Kmat

        piTpow = np.matmul(self.initdist, Tpow)
        den = 1.0 - np.sum(piTpow)  # P(Y <= left)
        self.piTyt = den

        piM = np.matmul(self.initdist, Mmat)
        Tpow_rowsum = np.sum(Tpow, axis=1)

        for i in range(self.nphases):
            self.bi[i] += (self.initdist[i] * (1.0 - Tpow_rowsum[i])) / den
            for j in range(self.nphases):
                # i==j (self-transition) included -- see note in __uncensored.
                self.nij[i, j] += (self.phgen[i, j] * (piM[i] - Kmat[j, i])) / den
            self.ni[i] += (self.exitrates[i] * piM[i]) / den

    def __intervalcensored(self, left: int, right: int) -> None:
        """
        Updates b_i, n_i, n_ij with the contribution from a single
        interval-censored observation known only to lie in (left, right].

        Args:
            left (int): Lower/left limit.
            right (int): Upper/right limit.

        Returns:
            None
        """

        self.__Mmatrix(left)
        TpowL = self.Tpow
        MmatL = self.Mmat

        self.__Kmatrix(left)
        KmatL = self.Kmat

        self.__Mmatrix(right)
        TpowR = self.Tpow
        MmatR = self.Mmat

        self.__Kmatrix(right)
        KmatR = self.Kmat

        piTpowL = np.matmul(self.initdist, TpowL)
        piTpowR = np.matmul(self.initdist, TpowR)
        den = np.sum(piTpowL) - np.sum(piTpowR)  # P(left < Y <= right)
        self.piTyt = den

        piM = np.matmul(self.initdist, MmatR - MmatL)
        KmatDiff = KmatR - KmatL
        TpowL_rowsum = np.sum(TpowL, axis=1)
        TpowR_rowsum = np.sum(TpowR, axis=1)

        for i in range(self.nphases):
            self.bi[i] += (self.initdist[i] * (TpowL_rowsum[i] - TpowR_rowsum[i])) / den
            for j in range(self.nphases):
                # i==j (self-transition) included -- see note in __uncensored.
                self.nij[i, j] += (self.phgen[i, j] * (piM[i] - KmatDiff[j, i])) / den
            self.ni[i] += (self.exitrates[i] * piM[i]) / den

    def __storefundamental(
                            self,
                            initdist: np.array,
                            phgen: np.array,
                            exitrates: np.array
                        ):
        """
        Stores the DPH distribution's fundamental parameters as instance attributes so they
        can be accessed by the other methods during the E-step calculations.

        initdist and exitrates are coerced to flat 1-D arrays, and phgen to a plain 2-D
        ndarray: if initdist/exitrates are passed as column vectors (shape (nphases,1))
        rather than flat vectors (shape (nphases,)), or if phgen is passed as a
        numpy.matrix rather than a plain ndarray, every matmul below that expects a
        1-D/plain-ndarray result would instead silently return a (nphases,1)-shaped
        array, or (if phgen is a numpy.matrix) a numpy.matrix -- a subclass whose
        arithmetic and indexing *always* stay 2-D, even for what should be a 1-D
        vector, since matrix multiplication with a numpy.matrix operand promotes the
        whole computation (block matrix, matrix_power, matmul) to numpy.matrix all the
        way through. Either way, a per-entry assignment such as `self.bi[i] += ...`
        then fails with "setting an array element with a sequence," since the
        right-hand side is a length-1 array/matrix rather than a scalar.
        `np.asarray(...)` strips both a numpy.matrix's subclass and any extra
        singleton dimension, so coercing all three inputs here -- once, centrally --
        avoids needing that at every call site (the original edph.py instead handled
        the column-vector case piecemeal, via scattered .flatten()/np.ravel() calls).

        Args:
            initdist (ndarray): Initial distribution vector.
            phgen (ndarray): Phase-type sub-transition matrix.
            exitrates (ndarray): Exit-probability vector.

        Returns:
            None
        """

        self.initdist = np.asarray(initdist).reshape(-1)
        self.phgen = np.asarray(phgen)
        self.exitrates = np.asarray(exitrates).reshape(-1)

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
            phgen (ndarray): Phase-type sub-transition matrix.
            exitrates (ndarray): Exit-probability vector.
            censoring (ndarray): Specifies censored observations.

        Returns:
            ndarray: Number of processes that initiated in state i (b_i).
            ndarray: Number of processes that exited to the absorbing state from state i (n_i).
            ndarray: Number of processes that jumped from state i to state j (n_ij).
        """

        self.__storefundamental(initdist, phgen, exitrates)

        self.bi.fill(0)
        self.ni.fill(0)
        self.nij.fill(0)

        self.loglikelihood = 0.0

        if censoring is not None:
            for idx, y in enumerate(obs):

                if np.isnan(censoring[idx, 0]) and np.isnan(censoring[idx, 1]):
                    self.__uncensored(int(y))  # uncensored observation
                elif np.isnan(censoring[idx, 0]) and not np.isnan(censoring[idx, 1]):
                    self.__rightcensored(int(censoring[idx, 1]))  # Right-censored observation
                elif not np.isnan(censoring[idx, 0]) and np.isnan(censoring[idx, 1]):
                    self.__leftcensored(int(censoring[idx, 0]))  # Left-censored observation
                elif not np.isnan(censoring[idx, 0]) and not np.isnan(censoring[idx, 1]):
                    self.__intervalcensored(int(censoring[idx, 0]), int(censoring[idx, 1]))  # Interval-censored observation

                self.loglikelihood += np.log(self.piTyt)  # depends on censoring type

        else:
            for y in obs:
                self.__uncensored(int(y))
                self.loglikelihood += np.log(self.piTyt)

        return self.bi, self.ni, self.nij
