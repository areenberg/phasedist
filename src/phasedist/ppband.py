import numpy as np
from scipy.special import betainc, betaincinv


class _ppband:
    """
    A simultaneous confidence band for n observations: Under the model, all
    n sorted probability integral transforms lie inside it at once with
    probability `confidence`. run() gives the band of the transforms, for the
    PP plot of fitreg; cdfband() the same band drawn around a fitted CDF, for
    the CDF plot of fit.

    The k-th of n sorted uniforms follows Beta(k, n-k+1). The band is the
    equal-precision ("tail-sensitive") band of Aldor-Noiman et al. (2013): The
    central interval of the same level gamma for every k, with gamma chosen by
    simulating sorted uniforms so that the whole curve is covered with the
    requested probability. It depends only on n and the level, not on the
    model, so no refits are needed.

    Warning: This class does not contain any input checks.

    References:
        Aldor-Noiman, S., Brown, L. D., Buja, A., Rolke, W. & Stine, R. A.
        (2013). The power to see: A new graphical test of normality. The
        American Statistician, 67(4), 249-260.
        https://doi.org/10.1080/00031305.2013.847865
    """

    def __init__(self, nsim: int = None, seed: int = 2013) -> None:
        """
        Initializes the band.

        Args:
            nsim (int, default=None): Number of simulated samples of sorted
                uniforms that set the level gamma. When None, 2e6/n within
                2,000 and 20,000: about a second of computing, and a coverage
                within 0.2-0.5 points of the requested one.
            seed (int, default=2013): Seed of the simulation, so the band of a
                given n and level is always the same.
        """

        self.nsim = nsim
        self.seed = seed
        self.gamma = None   # set by run()
        self.__bands = {}   # the bands computed so far, by n and level

        return None

    def run(self, n: int, confidence: float) -> tuple:
        """
        Returns the band of n sorted transforms. The band of each n and level
        is computed once per object.

        Args:
            n (int): Number of transforms.
            confidence (float): Probability that the whole curve lies inside.

        Returns:
            tuple: The lower and upper limit of the k-th sorted transform,
                k = 1,...,n (ndarray each).
        """

        if (n, confidence) in self.__bands:
            self.gamma, lower, upper = self.__bands[(n, confidence)]
            return lower, upper

        k = np.arange(1, n + 1, dtype=float)
        rng = np.random.default_rng(self.seed)

        # each simulated sample's smallest two-sided tail probability over k
        nsim = self.nsim if self.nsim is not None else int(min(max(2e6 / n, 2000), 20000))
        smallest = np.empty(nsim)
        batch = max(1, int(2e6 // n))
        for start in range(0, nsim, batch):
            stop = min(start + batch, nsim)
            ordered = np.sort(rng.random((stop - start, n)), axis=1)
            cdf = betainc(k, n - k + 1.0, ordered)
            smallest[start:stop] = np.min(2.0 * np.minimum(cdf, 1.0 - cdf), axis=1)

        # the curve leaves the band exactly when that probability is below gamma
        self.gamma = float(np.quantile(smallest, 1.0 - confidence))

        lower = betaincinv(k, n - k + 1.0, self.gamma / 2.0)
        upper = betaincinv(k, n - k + 1.0, 1.0 - self.gamma / 2.0)
        self.__bands[(n, confidence)] = (self.gamma, lower, upper)

        return lower, upper

    def cdfband(self, cdf: np.array, n: int, confidence: float) -> tuple:
        """
        Returns the band of run() around a fitted CDF: At a point x with fitted
        CDF value p, the empirical CDF of n observations from the model must
        lie between the number of upper limits below p and the number of
        lower limits at or below p, each over n. For a continuous model the
        empirical CDF lies inside at every x (left limits included) exactly
        when every sorted transform lies inside the band of run(), so with
        probability confidence. For a model with atoms, a discrete one or a
        point mass at zero, the band holds at least that probability.

        Args:
            cdf (ndarray): Fitted CDF values at the points of the plot.
            n (int): Number of observations.
            confidence (float): Probability that the whole empirical CDF lies
                inside.

        Returns:
            tuple: The lower and upper limit of the empirical CDF at each point
                (ndarray each).
        """

        lower, upper = self.run(n, confidence)
        cdf = np.asarray(cdf, dtype=float)

        return (np.searchsorted(upper, cdf, side="left") / float(n),
                np.searchsorted(lower, cdf, side="right") / float(n))
