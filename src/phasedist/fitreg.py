import sys
import warnings

import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.stats import beta, chi2, norm
from phasedist.fitcphreg import _fitcphreg
from phasedist.regdesign import _regformula, _regdesign
from phasedist.unif import _unif
from phasedist.dist import dist


class fitreg:
    """
    Fit a continuous-time phase-type regression to data.

    The model is specified by a formula in the notation of R's lm and lme4
    packages. 'y ~ x + z' is a regression of the response y on the variables x
    and z, 'y ~ x * z' adds their interaction (x + z + x:z), and
    'y ~ x + z + (1|g)' adds a Gaussian random intercept for each level of g.

    The regression is the proportional-intensities model of Bladt (2022):
    given the covariates x and the random intercept b of its group, an
    observation is the absorption time of a Markov jump process whose every
    intensity is multiplied by exp(x'beta + b). A positive coefficient
    therefore means shorter times. The conditional distribution is the
    phase-type distribution PH(pi, exp(x'beta + b) T), so that
    exp(x'beta + b) Y ~ PH(pi, T). The intercept is the scale of T, so the
    formula has none of its own. As in fit, only homogeneous phase-type
    distributions are fitted.

    The data are a pandas DataFrame, whose dtypes give the type of each
    variable: a column with the 'category' dtype is categorical, coded by
    treatment dummies against its first category (reorder the categories to
    change the reference level), and a numeric column is continuous. The
    grouping variable of a random intercept must be categorical.

    References:
        Bladt, M. (2022). Phase-type distributions for claim severity regression modeling.
        ASTIN Bulletin, 52(2), 417-448. https://doi.org/10.1017/asb.2021.40
    """

    def __init__(
        self,
        formula: str = None,
        data=None,
        nphases: int = 2,
        dtype: str = "coxian",
        quadnodes: int = 7,
        restarts: int = 5,
        seed: int = None,
        tolerance: float = 1e-5,
        itermax: int = 100000,
        accelerate: bool = True,
        staged: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initialize and fit a phase-type regression.

        Args:
            formula (str):
                The model, e.g. 'y ~ x + z', 'y ~ x * z' or
                'y ~ x + z + (1|g)'. Terms are variable names joined by '+',
                with ':' for an interaction and '*' for an interaction with
                its lower-order terms; at most one random intercept (1|g).
            data (DataFrame):
                The data, one column per variable. The response must be
                positive and numeric, a categorical variable (and the grouping
                variable) must have the 'category' dtype, and a continuous
                variable must be numeric.
            nphases (int, default=2):
                Number of phases.
            dtype (str, default="coxian"):
                Structure of the phase-type distribution: "coxian", "gencoxian",
                "generlang", "hyperexp" or "general".
            quadnodes (int, default=7):
                Adaptive Gauss-Hermite nodes per group for the random
                intercept, at least 2.
            restarts (int, default=5):
                Number of random starts, each run to convergence; the Coxian,
                generalized Coxian and generalized Erlang structures also get a
                deterministic start.
            seed (int, optional):
                Random seed.
            tolerance (float, default=1e-5):
                Convergence tolerance on the remaining gain in the
                log-likelihood.
            itermax (int, default=100000):
                Maximum number of iterations of each run to convergence.
            accelerate (bool, default=True):
                Whether to accelerate the iterations by SQUAREM.
            staged (bool, default=False):
                Whether to fit in stages: for every start, first the
                phase-type generator with the coefficients held at their
                quasi-likelihood start, then the coefficients with the
                generator held, then everything together; with a random
                intercept, first sigma and the coefficients with the generator
                held, then everything. The estimates are the same maximum
                likelihood estimates; only the path to them changes.
            verbose (bool, default=False):
                If True, prints progress output during fitting.

        Notes
        -----
        Input validation is performed automatically. If validation fails,
        program execution terminates.
        """

        # set parameters
        self.formula = formula
        self.data = data
        self.nphases = nphases
        self.dtype = dtype
        self.quadnodes = quadnodes
        self.restarts = restarts
        self.seed = seed
        self.tolerance = tolerance
        self.itermax = itermax
        self.accelerate = accelerate
        self.staged = staged
        self.verbose = verbose

        # checking and fitting
        if self.__checkinputs():
            if self.__fitmodel() != 0:
                sys.exit(1)  # terminate the program
        else:
            sys.exit(1)  # terminate the program

    # ----------------------------------------------------------------------
    #   PUBLIC METHODS
    # ----------------------------------------------------------------------

    def getcoef(self) -> dict:
        """
        Return the estimated regression coefficients.

        Returns:
            dict: Coefficient of each column of the design matrix, named after
                its variable, 'variable[level]' for a level of a categorical
                variable, and the names joined by ':' for an interaction.
        """
        return dict(zip(self.design.columns, self.beta.tolist()))

    def getstderr(self) -> dict:
        """
        Return the standard errors of the regression coefficients.

        Without a random intercept they are taken from the observed
        information with the phase-type representation held at its estimate,
        on the centred design; centring makes this conditional information
        valid for the coefficients (the information they share with the scale
        of the generator vanishes to first order). With a random intercept
        they are taken from the full observed information, the phase-type
        parameters profiled out, since sigma trades off against the shape of
        the base distribution.

        Returns:
            dict: Standard error of each coefficient.
        """
        return dict(zip(self.design.columns, self.stderr.tolist()))

    def getpvalue(self) -> dict:
        """
        Return the p-value of each regression coefficient: the two-sided Wald
        test that the coefficient is zero, 2 P(Z > |estimate / std. err.|).

        Returns:
            dict: p-value of each coefficient (NaN where the standard error is
                unavailable).
        """
        out = {}
        for name, b, se in zip(self.design.columns, self.beta, self.stderr):
            out[name] = float(2.0 * norm.sf(abs(b / se))) if np.isfinite(se) and se > 0 else np.nan
        return out

    def getwaldtest(self) -> dict:
        """
        Return the Wald test of each term of the formula: that all its
        coefficients are zero, e.g. every level of a categorical variable or
        every column of an interaction. The statistic b' V^(-1) b, with V the
        covariance of the term's coefficients, is referred to a chi-square
        distribution with as many degrees of freedom as the term has
        coefficients. For a term with one coefficient it is the square of the
        z value, and the p-value that of getpvalue.

        For a term that is part of a higher-order interaction in the model
        (x in 'y ~ x*g'), the test concerns its effect where the other
        variables of the interaction are zero or at their reference level, so
        it changes with the reference level and the origin of x, as type III
        tests do. The test of a highest-order term does not.

        Returns:
            dict: For each term, named as in the formula ('x', 'g', 'x:g'),
                the tuple (statistic, degrees of freedom, p-value); NaN where
                the covariance is unavailable.
        """
        out = {}
        for term, index in self.design.termcolumns.items():
            b = self.beta[index]
            V = self.covariance[np.ix_(index, index)]
            try:
                statistic = float(b @ np.linalg.solve(V, b)) if np.all(np.isfinite(V)) else np.nan
            except np.linalg.LinAlgError:
                statistic = np.nan
            pvalue = float(chi2.sf(statistic, len(index))) if np.isfinite(statistic) else np.nan
            out[term] = (statistic, len(index), pvalue)
        return out

    def getconfint(self, confidence: float = 0.95) -> dict:
        """
        Return Wald confidence intervals for the regression coefficients.

        Args:
            confidence (float, default=0.95): Confidence level.

        Returns:
            dict: The interval (lower, upper) of each coefficient.
        """
        z = norm.ppf(0.5 + confidence / 2.0)
        return {name: (b - z * se, b + z * se)
                for name, b, se in zip(self.design.columns, self.beta, self.stderr)}

    def getsigma(self) -> float:
        """
        Return the estimated standard deviation of the random intercept.

        Returns:
            float: sigma, or None for a model without random intercept. It is
                zero when the likelihood does not increase away from zero (the
                groups vary no more than the fixed effects explain); the fit is
                then that of the model without random intercept, a warning is
                given and sigma has no standard error.
        """
        return self.sigma

    def getranef(self) -> dict:
        """
        Return the predicted random intercept of each group, the mean of its
        conditional distribution given the group's data.

        Returns:
            dict: Predicted intercept of each group label, or None for a model
                without random intercept.
        """
        if self.ranef is None:
            return None
        return dict(zip(self.design.grouplabels, self.ranef.tolist()))

    def getinitdist(self) -> np.array:
        """
        Return the initial distribution of the baseline phase-type distribution.

        Returns:
            np.array: The initial distribution.
        """
        return self.pi

    def getphasegen(self) -> np.array:
        """
        Return the baseline phase-type generator: the one of an observation
        whose continuous covariates are zero, whose categorical covariates are
        at their reference levels, and whose random intercept is zero.

        Returns:
            np.array: The baseline sub-intensity matrix.
        """
        return self.phgen

    def getexitrates(self) -> np.array:
        """
        Return the exit rates of the baseline phase-type distribution.

        Returns:
            np.array: The baseline exit rates.
        """
        return self.exitrates

    def getloglik(self) -> float:
        """
        Return the log-likelihood of the fitted model, the marginal one when
        there is a random intercept.

        Returns:
            float: The log-likelihood.
        """
        return self.loglikelihood

    def getaic(self) -> float:
        """
        Compute Akaike's Information Criterion (AIC).

        Returns:
            float: The AIC value of the fitted model.
        """
        return -2.0 * self.loglikelihood + 2.0 * self.nparam

    def getbic(self) -> float:
        """
        Compute the Bayesian Information Criterion (BIC).

        Returns:
            float: The BIC value of the fitted model.
        """
        return -2.0 * self.loglikelihood + self.nparam * np.log(self.design.n)

    def getdist(self, covariates: dict = None, group=None) -> dist:
        """
        Return the conditional distribution of the response for given
        covariates as a dist object.

        Args:
            covariates (dict, optional): Value of each variable of the model, a
                level for a categorical variable. The baseline when omitted.
            group (optional): Label of a group, whose predicted random
                intercept is then included. Zero intercept when omitted.

        Returns:
            dist: The phase-type distribution PH(pi, exp(eta) T), or None if
                the covariates or the group are unusable.
        """
        eta = self.__eta(covariates, group)
        if eta is None:
            return None

        return dist(discrete=False, initdist=self.pi, phgen=np.exp(eta) * self.phgen,
                    seed=self.seed)

    def getdensity(self, x: float, covariates: dict = None, group=None) -> float:
        """
        Evaluate the conditional density of the response at x.

        Args:
            x (float): Evaluation point.
            covariates (dict, optional): Value of each variable, as for getdist.
            group (optional): Label of a group, as for getdist.

        Returns:
            float: The density.
        """
        eta = self.__eta(covariates, group)
        if eta is None:
            return None
        if x < 0.0:
            return 0.0
        if not x > 0.0:
            return float(np.exp(eta) * self.pi @ self.exitrates)

        a = self.pi @ self.__unif.run(self.phgen, np.array([np.exp(eta) * float(x)]))[0]
        return float(np.exp(eta) * (a @ self.exitrates))

    def getcumprob(self, x: float, covariates: dict = None, group=None) -> float:
        """
        Evaluate the conditional cumulative distribution function at x.

        Args:
            x (float): Evaluation point.
            covariates (dict, optional): Value of each variable, as for getdist.
            group (optional): Label of a group, as for getdist.

        Returns:
            float: The cumulative probability P(Y <= x).
        """
        eta = self.__eta(covariates, group)
        if eta is None:
            return None
        if not x > 0.0:
            return 0.0

        u = np.array([np.exp(eta) * float(x)])
        return float(1.0 - np.sum(self.pi @ self.__unif.run(self.phgen, u)[0]))

    def summary(self) -> None:
        """
        Print a summary of the fitted model: the coefficients with standard
        errors, Wald tests and 95% confidence intervals, the Wald test of each
        term with more than one coefficient, the random-intercept standard
        deviation where present, and the fit statistics.

        Returns:
            None: Prints to standard output.
        """

        print("Phase-type regression (%s, %d phases)" % (self.dtype, self.nphases))
        print("Formula: %s" % self.formula)
        line = "Observations: %d" % self.design.n
        if self.design.group is not None:
            line += "   Groups (%s): %d" % (self.design.group, len(self.design.grouplabels))
        print(line)
        print("")

        if self.design.d:
            print("Coefficients (every intensity is multiplied by exp(x'beta); positive means shorter times):")
            width = max(12, max(len(name) for name in self.design.columns) + 2)
            print("%-*s %10s %10s %9s %9s %21s" % (width, "", "Estimate", "Std.Err", "z value",
                                                   "Pr(>|z|)", "95% conf. int."))
            intervals = self.getconfint(0.95)
            pvalues = self.getpvalue()
            for name, b, se in zip(self.design.columns, self.beta, self.stderr):
                z = b / se if se > 0 else np.nan
                p = pvalues[name]
                lower, upper = intervals[name]
                ptext = "<0.001" if p < 0.001 else ("%.3f" % p)
                print("%-*s %10.4f %10.4f %9.2f %9s %10.4f %10.4f"
                      % (width, name, b, se, z, ptext, lower, upper))
            print("")

            tests = {term: test for term, test in self.getwaldtest().items() if test[1] > 1}
            if tests:
                print("Wald tests of the terms with more than one coefficient:")
                width = max(12, max(len(term) for term in tests) + 2)
                print("%-*s %10s %5s %9s" % (width, "", "Chi-sq", "df", "Pr(>Chi)"))
                for term, (statistic, df, p) in tests.items():
                    ptext = "<0.001" if p < 0.001 else ("%.3f" % p)
                    print("%-*s %10.2f %5d %9s" % (width, term, statistic, df, ptext))
                print("")

        if self.sigma is not None and self.boundary:
            print("Random intercept (1|%s): sigma = 0 (at the boundary: the groups vary no more "
                  "than the fixed effects explain)" % self.design.group)
        elif self.sigma is not None:
            print("Random intercept (1|%s): sigma = %.4f (std. err. %.4f)"
                  % (self.design.group, self.sigma, self.sigmastderr))
        print("Log-likelihood: %.4f   AIC: %.4f   BIC: %.4f   Parameters: %d"
              % (self.loglikelihood, self.getaic(), self.getbic(), self.nparam))
        print("Iterations: %d   Converged: %s" % (self.iterations, "yes" if self.converged else "no"))

    def plot(
        self,
        confint: bool = False,
        confidence: float = 0.95,
        title: str = "PP plot of the fitted conditional CDFs",
        filename: str = "PPcheck.png"
    ) -> None:
        """
        Plot the probability integral transforms F(y_i | x_i, b_i) of the
        observations against the uniform distribution (a PP plot). Under the
        model they are uniform, so the points should follow the diagonal. With
        a random intercept, b_i is the predicted intercept of the
        observation's group. Within a group the transforms are dependent, so
        the plot is a visual check rather than a test.

        Args:
            confint (bool, default=False): Whether to add pointwise
                Clopper-Pearson intervals for the empirical CDF of the
                transforms.
            confidence (float, default=0.95): Confidence level for the intervals.
            title (str): Plot title.
            filename (str, default="PPcheck.png"): Filename of the generated graph.

        Returns:
            None: Writes the figure to filename.
        """

        transforms = np.sort(self.pit)
        n = transforms.size
        counts = np.arange(1, n + 1)
        empirical = counts / float(n)

        plt.figure(figsize=(8, 6))
        plt.plot([0.0, 1.0], [0.0, 1.0], lw=1, linestyle="-", color="blue", label="Uniform")
        plt.plot(transforms, empirical, lw=1, linestyle="-", color="red",
                 label="Empirical CDF of the transforms")

        if confint:
            alpha = 1.0 - confidence
            lower = beta.ppf(alpha / 2.0, counts, n - counts + 1)
            upper = beta.ppf(1.0 - alpha / 2.0, counts + 1, n - counts)
            lower[counts == 0] = 0.0
            upper[counts == n] = 1.0
            plt.plot(transforms, upper, lw=1, linestyle="--", color="red", label="Upper conf. int.")
            plt.plot(transforms, lower, lw=1, linestyle="--", color="red", label="Lower conf. int.")

        plt.xlabel("Fitted conditional CDF")
        plt.ylabel("Empirical CDF")
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.savefig(filename)

        # the figure is only written to file, so it is closed rather than left
        # open for a later call to accumulate
        plt.close()

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS
    # ----------------------------------------------------------------------

    def __checkinputs(self) -> bool:
        """
        Validate input parameters, parse the formula and build the design
        prior to fitting.

        Returns:
            bool: True if all inputs are valid; otherwise False.
        """

        if self.formula is None:
            print("Error: A formula is needed, e.g. 'y ~ x + z + (1|g)'.")
            return False
        try:
            self.parsed = _regformula(self.formula)
        except ValueError as error:
            print("Error: %s" % error)
            return False

        if not isinstance(self.data, pandas.DataFrame):
            print("Error: The data must be given as a pandas DataFrame.")
            return False
        if not isinstance(self.nphases, int) or isinstance(self.nphases, bool) or self.nphases < 1:
            print("Error: The number of phases can only be specified as an integer larger than 0.")
            return False
        if self.dtype not in ("general", "generlang", "hyperexp", "coxian", "gencoxian"):
            print("Error: Unknown distribution type. The types are 'coxian', 'gencoxian', 'generlang', 'hyperexp' and 'general'.")
            return False
        if not isinstance(self.quadnodes, int) or isinstance(self.quadnodes, bool) or self.quadnodes < 2:
            print("Error: The number of quadrature nodes must be an integer of at least 2.")
            return False
        if not isinstance(self.restarts, int) or isinstance(self.restarts, bool) or self.restarts < 1:
            print("Error: The number of restarts must be an integer larger than 0.")
            return False
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            print("Error: The seed can only be specified as an integer.")
            return False
        if not isinstance(self.tolerance, float):
            print("Error: The argument 'tolerance' needs to be of type 'float'.")
            return False
        if (not isinstance(self.itermax, (int, float)) or isinstance(self.itermax, bool)
                or not self.itermax >= 1):
            print("Error: The argument 'itermax' needs to be a number of at least 1.")
            return False
        if not isinstance(self.accelerate, bool):
            print("Error: The argument 'accelerate' needs to be of type 'bool'.")
            return False
        if not isinstance(self.staged, bool):
            print("Error: The argument 'staged' needs to be of type 'bool'.")
            return False
        if not isinstance(self.verbose, bool):
            print("Error: The argument 'verbose' needs to be of type 'bool'.")
            return False

        try:
            self.design = _regdesign(self.data, self.parsed.response, self.parsed.terms,
                                     self.parsed.group)
        except ValueError as error:
            print("Error: %s" % error)
            return False

        # the variance of a random intercept is only identified through the
        # dependence within groups
        if self.design.group is not None:
            sizes = np.bincount(self.design.groupcodes)
            if np.count_nonzero(sizes >= 2) < 5:
                warnings.warn(
                    "Fewer than five groups have two or more observations, so the "
                    "standard deviation of the random intercept is poorly identified.",
                    UserWarning, stacklevel=3)

        return True

    def __structure(self) -> tuple:
        """
        Return the structure of the phase-type distribution named by dtype:
        the non-zero pattern of the initial distribution, the generator and
        the exit rates, as fit uses it for its random initialization.

        Returns:
            tuple: Initial distribution, generator and exit-rate patterns.
        """

        p = self.nphases
        initdist = np.zeros(p)
        initphgen = np.zeros((p, p))
        initexitrates = np.zeros(p)
        np.fill_diagonal(initphgen, 1.0)

        if self.dtype == "general":
            initdist[:] = 1.0
            initphgen[:, :] = 1.0
            initexitrates[:] = 1.0
        elif self.dtype == "hyperexp":
            initdist[:] = 1.0
            initexitrates[:] = 1.0
        else:
            for i in range(p - 1):
                initphgen[i, i + 1] = 1.0
            if self.dtype == "generlang":
                initdist[0] = 1.0
                initexitrates[p - 1] = 1.0
            elif self.dtype == "coxian":
                initdist[0] = 1.0
                initexitrates[:] = 1.0
            else:  # gencoxian
                initdist[:] = 1.0
                initexitrates[:] = 1.0

        return initdist, initphgen, initexitrates

    def __fitmodel(self) -> int:
        """
        Fit the model and transform the estimates back to the scale of the
        data.

        Returns:
            int: 0 if successful, 1 otherwise.
        """

        design = self.design
        initdist, initphgen, initexitrates = self.__structure()

        # the engine needs the observations of a group next to each other
        if design.groupcodes is not None:
            self.order = np.argsort(design.groupcodes, kind="stable")
            groups = design.groupcodes[self.order]
        else:
            self.order = np.arange(design.n)
            groups = None

        self.d = _fitcphreg(
            obs=design.y[self.order],
            design=design.Xtilde[self.order],
            groups=groups,
            initpi=initdist,
            initphgen=initphgen,
            initexitrates=initexitrates,
            quadnodes=self.quadnodes,
            restarts=self.restarts,
            seed=self.seed,
            tolerance=self.tolerance,
            itermax=int(self.itermax),
            accelerate=self.accelerate,
            staged=self.staged,
            verbose=self.verbose,
        )
        self.d.fit()

        if self.d.failed:
            print("Error: The fit failed: no start produced a finite likelihood.")
            return 1

        self.converged = self.d.converged
        self.boundary = self.d.boundary
        self.iterations = self.d.iterations
        self.loglikelihood = self.d.loglikelihood
        self.nparam = self.d.nparam

        # back to the scale of the data: the centring moved
        # delta = sum_k beta~_k c_k / s_k into the scale of the generator
        self.beta, delta = design.backtransform(self.d.beta)
        factor = np.exp(-delta)
        self.pi = np.asarray(self.d.pi, dtype=float)
        self.phgen = factor * self.d.phgen
        self.exitrates = factor * self.d.exitrates
        self.sigma = self.d.sigma

        # the covariance of the coefficients on the scale of the data:
        # beta = beta~ / s, so cov(beta) = cov(beta~) / (s s')
        standard = np.sqrt(np.diag(self.d.covariance))
        self.covariance = self.d.covariance[:design.d, :design.d] / np.outer(design.scale, design.scale)
        self.stderr = standard[:design.d] / design.scale if design.d else np.zeros(0)
        # the standard error of sigma by the delta method from that of
        # log sigma; none at the boundary sigma = 0
        self.sigmastderr = None
        if design.group is not None:
            self.sigmastderr = float(self.sigma * standard[design.d])

        self.ranef = None
        if design.group is not None:
            self.ranef = np.asarray(self.d.ranef, dtype=float)

        self.__unif = _unif(tolerance=1e-14)

        # the probability integral transforms, in the order of the data
        eta = design.X @ self.beta
        if self.ranef is not None:
            eta = eta + self.ranef[design.groupcodes]
        u = np.exp(eta) * design.y
        self.pit = np.empty(design.n)
        for start in range(0, design.n, 2048):
            block = self.__unif.run(self.phgen, u[start:start + 2048])
            self.pit[start:start + 2048] = 1.0 - np.sum(np.einsum("k,mkl->ml", self.pi, block), axis=1)

        if self.d.degenerate:
            print("Warning: The fitted distribution has a phase that holds a negligible share of the expected time.")

        return 0

    def __eta(self, covariates: dict, group) -> float:
        """
        Return the linear predictor for a set of covariates and a group.

        Args:
            covariates (dict): Value of each variable, or None for the baseline.
            group: Label of a group, or None for a zero intercept.

        Returns:
            float: The linear predictor, or None if the input is unusable.
        """

        try:
            x = self.design.row(covariates)
        except ValueError as error:
            print("Error: %s" % error)
            return None

        eta = float(x @ self.beta) if self.design.d else 0.0

        if group is not None:
            if self.ranef is None:
                print("Error: The model has no random intercept, so a group cannot be given.")
                return None
            labels = self.design.grouplabels
            matches = [index for index, label in enumerate(labels) if label == group]
            if not matches:
                print("Error: '%s' is not a group of the data." % group)
                return None
            eta += float(self.ranef[matches[0]])

        return eta
