import sys
import time
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

    This class implements the proportional-intensities model of 
    Bladt (2022), given the covariates x and a random effect b,
    Y ~ PH(pi, exp(x'beta + b) T), so a positive coefficient means
    shorter absorption times. Note though that only homogeneous phase-type
    distributions are considered (similar to the fit class).
    
    The model is specified by a formula, e.g. 'y ~ x + z',
    'y ~ x * z' (corresponding to 'y ~ x + z + x:z') or 'y ~ x + z + (1|g)'
    with a Gaussian random effect (a random intercept) per level of g. 

    Observations at zero are treated independently from the non-zero observations 
    assuming that the probability of a zero-observation is constant (i.e. independent
    from the covariates).
    
    The fixed effects of the formula and the individual coefficients are tested by
    likelihood-ratio tests (getlrtest, summary, coeftest), and the coefficients get
    profile-likelihood confidence intervals (confint); Wald tests and intervals are
    available on request. The random effect is tested by a score test (getsigmatest,
    summary), see Commenges and Andersen (1995) and Lin (1997).

    The data are imported as a pandas DataFrame. A column with the 'category' dtype is
    categorical, and a numeric column is continuous.

    Right-, left- and interval-censored observations are given by two columns of
    limits, named in the argument censoring. Censored observations count as positive,
    as in the fit class.

    References:
        Bladt, M. (2022). Phase-type distributions for claim severity regression modeling.
        ASTIN Bulletin, 52(2), 417-448. https://doi.org/10.1017/asb.2021.40

        Commenges, D., & Andersen, P. K. (1995). Score test of homogeneity for survival
        data. Lifetime Data Analysis, 1(2), 145-156. https://doi.org/10.1007/BF00985764

        Lin, X. (1997). Variance component testing in generalised linear models with
        random effects. Biometrika, 84(2), 309-326. https://doi.org/10.1093/biomet/84.2.309

        Self, S. G., & Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood
        estimators and likelihood ratio tests under nonstandard conditions. Journal of
        the American Statistical Association, 82(398), 605-610.
        https://doi.org/10.1080/01621459.1987.10478472
    """

    def __init__(
        self,
        formula: str = None,
        data=None,
        nphases: int = 2,
        dtype: str = "coxian",
        quadnodes: int = 7,
        restarts: int = 10,
        seed: int = None,
        tolerance: float = 1e-5,
        itermax: int = 100000,
        accelerate: bool = True,
        verbose: bool = False,
        censoring: tuple = None,
    ) -> None:
        """
        Initialize and fit a phase-type regression.

        Args:
            formula (str):
                The model, e.g. 'y ~ x + z', 'y ~ x * z' or 'y ~ x + (1|g)';
                ':' for an interaction, '*' for an interaction with its
                lower-order effects, at most one random effect.
            data (DataFrame):
                The data. The response must be numeric and non-negative, a
                categorical variable and the grouping variable must have the
                'category' dtype.
            nphases (int, default=2):
                Number of phases.
            dtype (str, default="coxian"):
                Structure: "coxian", "gencoxian", "generlang", "hyperexp" or
                "general".
            quadnodes (int, default=7):
                Adaptive Gauss-Hermite nodes per group, at least 2.
            restarts (int, default=10):
                Number of random starts, besides a deterministic start for
                the chain structures. The likelihood often has several local
                maxima.
            seed (int, optional):
                Random seed.
            tolerance (float, default=1e-5):
                Convergence tolerance on the remaining gain in the
                log-likelihood.
            itermax (int, default=100000):
                Maximum number of iterations of each start.
            accelerate (bool, default=True):
                Whether to accelerate the iterations by SQUAREM.
            verbose (bool, default=False):
                If True, prints progress output during fitting.
            censoring (tuple, optional):
                Names of two numeric columns of data, the lower and the upper
                censoring limit of each observation. A limit can be set to missing
                (np.nan) when there is no limit, both missing for an uncensored
                observation, only the lower for a right-censored one (Y > lower),
                only the upper for a left-censored one (Y <= upper), both for an
                interval-censored one (lower < Y <= upper). The response is
                ignored, and may be missing, in censored rows.

        Notes
        -----
        Input validation is performed automatically. If validation fails,
        program execution terminates.
        """

        # set parameters
        self.formula = formula
        self.data = data
        self.censoringcolumns = censoring
        self.nphases = nphases
        self.dtype = dtype
        self.quadnodes = quadnodes
        self.restarts = restarts
        self.seed = seed
        self.tolerance = tolerance
        self.itermax = itermax
        self.accelerate = accelerate
        self.verbose = verbose
        # random starts of each fit to a simulated data set in getlrtest
        self.lrsimrestarts = 1
        # getlrtest, coeftest and confint say how long they may take when that
        # exceeds this (seconds)
        self.lrnotice = 10.0
        # expected refits per profile-likelihood interval, for the runtime note
        self.profilepoints = 12

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
            dict: Coefficient of each column of the design, named 'x',
                'g[level]' and 'x:g[level]'.
        """
        return dict(zip(self.design.columns, self.beta.tolist()))

    def getstderr(self) -> dict:
        """
        Return the standard errors of the regression coefficients, from the
        observed information on the centred design with the phase-type
        parameters profiled out.

        Returns:
            dict: Standard error of each coefficient.
        """
        return dict(zip(self.design.columns, self.stderr.tolist()))

    def coeftest(self, method: str = "lrt", nsim: int = 0, seed: int = None, confidence: float = 0.95,
                 coefficients=None, show: bool = True) -> dict:
        """
        Test whether each coefficient is zero using either the likelihood-ratio
        test ("lrt"), which is the default, or the Wald test ("wald"). When the
        likelihood-ratio test is selected, the reference distribution can be simulated
        by setting nsim > 0. Note that the Wald test is much faster to compute, but
        less accurate in small samples.

        Args:
            method (str, default="lrt"): "lrt" or "wald".
            nsim (int, default=0): With "lrt", simulated data sets per
                coefficient; 0 for none.
            seed (int, optional): Random seed of the simulation.
            confidence (float, default=0.95): Confidence level of the Monte
                Carlo interval of a simulated p-value.
            coefficients (str or list, optional): The coefficients to test, named
                as in getcoef; all when omitted.
            show (bool, default=True): Whether to print the table.

        Returns:
            dict: For each coefficient, a dictionary with 'estimate', 'stderr',
                'statistic' (the likelihood-ratio statistic, or z for "wald") and
                'pvalue', and with nsim > 0 'simpvalue', 'siminterval' and 'nsim'.
                None if the arguments are invalid.
        """

        names = self.__checkcoefargs(method, ("lrt", "wald"), nsim, seed, confidence, coefficients, show)
        if names is None:
            return None

        out = {}
        if method == "wald":
            pvalues = self._getwaldcoeftest()
            for name in names:
                k = self.design.columns.index(name)
                b, se = float(self.beta[k]), float(self.stderr[k])
                out[name] = dict(estimate=b, stderr=se, statistic=b / se if np.isfinite(se) and se > 0 else np.nan,
                                 pvalue=pvalues[name])
        else:
            indices = [self.design.columns.index(name) for name in names]
            refits = sum((k,) not in self.__lrcache for k in indices)
            expected = refits * self.fittime + 2 * nsim * len(names) * self.__simfittime()
            if expected > self.lrnotice:
                print("Note: The likelihood-ratio tests of the coefficients are being computed. This can take "
                      "about %s." % self.__duration(expected), flush=True)
            rng = np.random.default_rng(seed)
            for name, k in zip(names, indices):
                result = dict(estimate=float(self.beta[k]), stderr=float(self.stderr[k]))
                dropped = self.__dropfit([k], name)
                if dropped is None:
                    result.update(statistic=np.nan, pvalue=np.nan)
                    out[name] = result
                    continue
                statistic, reduced = dropped
                result.update(statistic=float(statistic), pvalue=float(chi2.sf(statistic, 1)))
                if nsim:
                    keep = [j for j in range(self.design.d) if j != k]
                    simulated = self.__simulatelr(reduced, keep, nsim, rng, name)
                    hits = int(np.count_nonzero(simulated >= statistic))
                    m = simulated.size
                    result.update(simpvalue=(hits + 1.0) / (m + 1.0) if m else np.nan,
                                  siminterval=self.__clopperpearson(hits, m, confidence) if m else (np.nan, np.nan),
                                  nsim=m)
                out[name] = result

        if show:
            if not out:
                print("The model has no coefficients.")
                return out
            width = max(12, max(len(name) for name in out) + 2)
            if method == "wald":
                print("Wald tests of the coefficients (every intensity is multiplied by exp(x'beta), "
                      "i.e. positive means lower response):")
                print("%-*s %10s %10s %9s %9s" % (width, "", "Estimate", "Std.Err", "z value", "Pr(>|z|)"))
                for name, row in out.items():
                    print("%-*s %10.4f %10.4f %9.2f %9s" % (width, name, row["estimate"], row["stderr"],
                                                         row["statistic"], self.__ptext(row["pvalue"])))
            else:
                print("Likelihood-ratio tests of the coefficients (every intensity is multiplied by "
                      "exp(x'beta), i.e. positive means lower response):")
                header = "%-*s %10s %10s %9s %9s" % (width, "", "Estimate", "Std.Err", "LR stat", "Pr(>Chi)")
                if nsim:
                    header += " %9s %17s" % ("Sim. p", "%g%% MC interval" % (100.0 * confidence))
                print(header)
                for name, row in out.items():
                    text = "%-*s %10.4f %10.4f %9.2f %9s" % (width, name, row["estimate"], row["stderr"],
                                                           row["statistic"], self.__ptext(row["pvalue"]))
                    if nsim:
                        text += " %9.4f %8.4f %8.4f" % ((row["simpvalue"],) + tuple(row["siminterval"]))
                    print(text)
                if nsim:
                    print("Simulated p-values from %d data sets per coefficient simulated without the "
                          "coefficient." % nsim)

        return out

    def confint(self, confidence: float = 0.95, method: str = "profile", nsim: int = 0, seed: int = None,
                coefficients=None, show: bool = True) -> dict:
        """
        Return confidence intervals of the coefficients. The default is the
        profile-likelihood interval ("profile"), optionally with a simulated critical
        value (nsim > 0). The alternative is the Wald interval ("wald"). Note that
        the Wald interval is much faster to compute, but less accurate in small
        samples.

        Args:
            confidence (float, default=0.95): Confidence level.
            method (str, default="profile"): "profile" or "wald".
            nsim (int, default=0): With "profile", simulated data sets per
                coefficient for a simulated critical value; 0 for none.
            seed (int, optional): Random seed of the simulation.
            coefficients (str or list, optional): The coefficients, named as in
                getcoef; all when omitted.
            show (bool, default=True): Whether to print the table.

        Returns:
            dict: For each coefficient, a dictionary with 'estimate', 'lower' and
                'upper' (infinite for an unbounded interval), with "profile" also
                'critical' (the critical value), and with nsim > 0 'nsim'. None if
                the arguments are invalid.
        """

        names = self.__checkcoefargs(method, ("profile", "wald"), nsim, seed, confidence, coefficients, show)
        if names is None:
            return None

        out = {}
        if method == "wald":
            intervals = self._getwaldconfint(confidence)
            for name in names:
                k = self.design.columns.index(name)
                out[name] = dict(estimate=float(self.beta[k]), lower=float(intervals[name][0]),
                                 upper=float(intervals[name][1]))
        else:
            chain = 2 if self.dtype in ("coxian", "gencoxian", "generlang") else 1
            expected = len(names) * (self.profilepoints * chain + 2 * nsim * (self.lrsimrestarts + chain)) \
                * self.fittime / max(1, len(self.d.startstatus))
            if expected > self.lrnotice:
                print("Note: The profile-likelihood intervals are being computed. This can take about %s."
                      % self.__duration(expected), flush=True)
            rng = np.random.default_rng(seed)
            quantile = float(chi2.ppf(confidence, 1))
            for name in names:
                k = self.design.columns.index(name)
                result = dict(estimate=float(self.beta[k]))
                if (k,) in self.__lrcache:
                    # the refit without the column is the profile at zero
                    statistic, reduced = self.__lrcache[(k,)]
                    self.__profilecache.setdefault(k, {})[0.0] = (self.d.loglikelihood - statistic / 2.0, reduced)
                critical = quantile
                if nsim:
                    simulated = self.__simulateprofile(k, nsim, rng, name)
                    critical = float(np.quantile(simulated, confidence, method="higher")) if simulated.size else np.nan
                    result["nsim"] = int(simulated.size)
                scale = float(self.design.scale[k])
                ends = [self.__profilebound(k, critical, side, name) / scale for side in (-1.0, 1.0)]
                result.update(lower=float(ends[0]), upper=float(ends[1]), critical=critical)
                out[name] = result

        if show:
            if not out:
                print("The model has no coefficients.")
                return out
            width = max(12, max(len(name) for name in out) + 2)
            kind = "Wald" if method == "wald" else "Profile-likelihood"
            print("%s confidence intervals of the coefficients (%g%%):" % (kind, 100.0 * confidence))
            header = "%-*s %10s %10s %10s" % (width, "", "Estimate", "Lower", "Upper")
            if method == "profile":
                header += " %10s" % "Crit. val."
            print(header)
            for name, row in out.items():
                text = "%-*s %10.4f %10.4f %10.4f" % (width, name, row["estimate"], row["lower"], row["upper"])
                if method == "profile":
                    text += " %10.4f" % row["critical"]
                print(text)
            if method == "profile":
                if nsim:
                    print("Critical values simulated from %d data sets per coefficient, simulated from the fit." % nsim)
                else:
                    print("Critical value: The %g%% quantile of chi-square(1)." % (100.0 * confidence))

        return out

    def getwaldtest(self) -> dict:
        """
        Return the Wald test of each fixed effect: b' V^(-1) b for its
        coefficients b with covariance V, against a chi-square distribution
        with one degree of freedom per coefficient. The likelihood-ratio test
        (getlrtest) is the default test of an effect; for an effect within a
        higher-order interaction both depend on the coding, as type III tests.

        Returns:
            dict: For each effect ('x', 'g', 'x:g'), the tuple (statistic,
                degrees of freedom, p-value); NaN without a covariance.
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

    def getlrtest(self, effects=None, nsim: int = 0, seed: int = None, confidence: float = 0.95) -> dict:
        """
        Return the likelihood-ratio test of each fixed effect: The
        model is refitted without the effect's columns, and
        D = 2 (loglik - loglik without the effect) is referred to a chi-square
        distribution. With nsim > 0 the reference distribution is also
        simulated by a parametric bootstrap from the model without the effect,
        with a Clopper-Pearson interval for the Monte Carlo error; this takes
        two fits per simulated data set. Zeros do not affect the test. When
        the refits are expected to take more than lrnotice seconds, a note will
        notify the user.

        Args:
            effects (str or list, optional): The fixed effects to test, named as in
                the formula ('x', 'g', 'x:g'); all effects when omitted.
            nsim (int, default=0): Simulated data sets per effect; 0 for none.
            seed (int, optional): Random seed of the simulation.
            confidence (float, default=0.95): Confidence level of the Monte
                Carlo interval.

        Returns:
            dict: For each effect, a dictionary with 'statistic', 'df' and
                'pvalue', and with nsim > 0 'simpvalue', 'siminterval' and
                'nsim' (the data sets fitted). None if the arguments are invalid.
        """

        available = list(self.design.termcolumns)
        if effects is None:
            effects = available
        elif isinstance(effects, str):
            effects = [effects]
        if not isinstance(effects, (list, tuple)) or not all(isinstance(t, str) for t in effects):
            print("Error: The fixed effects must be given as a name or a list of names, e.g. 'x' or ['x', 'x:g'].")
            return None
        unknown = [t for t in effects if t not in self.design.termcolumns]
        if unknown:
            print("Error: '%s' is not a fixed effect of the formula. The fixed effects are: %s."
                  % (unknown[0], ", ".join(available) if available else "none"))
            return None
        if not isinstance(nsim, int) or isinstance(nsim, bool) or nsim < 0:
            print("Error: The number of simulations must be an integer of at least 0.")
            return None
        if nsim and self.censored:
            print("Error: Simulated p-values are not available with censored observations, which would need a model of the censoring; use nsim=0.")
            return None
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            print("Error: The seed can only be specified as an integer.")
            return None
        if not isinstance(confidence, float) or not 0.0 < confidence < 1.0:
            print("Error: The confidence level must be a float between 0 and 1.")
            return None

        # the refits take about as long as the fit itself, notify user when that is long
        refits = sum(tuple(sorted(self.design.termcolumns[term])) not in self.__lrcache for term in effects)
        expected = refits * self.fittime + 2 * nsim * len(effects) * self.__simfittime()
        if expected > self.lrnotice:
            print("Note: The likelihood-ratio tests are being computed. This can take about %s."
                  % self.__duration(expected), flush=True)

        rng = np.random.default_rng(seed)
        out = {}
        for term in effects:
            index = list(self.design.termcolumns[term])
            keep = [k for k in range(self.design.d) if k not in index]
            df = len(index)

            dropped = self.__dropfit(index, term)
            if dropped is None:
                out[term] = dict(statistic=np.nan, df=df, pvalue=np.nan)
                continue
            statistic, reduced = dropped
            result = dict(statistic=float(statistic), df=df, pvalue=float(chi2.sf(statistic, df)))

            if nsim:
                simulated = self.__simulatelr(reduced, keep, nsim, rng, term)
                k = int(np.count_nonzero(simulated >= statistic))
                m = simulated.size
                result.update(simpvalue=(k + 1.0) / (m + 1.0) if m else np.nan,
                              siminterval=self.__clopperpearson(k, m, confidence) if m else (np.nan, np.nan),
                              nsim=m)

            out[term] = result

        return out

    def getsigma(self) -> float:
        """
        Return the estimated standard deviation of the random effect.

        Returns:
            float: The random effect std. dev., None without a random effect; zero at the boundary,
                where the fit is that of the model without random effect.
        """
        return self.sigma

    def getsigmatest(self, method: str = "score") -> dict:
        """
        Return the test of random effect std. dev. = 0. The default is a score
        test, which holds its level also with few groups, see Commenges and
        Andersen (1995) and Lin (1997). The likelihood-ratio test ("lrt") is
        referred to 0.5 chi2(0) + 0.5 chi2(1), see Self and Liang (1987), and is
        conservative with few groups.

        Args:
            method (str, default="score"): "score" or "lrt".

        Returns:
            dict: 'method', 'statistic' (standardized score, or likelihood
                ratio) and 'pvalue'; None without a random effect.
        """
        if method not in ("score", "lrt"):
            print("Error: The test of the random effect must be 'score' or 'lrt'.")
            return None
        if self.sigma is None:
            return None
        if method == "lrt":
            statistic, pvalue = self.__sigmatest()
        else:
            statistic, pvalue = self.d.sigmascore["statistic"], self.d.sigmascore["pvalue"]
        return dict(method=method, statistic=float(statistic), pvalue=float(pvalue))

    def getranef(self) -> dict:
        """
        Return the predicted random effect of each group, its conditional
        mean given the group's data.

        Returns:
            dict: Predicted random effect of each group label, or None without a
                random effect.
        """
        if self.ranef is None:
            return None
        return dict(zip(self.design.grouplabels, self.ranef.tolist()))

    def getinitdist(self) -> np.array:
        """
        Return the initial distribution of the baseline phase-type distribution.
        With observations at zero it sums to 1 - p, as in fit.

        Returns:
            np.array: The initial distribution.
        """
        return self.pi

    def getzeroprob(self) -> float:
        """
        Return the estimated probability of an observation at zero, the
        proportion of zeros in the data.

        Returns:
            float: The probability, zero without zeros.
        """
        return self.zeroprob

    def getphasegen(self) -> np.array:
        """
        Return the baseline phase-type generator: Continuous covariates zero,
        categorical covariates at their reference levels, random effect zero.

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
        Return the log-likelihood of the fitted model, the marginal one with a
        random effect.

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
        return -2.0 * self.loglikelihood + self.nparam * np.log(self.nobs)

    def getdist(self, covariates: dict = None, group=None) -> dist:
        """
        Return the conditional distribution of the response as a dist object.

        Args:
            covariates (dict, optional): Value of each variable, a level for a
                categorical variable; the baseline when omitted.
            group (optional): Label of a group, whose predicted random effect
                is then included; zero when omitted.

        Returns:
            dist: PH(pi, exp(eta) T), or None if the input is unusable.
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

        f = self.__unif.vectorrun(self.phgen, np.array([np.exp(eta) * float(x)]), self.pi, self.exitrates)[0]
        return float(np.exp(eta) * f)

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
        if not x >= 0.0:
            return 0.0
        if x == 0.0:
            # the point mass at zero, if any
            return float(self.zeroprob)

        u = np.array([np.exp(eta) * float(x)])
        return float(1.0 - self.__unif.vectorrun(self.phgen, u, self.pi, np.ones(self.phgen.shape[0]))[0])

    def summary(self, effecttest: str = "lrt", nsim: int = 0, seed: int = None) -> None:
        """
        Print a summary of the model and its effects.

        Args:
            effecttest (str, default="lrt"): The test of the fixed effects: "lrt"
                (getlrtest), "wald" (getwaldtest) or None.
            nsim (int, default=0): With "lrt", simulated data sets per effect
                for a simulated p-value; 0 for none.
            seed (int, optional): Random seed of the simulation.

        Returns:
            None: Prints to standard output.
        """

        if effecttest not in ("lrt", "wald", None):
            print("Error: The effect test must be 'lrt', 'wald' or None.")
            return None
        if not isinstance(nsim, int) or isinstance(nsim, bool) or nsim < 0:
            print("Error: The number of simulations must be an integer of at least 0.")
            return None
        if nsim and effecttest != "lrt":
            print("Error: Simulated p-values are available for the likelihood-ratio tests (effecttest='lrt').")
            return None
        if nsim and self.censored:
            print("Error: Simulated p-values are not available with censored observations. Use nsim=0.")
            return None
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            print("Error: The seed can only be specified as an integer.")
            return None

        # The likelihood-ratio tests first: They refit the model, which can take
        # a while, and getlrtest says so before it starts
        effects = list(self.design.termcolumns)
        if effects and effecttest == "lrt":
            lrtests = self.getlrtest(effects, nsim=nsim, seed=seed)

        # the model and the data
        print("Phase-type regression (%s, %d phases)" % (self.dtype, self.nphases))
        print("Formula: %s" % self.formula)
        line = "Observations: %d" % self.nobs
        notes = []
        if self.nzero:
            notes.append("%d at zero" % self.nzero)
        for count, label in zip(self.ncensored, ("right", "left", "interval")):
            if count:
                notes.append("%d %s-censored" % (count, label))
        if notes:
            line += " (%s)" % ", ".join(notes)
        if self.design.group is not None:
            sizes = np.bincount(self.design.groupcodes)
            line += "   Groups (%s): %d, of %d to %d observations" % (self.design.group, sizes.size,
                                                                     int(sizes.min()), int(sizes.max()))
        print(line)
        print("Fixed effects: %d, with %d coefficient%s" % (len(effects), self.design.d, "" if self.design.d == 1 else "s")
              if effects else "Fixed effects: None (no covariates)")
        print("")

        # the fixed effects
        if effects and effecttest is not None:
            width = max(12, max(len(effect) for effect in effects) + 2)
            if effecttest == "wald":
                tests = self.getwaldtest()
                print("Wald tests of the fixed effects:")
                print("%-*s %10s %5s %9s" % (width, "", "Chi-sq", "df", "Pr(>Chi)"))
                for effect in effects:
                    statistic, df, p = tests[effect]
                    print("%-*s %10.2f %5d %9s" % (width, effect, statistic, df, self.__ptext(p)))
            else:
                tests = lrtests
                print("Likelihood-ratio tests of the fixed effects:")
                if nsim:
                    print("%-*s %10s %5s %9s %9s %17s" % (width, "", "LR stat", "df", "Pr(>Chi)",
                                                         "Sim. p", "95% MC interval"))
                else:
                    print("%-*s %10s %5s %9s" % (width, "", "LR stat", "df", "Pr(>Chi)"))
                for effect in effects:
                    test = tests[effect]
                    text = "%-*s %10.2f %5d %9s" % (width, effect, test["statistic"], test["df"],
                                                    self.__ptext(test["pvalue"]))
                    if nsim:
                        text += " %9.4f %8.4f %8.4f" % ((test["simpvalue"],) + tuple(test["siminterval"]))
                    print(text)
                if nsim:
                    print("Simulated p-values from %d data sets per effect simulated without the effect." % nsim)
            print("(The tests of the individual coefficients are given by coeftest(), their confidence "
                  "intervals by confint())")
            print("")

        # the random effect and the point mass
        if self.sigma is not None:
            test = self.getsigmatest()
            if self.boundary:
                print("Random effect (1|%s): Std. dev. = 0 (at the boundary, i.e. groups vary no more "
                      "than the fixed effects explain)" % self.design.group)
            else:
                print("Random effect (1|%s): Std. dev. = %.4f (std. err. %.4f)"
                      % (self.design.group, self.sigma, self.sigmastderr))
            print("  Score test of std. dev. = 0: z = %.2f, p-value %s"
                  % (test["statistic"], self.__ptext(test["pvalue"])))
        if self.nzero:
            print("Probability of observations being zero: p = %.4f (std. err. %.4f)"
                  % (self.zeroprob, np.sqrt(self.zeroprob * (1.0 - self.zeroprob) / self.nobs)))
            print("Note: The probability of a zero is assumed constant")
        if self.sigma is not None or self.nzero:
            print("")

        # the fit
        nsigma = int(self.sigma is not None)
        nzero = int(self.nzero > 0)
        nphase = self.nparam - self.design.d - nsigma - nzero
        parts = ["%d phase-type" % nphase, "%d coefficient%s" % (self.design.d, "" if self.design.d == 1 else "s")]
        if nsigma:
            parts.append("1 random effect std. dev.")
        if nzero:
            parts.append("1 zero prob.")
        print("Log-likelihood: %.4f   Deviance (-2 log-likelihood): %.4f"
              % (self.loglikelihood, -2.0 * self.loglikelihood))
        print("AIC: %.4f   BIC: %.4f   Parameters: %d (%s)"
              % (self.getaic(), self.getbic(), self.nparam, ", ".join(parts)))
        statuses = list(self.d.startstatus)
        best = self.d.fixedloglikelihood
        reached = sum(1 for v, status in zip(self.d.startlogliks, statuses)
                      if status in ("converged", "itermax") and np.isfinite(v) and v >= best - 1e-3)
        print("Iterations: %d   Converged: %s   Starts: %d, of which %d reached the best within 0.001"
              % (self.iterations, "Yes" if self.converged else "No", len(statuses), reached))
        if self.d.degenerate:
            print("Warning: A phase of the fitted distribution holds a negligible share of the expected time; "
                  "fewer phases may suffice.")

    def plot(
        self,
        confint: bool = False,
        confidence: float = 0.95,
        title: str = "PP plot of the fitted conditional CDFs",
        filename: str = "PPcheck.png"
    ) -> None:
        """
        PP plot of the probability integral transforms F(y_i | x_i, b_i) of the
        positive observations against the uniform distribution, b_i being the
        predicted random effect of the group. Within a group the transforms are
        dependent, so the plot is a visual check rather than a test.

        Args:
            confint (bool, default=False): Whether to add pointwise
                Clopper-Pearson intervals.
            confidence (float, default=0.95): Confidence level for the intervals.
            title (str): Plot title.
            filename (str, default="PPcheck.png"): Filename of the generated graph.

        Returns:
            None: Writes the figure to filename; nothing with censored
                observations, whose transforms are censored too.
        """

        if self.censored:
            warnings.warn(
                "Some observations are censored, so the probability integral transforms "
                "are censored too; the PP plot is left out and no file is written.",
                UserWarning, stacklevel=2)
            return None

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
        plt.close()

    # ----------------------------------------------------------------------
    #   INTERNAL METHODS (used by coeftest)
    # ----------------------------------------------------------------------

    def _getwaldcoeftest(self) -> dict:
        """
        Return the two-sided Wald p-value of each coefficient,
        2 P(Z > |estimate / std. err.|).

        Returns:
            dict: p-value of each coefficient, NaN without a standard error.
        """
        out = {}
        for name, b, se in zip(self.design.columns, self.beta, self.stderr):
            out[name] = float(2.0 * norm.sf(abs(b / se))) if np.isfinite(se) and se > 0 else np.nan
        return out

    def _getwaldconfint(self, confidence: float = 0.95) -> dict:
        """
        Return the Wald confidence interval of each coefficient.

        Args:
            confidence (float, default=0.95): Confidence level.

        Returns:
            dict: The interval (lower, upper) of each coefficient.
        """
        z = norm.ppf(0.5 + confidence / 2.0)
        return {name: (b - z * se, b + z * se)
                for name, b, se in zip(self.design.columns, self.beta, self.stderr)}

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS
    # ----------------------------------------------------------------------

    def __checkinputs(self) -> bool:
        """
        Validate the input, parse the formula and build the design.

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
        if not isinstance(self.verbose, bool):
            print("Error: The argument 'verbose' needs to be of type 'bool'.")
            return False
        if self.censoringcolumns is not None:
            names = self.censoringcolumns
            if (not isinstance(names, (tuple, list)) or len(names) != 2
                    or not all(isinstance(name, str) for name in names) or names[0] == names[1]):
                print("Error: The censoring must be given as the names of two columns of the data, "
                      "the lower and the upper limit, e.g. censoring=('lower', 'upper').")
                return False
            self.censoringcolumns = tuple(names)

        try:
            self.design = _regdesign(self.data, self.parsed.response, self.parsed.terms,
                                     self.parsed.group, self.censoringcolumns)
        except ValueError as error:
            print("Error: %s" % error)
            return False

        # zeros are a point mass; the regression is fitted to the positive
        # observations, censored ones included
        zeros = self.design.y == 0.0
        self.nobs = self.design.n
        self.nzero = int(np.count_nonzero(zeros))
        self.zeroprob = self.nzero / self.nobs
        if self.nzero:
            if self.nobs - self.nzero < 2:
                print("Error: At least two observations must be positive; the others are fitted as a point mass at zero.")
                return False
            try:
                self.design = _regdesign(self.data[~zeros], self.parsed.response, self.parsed.terms,
                                         self.parsed.group, self.censoringcolumns)
            except ValueError as error:
                print("Error: %s (This concerns the %d positive observations; the %d at zero are fitted as a point mass.)"
                      % (str(error).rstrip("."), self.nobs - self.nzero, self.nzero))
                return False

        # the number of rows of each kind; with only right-censored rows the
        # likelihood grows as the rates go to zero
        kind = self.design.kind
        self.ncensored = [0, 0, 0] if kind is None else [int(np.count_nonzero(kind == k)) for k in (1, 2, 3)]
        self.censored = sum(self.ncensored) > 0
        if kind is not None and np.all(kind == 1):
            print("Error: Every positive observation is right-censored, so the model cannot be estimated.")
            return False

        # sigma is identified only through the dependence within groups
        if self.design.group is not None:
            sizes = np.bincount(self.design.groupcodes)
            if np.count_nonzero(sizes >= 2) < 5:
                warnings.warn(
                    "Fewer than five groups have two or more observations, so the "
                    "standard deviation of the random effect is poorly identified.",
                    UserWarning, stacklevel=3)

        return True

    def __structure(self) -> tuple:
        """
        Return the non-zero patterns of the initial distribution, generator
        and exit rates of the structure dtype, as in fit.

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
        Fit the model and transform the estimates back to the scale of the data.

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
        # kept for the likelihood-ratio tests
        self.__groups = groups
        self.__lrobs = design.y[self.order]
        self.__lrcensoring = design.limits[self.order] if self.censored else None
        # refits without columns, keyed by the dropped columns, and profile
        # points, keyed by column and value
        self.__lrcache = {}
        self.__profilecache = {}
        exact = np.ones(design.n, dtype=bool) if not self.censored else np.all(np.isnan(self.__lrcensoring), axis=1)
        self.__exactrows = exact

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
            verbose=self.verbose,
            censoring=self.__lrcensoring,
        )
        started = time.time()
        self.d.fit()
        self.fittime = time.time() - started

        if self.d.failed:
            print("Error: The fit failed: No start produced a finite likelihood.")
            return 1

        self.converged = self.d.converged
        self.boundary = self.d.boundary
        self.iterations = self.d.iterations
        self.loglikelihood = self.d.loglikelihood
        self.nparam = self.d.nparam

        # the point mass at zero, its probability one more parameter
        if self.nzero:
            npositive = self.nobs - self.nzero
            self.loglikelihood += (self.nzero * np.log(self.zeroprob)
                                   + npositive * np.log(1.0 - self.zeroprob))
            self.nparam += 1

        # back to the scale of the data; the centring moved delta into the
        # scale of the generator
        self.beta, delta = design.backtransform(self.d.beta)
        factor = np.exp(-delta)
        self.pi = (1.0 - self.zeroprob) * np.asarray(self.d.pi, dtype=float)
        self.phgen = factor * self.d.phgen
        self.exitrates = factor * self.d.exitrates
        self.sigma = self.d.sigma

        # cov(beta) = cov(beta~) / (s s'); the standard error of sigma by the
        # delta method from that of log sigma
        standard = np.sqrt(np.diag(self.d.covariance))
        self.covariance = self.d.covariance[:design.d, :design.d] / np.outer(design.scale, design.scale)
        self.stderr = standard[:design.d] / design.scale if design.d else np.zeros(0)
        self.sigmastderr = None
        if design.group is not None:
            self.sigmastderr = float(self.sigma * standard[design.d])

        self.ranef = None
        if design.group is not None:
            self.ranef = np.asarray(self.d.ranef, dtype=float)

        self.__unif = _unif(tolerance=1e-14)

        # probability integral transforms of the positive observations; with
        # censored observations they are censored too, and not computed
        self.pit = None
        if self.censored:
            if self.d.degenerate:
                print("Warning: The fitted distribution has a phase that holds a negligible share of the expected time.")
            return 0
        basepi = np.asarray(self.d.pi, dtype=float)
        eta = design.X @ self.beta
        if self.ranef is not None:
            eta = eta + self.ranef[design.groupcodes]
        u = np.exp(eta) * design.y
        self.pit = np.empty(design.n)
        ones = np.ones(self.phgen.shape[0])
        for start in range(0, design.n, 2048):
            survival = self.__unif.vectorrun(self.phgen, u[start:start + 2048], basepi, ones)
            self.pit[start:start + 2048] = 1.0 - survival

        if self.d.degenerate:
            print("Warning: The fitted distribution has a phase that holds a negligible share of the expected time.")

        return 0

    def __engine(self, obs: np.array, X: np.array, restarts: int, seed: int, starts: list, censoring=False):
        """
        Fit the engine with the settings of the fit, without covariance and
        warnings, for the likelihood-ratio tests and the profiles.

        Args:
            obs (ndarray): Positive observations, in the order of the fit.
            X (ndarray): Centred design, in the order of the fit.
            restarts (int): Number of random starts.
            seed (int): Random seed of the engine.
            starts (list): Further starting values (see _fitcphreg).
            censoring (ndarray, optional): Censoring limits in the order of the
                fit; those of the data when omitted.

        Returns:
            _fitcphreg: The fitted engine, or None if the fit failed.
        """

        initdist, initphgen, initexitrates = self.__structure()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine = _fitcphreg(obs=obs, design=X, groups=self.__groups, initpi=initdist,
                                initphgen=initphgen, initexitrates=initexitrates,
                                quadnodes=self.quadnodes, restarts=restarts, seed=seed,
                                tolerance=self.tolerance, itermax=int(self.itermax),
                                accelerate=self.accelerate, verbose=False, starts=starts,
                                covariance=False,
                                censoring=self.__lrcensoring if censoring is False else censoring)
            engine.fit()
        return None if engine.failed else engine

    @staticmethod
    def __startfrom(engine) -> dict:
        """
        Return the estimates of a fitted engine as a starting value.

        Args:
            engine (_fitcphreg): A fitted engine.

        Returns:
            dict: Copies of 'pi', 'T' and 'beta'.
        """
        return dict(pi=np.array(engine.pi, dtype=float), T=np.array(engine.phgen, dtype=float),
                    beta=np.array(engine.beta, dtype=float).reshape(-1))

    @staticmethod
    def __insert(values: np.array, keep: list, d: int) -> np.array:
        """
        Return the coefficients of the kept columns as a full coefficient
        vector, zero in the dropped columns.

        Args:
            values (ndarray): Coefficients of the kept columns.
            keep (list): Indices of the kept columns.
            d (int): Number of columns of the full model.

        Returns:
            ndarray: The full coefficient vector.
        """
        out = np.zeros(d)
        out[keep] = values
        return out

    def __checkcoefargs(self, method: str, methods: tuple, nsim: int, seed, confidence: float,
                        coefficients, show: bool):
        """
        Check the arguments of coeftest and confint.

        Args:
            method (str): The method asked for.
            methods (tuple): The methods available, the simulating one first.
            nsim (int): Simulated data sets.
            seed: Random seed.
            confidence (float): Confidence level.
            coefficients: Names of coefficients, or None for all.
            show (bool): Whether to print.

        Returns:
            list: The names of the coefficients, or None (with an error printed)
                if an argument is invalid.
        """
        if method not in methods:
            print("Error: The method must be '%s' or '%s'." % methods)
            return None
        if not isinstance(nsim, int) or isinstance(nsim, bool) or nsim < 0:
            print("Error: The number of simulations must be an integer of at least 0.")
            return None
        if nsim and method != methods[0]:
            print("Error: Simulations are available for method='%s'." % methods[0])
            return None
        if nsim and self.censored:
            print("Error: Simulations are not available with censored observations, which would need a model "
                  "of the censoring; use nsim=0.")
            return None
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            print("Error: The seed can only be specified as an integer.")
            return None
        if not isinstance(confidence, float) or not 0.0 < confidence < 1.0:
            print("Error: The confidence level must be a float between 0 and 1.")
            return None
        if not isinstance(show, bool):
            print("Error: The argument 'show' needs to be of type 'bool'.")
            return None
        columns = list(self.design.columns)
        if coefficients is None:
            return columns
        if isinstance(coefficients, str):
            coefficients = [coefficients]
        if not isinstance(coefficients, (list, tuple)) or not all(isinstance(c, str) for c in coefficients):
            print("Error: The coefficients must be given as a name or a list of names, e.g. 'x' or ['x', 'g[B]'].")
            return None
        unknown = [c for c in coefficients if c not in columns]
        if unknown:
            print("Error: '%s' is not a coefficient of the model. The coefficients are: %s."
                  % (unknown[0], ", ".join(columns) if columns else "none"))
            return None
        return list(dict.fromkeys(coefficients))

    def __dropfit(self, index: list, label: str):
        """
        Return the likelihood-ratio statistic for dropping columns of the
        design, refitting the model without them, also started from the fit.
        Refits are cached by the dropped columns, so getlrtest, summary and
        coeftest share them.

        Args:
            index (list): Indices of the dropped columns.
            label (str): Name of the effect or coefficient, for messages.

        Returns:
            tuple: The statistic and the fitted engine without the columns, or
                None (with an error printed) if that fit failed.
        """
        key = tuple(sorted(index))
        if key not in self.__lrcache:
            keep = [k for k in range(self.design.d) if k not in key]
            X = self.design.Xtilde[self.order]
            fullstart = self.__startfrom(self.d)
            reduced = self.__engine(self.__lrobs, X[:, keep], self.restarts, self.seed,
                                    [dict(fullstart, beta=fullstart["beta"][keep])])
            if reduced is None:
                print("Error: The model without '%s' could not be fitted." % label)
                return None
            full = self.d.loglikelihood
            if reduced.loglikelihood > full + 1e-6:
                # The fit is at a local maximum: Refit it from the smaller model
                start = self.__startfrom(reduced)
                start["beta"] = self.__insert(start["beta"], keep, self.design.d)
                refit = self.__engine(self.__lrobs, X, self.restarts, self.seed, [start])
                if refit is not None and refit.loglikelihood > full:
                    full = refit.loglikelihood
                print("Warning: The model without '%s' reached a higher likelihood than the fit, which "
                      "is therefore not at the maximum likelihood; refit with more restarts. The test "
                      "uses the better fit of the full model." % label)
            self.__lrcache[key] = (max(2.0 * (full - reduced.loglikelihood), 0.0), reduced)
        return self.__lrcache[key]

    def __simfittime(self) -> float:
        """
        Return the expected duration of one fit to a simulated data set: Its
        starts (the given one, the preset of a chain, lrsimrestarts random ones)
        at the duration of one start of the fit.

        Returns:
            float: Seconds.
        """
        chain = 2 if self.dtype in ("coxian", "gencoxian", "generlang") else 1
        return (self.lrsimrestarts + chain) * self.fittime / max(1, len(self.d.startstatus))

    def __profile(self, k: int, value: float, obs: np.array = None, start: dict = None, seed: int = None):
        """
        Return the profile log-likelihood with the coefficient of centred column k
        held at value: The model without the column, fitted to the responses
        (and censoring limits) multiplied by exp(value x_k), plus value times the
        sum of x_k over the uncensored rows, the Jacobian of that change of time
        scale. The fit starts from the nearest cached profile point (or the fit).

        Args:
            k (int): Index of the column on the centred design.
            value (float): The coefficient on the centred design.
            obs (ndarray, optional): Responses in the order of the fit; those of
                the data when omitted (and then cached).
            start (dict, optional): Starting value, with the coefficients of the
                kept columns.
            seed (int, optional): Random seed of the engine; that of the fit when
                omitted.

        Returns:
            tuple: The profile log-likelihood (-inf if the fit failed) and the
                fitted engine (None if it failed).
        """
        X = self.design.Xtilde[self.order]
        keep = [j for j in range(self.design.d) if j != k]
        xk = X[:, k]
        cache = self.__profilecache.setdefault(k, {}) if obs is None else None
        if cache is not None and value in cache:
            return cache[value]
        if start is None:
            if cache:
                nearest = min(cache, key=lambda v: abs(v - value))
                start = self.__startfrom(cache[nearest][1]) if cache[nearest][1] is not None else None
            if start is None:
                full = self.__startfrom(self.d)
                start = dict(full, beta=full["beta"][keep])
        factor = np.exp(value * xk)
        y = (self.__lrobs if obs is None else obs) * factor
        censoring = None if self.__lrcensoring is None else self.__lrcensoring * factor[:, None]
        engine = self.__engine(y, X[:, keep], 0, self.seed if seed is None else seed, [start], censoring=censoring)
        loglik = -np.inf if engine is None else engine.loglikelihood + value * float(np.sum(xk[self.__exactrows]))
        if cache is not None:
            cache[value] = (loglik, engine)
        if self.verbose and obs is None:
            print("Profile of coefficient %d: Value %.6f, log-likelihood %.6f" % (k, value, loglik))
        return loglik, engine

    def __profilebound(self, k: int, critical: float, side: float, name: str) -> float:
        """
        Return one end of the profile-likelihood interval of centred column k:
        The value on the given side of the estimate where
        2 (loglik - profile loglik) equals the critical value, found by the
        secant method on the signed root of the statistic, bracketed.

        Args:
            k (int): Index of the column on the centred design.
            critical (float): The critical value of the statistic.
            side (float): -1.0 for the lower end, 1.0 for the upper.
            name (str): Name of the coefficient, for messages.

        Returns:
            float: The end on the centred scale; infinite if the profile does
                not reach the critical value, NaN if it cannot be evaluated.
        """
        if not np.isfinite(critical):
            return np.nan
        estimate = float(self.d.beta[k])
        full = self.d.loglikelihood
        target = np.sqrt(critical)
        se = float(np.sqrt(self.d.covariance[k, k])) if np.isfinite(self.d.covariance[k, k]) else np.nan
        step = se if np.isfinite(se) and se > 0 else 1.0
        warned = [False]

        def root(value):
            loglik, _ = self.__profile(k, value)
            if not np.isfinite(loglik):
                return np.nan
            if loglik > full + 1e-6 and not warned[0]:
                warned[0] = True
                print("Warning: The profile of '%s' reached a higher likelihood than the fit, which is "
                      "therefore not at the maximum likelihood; refit with more restarts." % name)
            return np.sqrt(max(2.0 * (full - loglik), 0.0))

        # bracket the end: (inner, r_inner) below the target, (outer, r_outer) above
        inner, rinner = estimate, 0.0
        outer = estimate + side * target * step
        router = root(outer)
        for _ in range(30):
            if not np.isfinite(router):
                return np.nan
            if router >= target:
                break
            # extrapolate the signed root linearly, by at least half the distance
            distance = abs(outer - estimate)
            slope = (router - rinner) / abs(outer - inner) if router > rinner else 0.0
            grow = (target - router) / slope if slope > 0 else 2.0 * distance
            grow = min(max(grow * 1.1, 0.5 * distance), 10.0 * distance)
            inner, rinner = outer, router
            outer = outer + side * grow
            if abs(outer - estimate) > 1e3 * step:
                return side * np.inf
            router = root(outer)
        else:
            return side * np.inf

        # the secant method within the bracket (Illinois variant)
        lo, rlo, hi, rhi = inner, rinner - target, outer, router - target
        for _ in range(40):
            if abs(rhi) < 1e-5 or abs(hi - lo) < 1e-7 * step:
                break
            trial = hi - rhi * (hi - lo) / (rhi - rlo)
            rtrial = root(trial)
            if not np.isfinite(rtrial):
                return np.nan
            rtrial -= target
            if rtrial * rhi < 0:
                lo, rlo = hi, rhi
            else:
                rlo /= 2.0
            hi, rhi = trial, rtrial
        return hi

    def __simulateprofile(self, k: int, nsim: int, rng, name: str) -> np.array:
        """
        Simulate the profile likelihood-ratio statistic of centred column k at
        its estimate, by a parametric bootstrap from the fit: The profile at the
        estimate is started from the parameters that generated the data, and the
        full model from both, so the statistic is not negative. Data sets whose
        fits fail are left out.

        Args:
            k (int): Index of the column on the centred design.
            nsim (int): Number of simulated data sets.
            rng (Generator): Random number generator.
            name (str): Name of the coefficient, for messages.

        Returns:
            ndarray: The simulated statistics.
        """
        X = self.design.Xtilde[self.order]
        keep = [j for j in range(self.design.d) if j != k]
        truth = self.__startfrom(self.d)
        eta0 = X @ truth["beta"]
        sigma0 = self.d.sigma if self.__groups is not None else None
        value = float(truth["beta"][k])

        values = []
        for r in range(nsim):
            y = self.__simulatenull(truth, eta0, sigma0, rng)
            seed = int(rng.integers(2 ** 31))
            loglik, profiled = self.__profile(k, value, obs=y, start=dict(truth, beta=truth["beta"][keep]), seed=seed)
            if profiled is None:
                continue
            start = self.__startfrom(profiled)
            start["beta"] = np.insert(start["beta"], k, value)
            full = self.__engine(y, X, self.lrsimrestarts, seed, [start, truth])
            if full is None:
                continue
            values.append(max(2.0 * (full.loglikelihood - loglik), 0.0))
            if self.verbose:
                print("Profile interval of %s: Simulation %d of %d, statistic %.4f" % (name, r + 1, nsim, values[-1]))

        if len(values) < nsim:
            print("Warning: %d of the %d simulated data sets for '%s' could not be fitted and were left out."
                  % (nsim - len(values), nsim, name))
        return np.array(values)

    def __sigmatest(self) -> tuple:
        """
        Return the likelihood-ratio test of sigma = 0 against the best
        fixed-effects start of the fit, referred to 0.5 chi2(0) + 0.5 chi2(1).

        Returns:
            tuple: The statistic and the p-value (1 at the boundary).
        """
        statistic = max(2.0 * (self.d.loglikelihood - self.d.fixedloglikelihood), 0.0)
        pvalue = 0.5 * float(chi2.sf(statistic, 1)) if statistic > 0.0 else 1.0
        return statistic, pvalue

    @staticmethod
    def __duration(seconds: float) -> str:
        """
        Format a duration for printing.

        Args:
            seconds (float): The duration.

        Returns:
            str: Seconds, minutes or hours.
        """
        if seconds < 90:
            return "%.0f seconds" % seconds
        if seconds < 5400:
            return "%.0f minutes" % (seconds / 60.0)
        return "%.1f hours" % (seconds / 3600.0)

    @staticmethod
    def __ptext(p: float) -> str:
        """
        Format a p-value for printing.

        Args:
            p (float): The p-value.

        Returns:
            str: '<0.001', three decimals, or 'NA'.
        """
        if not np.isfinite(p):
            return "NA"
        return "<0.001" if p < 0.001 else ("%.3f" % p)

    @staticmethod
    def __clopperpearson(k: int, m: int, confidence: float) -> tuple:
        """
        Return the Clopper-Pearson interval for k successes in m trials.

        Args:
            k (int): Number of successes.
            m (int): Number of trials, at least 1.
            confidence (float): Confidence level.

        Returns:
            tuple: Lower and upper limit.
        """
        alpha = 1.0 - confidence
        lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, m - k + 1))
        upper = 1.0 if k == m else float(beta.ppf(1.0 - alpha / 2.0, k + 1, m - k))
        return lower, upper

    @staticmethod
    def __simulateph(pi: np.array, T: np.array, n: int, rng) -> np.array:
        """
        Simulate n absorption times of PH(pi, T), running the Markov jump
        processes side by side.

        Args:
            pi (ndarray): Initial distribution.
            T (ndarray): Sub-intensity matrix.
            n (int): Number of absorption times.
            rng (Generator): Random number generator.

        Returns:
            ndarray: The absorption times.
        """

        p = T.shape[0]
        rates = -np.diag(T)
        jumps = np.zeros((p, p + 1))
        jumps[:, :p] = np.maximum(T, 0.0) / rates[:, None]
        np.fill_diagonal(jumps[:, :p], 0.0)
        jumps[:, p] = np.maximum(-np.sum(T, axis=1), 0.0) / rates
        cumulative = np.cumsum(jumps / np.sum(jumps, axis=1, keepdims=True), axis=1)
        start = np.cumsum(pi / np.sum(pi))

        state = np.minimum(np.searchsorted(start, rng.random(n), side="right"), p - 1)
        times = np.zeros(n)
        alive = np.arange(n)
        while alive.size:
            current = state[alive]
            times[alive] += rng.exponential(size=alive.size) / rates[current]
            nxt = np.minimum(np.sum(rng.random(alive.size)[:, None] >= cumulative[current], axis=1), p)
            state[alive] = nxt
            alive = alive[nxt < p]
        return times

    def __simulatenull(self, truth: dict, eta0: np.array, sigma0, rng) -> np.array:
        """
        Simulate positive responses in the order of the fit,
        y = U exp(-eta0 - b) with U ~ PH(pi, T) and b ~ N(0, sigma0^2) per group.

        Args:
            truth (dict): 'pi' and 'T' on the scale of the centred design.
            eta0 (ndarray): Linear predictor without random effect.
            sigma0 (float or None): Standard deviation of the random effect.
            rng (Generator): Random number generator.

        Returns:
            ndarray: The simulated responses.
        """
        eta = eta0
        if sigma0:
            eta = eta0 + rng.normal(scale=sigma0, size=len(self.design.grouplabels))[self.__groups]
        return self.__simulateph(truth["pi"], truth["T"], eta0.size, rng) * np.exp(-eta)

    def __simulatelr(self, reduced, keep: list, nsim: int, rng, term: str) -> np.array:
        """
        Simulate the likelihood-ratio statistic of an effect under the model
        fitted without it. The smaller model is started from the parameters
        that generated the data and the larger from the smaller fit, so the
        statistic is not negative. Data sets whose fits fail are left out.

        Args:
            reduced (_fitcphreg): The model fitted without the effect.
            keep (list): Indices of the columns without the effect.
            nsim (int): Number of simulated data sets.
            rng (Generator): Random number generator.
            term (str): Name of the effect, for progress output.

        Returns:
            ndarray: The simulated statistics.
        """

        X = self.design.Xtilde[self.order]
        Xr = X[:, keep]
        truth = self.__startfrom(reduced)
        eta0 = Xr @ truth["beta"] if keep else np.zeros(X.shape[0])
        sigma0 = reduced.sigma if self.__groups is not None else None

        values = []
        for r in range(nsim):
            y = self.__simulatenull(truth, eta0, sigma0, rng)
            seed = int(rng.integers(2 ** 31))
            fitted = self.__engine(y, Xr, self.lrsimrestarts, seed, [truth])
            if fitted is None:
                continue
            start = self.__startfrom(fitted)
            start["beta"] = self.__insert(start["beta"], keep, self.design.d)
            full = self.__engine(y, X, self.lrsimrestarts, seed, [start])
            if full is None:
                continue
            values.append(max(2.0 * (full.loglikelihood - fitted.loglikelihood), 0.0))
            if self.verbose:
                print("Likelihood-ratio test of %s: Simulation %d of %d, statistic %.4f"
                      % (term, r + 1, nsim, values[-1]))

        if len(values) < nsim:
            print("Warning: %d of the %d simulated data sets for '%s' could not be fitted and were left out."
                  % (nsim - len(values), nsim, term))
        return np.array(values)

    def __eta(self, covariates: dict, group) -> float:
        """
        Return the linear predictor for a set of covariates and a group.

        Args:
            covariates (dict): Value of each variable, or None for the baseline.
            group: Label of a group, or None for a zero random effect.

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
                print("Error: The model has no random effect, so a group cannot be given.")
                return None
            labels = self.design.grouplabels
            matches = [index for index, label in enumerate(labels) if label == group]
            if not matches:
                print("Error: '%s' is not a group of the data." % group)
                return None
            eta += float(self.ranef[matches[0]])

        return eta
