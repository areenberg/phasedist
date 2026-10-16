import itertools
import re

import numpy as np


# a variable name: a letter (any alphabet) or an underscore, then letters,
# digits, underscores or dots; any other name is written in backticks, as in R
_NAME = re.compile(r"^[^\W\d][\w.]*$")
# a name in backticks is replaced by a placeholder while the formula is parsed
_QUOTED = re.compile(r"\x00(\d+)\x00")


class _regformula:
    """
    Parses a regression formula in the target notation, restricted
    to what fitreg supports: 'y ~ x + z', 'y ~ x * z' (x + z + x:z),
    'y ~ x + z + (1|g)' and 'y ~ 1 + (1|g)'. A name with spaces or other
    symbols is written in backticks, e.g. 'y ~ `body mass index` + age'.
    Spaces around the operators do not matter. Every effect is given once:
    'x + y + x*y' and 'x*y + x*y*z' are refused.
    An interaction of three or more variables needs all its lower-order
    effects; whether an interaction of two may leave out a main effect
    depends on the variable types and is decided by _regdesign. The effects
    are tuples of variable names, sorted by degree.

    Warning: This class returns a warning if the formula contains a syntax
    error.
    """

    def __init__(self, formula: str) -> None:
        """
        Parses the formula.

        Args:
            formula (str): The formula, e.g. 'y ~ x * z + (1|g)'.

        Raises:
            ValueError: If the formula is malformed or not supported.
        """

        self.formula = formula
        self.response = None
        self.terms = []
        self.variables = []
        self.group = None

        self.__parse()

        return None

    def __parse(self) -> None:
        """
        Splits the formula into response, fixed effects and grouping variable.

        Args:
            None

        Raises:
            ValueError: If the formula is malformed or not supported.
        """

        if not isinstance(self.formula, str):
            raise ValueError("The formula must be a string, e.g. 'y ~ x + z + (1|g)'.")

        # names in backticks become placeholders, so their symbols are not read
        # as operators; they are put back as each name is read
        pieces = self.formula.split("`")
        if len(pieces) % 2 == 0:
            raise ValueError("The formula has an unmatched backtick; names with spaces or other symbols are written as `name`.")
        quoted = pieces[1::2]
        for name in quoted:
            if name.strip() == "":
                raise ValueError("The formula has an empty name in backticks.")
        formula = "".join(piece if k % 2 == 0 else "\x00%d\x00" % (k // 2)
                          for k, piece in enumerate(pieces))

        def isname(token):
            return bool(_NAME.match(token)) or bool(_QUOTED.fullmatch(token))

        def unquote(token):
            match = _QUOTED.fullmatch(token)
            return quoted[int(match.group(1))] if match else token

        def show(text):
            return _QUOTED.sub(lambda m: "`%s`" % quoted[int(m.group(1))], text)

        sides = formula.split("~")
        if len(sides) != 2:
            raise ValueError("The formula must contain exactly one '~', e.g. 'y ~ x + z'.")

        response = sides[0].strip()
        if not isname(response):
            raise ValueError("The left-hand side of the formula must be the name of the response variable, not '%s'." % show(response))
        response = unquote(response)

        righthand = sides[1]

        # the random effects are the parenthesized parts
        groups = []
        for match in re.finditer(r"\(([^()]*)\)", righthand):
            inner = match.group(1)
            if "|" not in inner:
                raise ValueError("Parentheses are only used for random effects such as (1|g). Transformations such as log(x) and functions such as C(x) are not supported; categorical variables are marked by giving them the 'category' dtype in the DataFrame.")
            if "||" in inner:
                raise ValueError("The '||' notation is not supported; only random effects such as (1|g) are.")
            left, right = inner.split("|", 1)
            if left.strip() != "1":
                raise ValueError("Only random effects of the form (1|g), random intercepts, are supported; '(%s)' asks for random slopes." % show(inner.strip()))
            group = right.strip()
            if not isname(group):
                raise ValueError("The grouping variable of a random effect must be a single variable name, not '%s'." % show(group))
            groups.append(unquote(group))

        fixedpart = re.sub(r"\([^()]*\)", " ", righthand)
        if "(" in fixedpart or ")" in fixedpart:
            raise ValueError("The formula has unbalanced or nested parentheses.")

        if len(groups) > 1:
            raise ValueError("The class fitreg supports one random effect, but the formula has %d: %s. Keep one of them."
                             % (len(groups), ", ".join("(1|%s)" % group for group in groups)))

        # the fixed effects, each a tuple of variable names
        order = []          # variables in order of first appearance
        found = []          # effects in order of appearance
        for token in fixedpart.split("+"):
            term = token.strip()
            if term == "" or term == "1":
                continue
            if term == "0" or "-" in term:
                raise ValueError("The intercept cannot be removed: In a phase-type regression it is the scale of the phase-type generator.")
            if "*" in term and ":" in term:
                raise ValueError("'%s' mixes '*' and ':'; write such effects out with '+', e.g. x + z + x:z." % show(term))

            separator = "*" if "*" in term else ":"
            names = [name.strip() for name in term.split(separator)]
            for name in names:
                if not isname(name):
                    raise ValueError("'%s' is not a variable name. Effects are variable names joined by '+', with ':' or '*' for interactions; a name with spaces or other symbols is written in backticks, e.g. `body mass index`." % show(name))
            names = [unquote(name) for name in names]
            for name in names:
                if name not in order:
                    order.append(name)
            if len(set(names)) != len(names):
                raise ValueError("The effect '%s' names a variable more than once." % show(term))

            if separator == "*":
                # every main effect and interaction of the variables
                for size in range(1, len(names) + 1):
                    for subset in itertools.combinations(names, size):
                        found.append((subset, term))
            else:
                found.append((tuple(names), term))

        # every effect once: a repeated effect, written twice or also added by
        # a '*', means the formula is not what was intended (x + y + x*y was
        # probably meant as x + y + x:y); sorted by degree
        terms = []
        seen = {}
        for term, written in found:
            canonical = tuple(sorted(term, key=order.index))
            key = frozenset(canonical)
            if key in seen:
                raise ValueError("Infeasible formula: The effect '%s' is given more than once." % ":".join(canonical))
            seen[key] = written
            terms.append(canonical)
        terms.sort(key=len)

        # an interaction of three or more variables needs all its lower-order
        # effects; one of two variables is checked by _regdesign, which knows
        # the types (x + x:w and x:w are allowed for a continuous x and a
        # categorical w)
        for term in terms:
            if len(term) < 3:
                continue
            for size in range(1, len(term)):
                for subset in itertools.combinations(term, size):
                    if frozenset(subset) not in seen:
                        raise ValueError("The interaction '%s' needs its lower-order effect '%s' in the formula as well; '%s' adds them all." % (":".join(term), ":".join(subset), "*".join(term)))

        variables = [name for name in order if any(name in term for term in terms)]
        group = groups[0] if groups else None

        if response in variables or response == group:
            raise ValueError("The response '%s' cannot also appear on the right-hand side of the formula." % response)
        if group is not None and group in variables:
            raise ValueError("The grouping variable '%s' cannot also be a fixed effect." % group)
        if not terms and group is None:
            raise ValueError("The formula has neither covariates nor a random effect. Use the class fit for a phase-type distribution without covariates.")

        self.response = response
        self.terms = terms
        self.variables = variables
        self.group = group

        return None


class _regdesign:
    """
    Builds the design of a phase-type regression from a pandas DataFrame: the
    response, the design matrix of the fixed effects, and the group codes of a
    random effect. A column with the 'category' dtype is coded by treatment
    dummies against its first category with observations; a numeric column is
    continuous. In an interaction of a continuous x and a categorical w, the
    main effects may be left out: in 'x + x:w' w has treatment dummies, in
    'x:w' and 'w + x:w' one column per level, as in R. Every column is centred
    and the continuous ones are scaled; backtransform() returns to the scale
    of the data. Censoring limits are read from two columns, lower and
    upper, as kind and limits of each row.

    Warning: This class reports unusable data by raising ValueError with a
    message meant for the user; the caller prints it.
    """

    def __init__(
        self,
        data,
        response: str,
        terms: list,
        group: str = None,
        censoring: tuple = None,
    ) -> None:
        """
        Builds the design.

        Args:
            data (DataFrame): The data.
            response (str): Name of the response variable.
            terms (list): The fixed effects, tuples of variable names.
            group (str, optional): Name of the grouping variable.
            censoring (tuple, optional): Names of the columns with the lower
                and the upper censoring limits.

        Raises:
            ValueError: If the data cannot be used.
        """

        self.data = data
        self.response = response
        self.terms = [tuple(term) for term in terms]
        self.group = group
        self.censoringcolumns = None if censoring is None else tuple(censoring)

        self.__build()

        return None

    # ----------------------------------------------------------------------
    #   PUBLIC METHODS (used by fitreg)
    # ----------------------------------------------------------------------

    def backtransform(self, betatilde: np.array) -> tuple:
        """
        Returns the coefficients on the scale of the data, and the shift delta that
        the centring moved into the scale of the generator.

        Args:
            betatilde (ndarray): Coefficients on the centred and scaled design.

        Returns:
            tuple: The coefficients (ndarray) and delta (float).
        """

        betatilde = np.asarray(betatilde, dtype=float)
        if betatilde.size == 0:
            return betatilde.copy(), 0.0

        beta = betatilde / self.scale
        delta = float(np.sum(betatilde * self.center / self.scale))

        return beta, delta

    def row(self, covariates: dict = None) -> np.array:
        """
        Returns the design row for one set of covariate values; the baseline
        without covariates.

        Args:
            covariates (dict, optional): Value of each variable, a level for a
                categorical variable.

        Returns:
            ndarray: The design row.

        Raises:
            ValueError: If a variable is missing or a level is unknown.
        """

        if covariates is None:
            return np.zeros(self.d)

        if not isinstance(covariates, dict):
            raise ValueError("The covariates must be given as a dictionary from variable name to value.")

        unknown = [name for name in covariates if name not in self.info]
        if unknown:
            raise ValueError("The model has no variable named '%s'." % unknown[0])

        bases, fullbases = {}, {}
        for name, info in self.info.items():
            if name not in covariates:
                raise ValueError("A value is needed for every variable of the model, and '%s' has none." % name)
            value = covariates[name]
            if info["kind"] == "numeric":
                try:
                    bases[name] = np.array([[float(value)]])
                except (TypeError, ValueError):
                    raise ValueError("The variable '%s' is numeric, so its value must be a number." % name)
            else:
                if not any(value == level for level in info["levels"]):
                    raise ValueError("'%s' is not a level of '%s'; the levels are %s." % (value, name, info["levels"]))
                fullbases[name] = np.array([[float(value == level) for level in info["levels"]]])
                bases[name] = fullbases[name][:, 1:]

        return self.__products(bases, fullbases)[0]

    # ----------------------------------------------------------------------
    #   PRIVATE METHODS
    # ----------------------------------------------------------------------

    def __column(self, name: str):
        """
        Returns a column of the data.

        Args:
            name (str): Name of the variable.

        Returns:
            Series: The column.

        Raises:
            ValueError: If the variable is missing.
        """

        if name not in self.data.columns:
            raise ValueError("The data have no variable named '%s'." % name)

        column = self.data[name]
        if getattr(column, "ndim", 1) != 1:
            raise ValueError("The name '%s' refers to more than one column of the data." % name)

        return column

    @staticmethod
    def __iscategorical(column) -> bool:
        """
        Returns whether a column has the 'category' dtype.

        Args:
            column (Series): The column.

        Returns:
            bool: True for a categorical column.
        """

        return str(column.dtype) == "category"

    def __censoring(self) -> None:
        """
        Reads the lower and upper censoring limits and sets the kind of each
        row: both missing exact, only lower right-censored (Y > lower), only
        upper left-censored (Y <= upper), both interval-censored
        (lower < Y <= upper). An infinite upper limit counts as missing.

        Args:
            None

        Returns:
            None

        Raises:
            ValueError: If the limits cannot be used.
        """

        names = self.censoringcolumns
        used = {self.response, self.group} | {name for term in self.terms for name in term}
        for name in names:
            if name in used:
                raise ValueError("The censoring column '%s' is also used in the formula." % name)

        limits = []
        for name in names:
            column = self.__column(name)
            kind = getattr(column.dtype, "kind", "O")
            if self.__iscategorical(column) or kind not in "iuf":
                raise ValueError("The censoring column '%s' must be numeric, with missing values (NaN) where there is no limit." % name)
            limits.append(np.asarray(column, dtype=float).copy())
        lower, upper = limits
        upper[upper == np.inf] = np.nan

        given = ~np.isnan(lower)
        if not np.all(np.isfinite(lower[given])) or np.any(lower[given] < 0.0):
            raise ValueError("The lower censoring limits ('%s') must be finite and zero or positive." % names[0])
        if np.any(upper[~np.isnan(upper)] <= 0.0):
            raise ValueError("The upper censoring limits ('%s') must be positive." % names[1])

        both = given & ~np.isnan(upper)
        if np.any(lower[both] >= upper[both]):
            raise ValueError("In an interval-censored row the lower limit ('%s') must be smaller than the upper limit ('%s')." % names)

        self.kind = np.where(given, np.where(np.isnan(upper), 1, 3), np.where(np.isnan(upper), 0, 2))
        self.limits = np.column_stack([lower, upper])

        return None

    def __response(self, column) -> np.array:
        """
        Returns the response with censoring: needed and finite in exact rows,
        ignored (NaN) in censored rows.

        Args:
            column (Series): The response column.

        Returns:
            ndarray: The response, NaN in censored rows.

        Raises:
            ValueError: If the response is not numeric or missing in an exact row.
        """

        kind = getattr(column.dtype, "kind", "O")
        if kind not in "iuf":
            raise ValueError("The response '%s' has dtype '%s' and must be numeric." % (self.response, column.dtype))

        values = np.asarray(column, dtype=float).copy()
        exact = self.kind == 0
        if not np.all(np.isfinite(values[exact])):
            raise ValueError("The response '%s' has missing or infinite values in rows that are not censored." % self.response)
        values[~exact] = np.nan

        return values

    def __numeric(self, name: str, column) -> np.array:
        """
        Returns a numeric column as floating point.

        Args:
            name (str): Name of the variable.
            column (Series): The column.

        Returns:
            ndarray: The values.

        Raises:
            ValueError: If the column is not numeric or has missing values.
        """

        kind = getattr(column.dtype, "kind", "O")
        if kind not in "iuf":
            raise ValueError("The variable '%s' has dtype '%s'. A continuous variable must be numeric, and a categorical one must have the 'category' dtype, e.g. data['%s'] = data['%s'].astype('category')." % (name, column.dtype, name, name))

        values = np.asarray(column, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("The variable '%s' has missing or infinite values." % name)

        return values

    def __categorical(self, name: str, column) -> tuple:
        """
        Returns the levels of a categorical column with observations, and the code
        of each row.

        Args:
            name (str): Name of the variable.
            column (Series): The column.

        Returns:
            tuple: The levels (list) and the codes (ndarray of int).

        Raises:
            ValueError: If the column has missing values.
        """

        codes = np.asarray(column.cat.codes, dtype=int)
        if np.any(codes < 0):
            raise ValueError("The variable '%s' has missing values." % name)

        categories = list(column.cat.categories)
        present = np.unique(codes)
        levels = [categories[code] for code in present]
        recode = np.full(len(categories), -1, dtype=int)
        recode[present] = np.arange(present.size)

        return levels, recode[codes]

    def __products(self, bases: dict, fullbases: dict) -> np.array:
        """
        Returns the design matrix: for every effect, the products of the columns of
        its variables.

        Args:
            bases (dict): The columns of each variable, treatment dummies for a
                categorical one.
            fullbases (dict): One column per level of each categorical variable.

        Returns:
            ndarray: The design matrix.
        """

        blocks = []
        for term in self.terms:
            block = None
            for name in term:
                other = fullbases[name] if self.coding[term][name] == "full" else bases[name]
                block = other if block is None else (block[:, :, None] * other[:, None, :]).reshape(block.shape[0], -1)
            blocks.append(block)

        if not blocks:
            first = next(iter(bases.values())) if bases else np.zeros((self.n, 0))
            return np.zeros((first.shape[0], 0))

        return np.column_stack(blocks)

    def __codings(self) -> dict:
        """
        Returns how each variable of each effect is coded: a categorical
        variable by treatment dummies ("contrast") if the effect without it is
        in the model, else by one column per level ("full"), as in R; a
        continuous variable by its values ("numeric"). Only an interaction of
        a continuous and a categorical variable may leave out a main effect.

        Args:
            None

        Returns:
            dict: For every effect, the coding of each of its variables.

        Raises:
            ValueError: If an interaction leaves out a main effect it needs.
        """

        present = {frozenset(term) for term in self.terms}
        coding = {}
        for term in self.terms:
            kinds = [self.info[name]["kind"] for name in term]
            missing = [name for name in term if len(term) > 1 and frozenset(term) - {name} not in present]
            if missing and not (len(term) == 2 and sorted(kinds) == ["categorical", "numeric"]):
                lacking = [other for other in term if frozenset([other]) not in present]
                raise ValueError("The interaction '%s' needs its lower-order effect '%s' in the formula as well; '%s' adds them all. "
                                 "Only an interaction of a continuous and a categorical variable may leave out main effects (e.g. 'x + x:w' or 'x:w')."
                                 % (":".join(term), lacking[0] if lacking else ":".join(missing), "*".join(term)))
            coding[term] = {}
            for name, kind in zip(term, kinds):
                if kind == "numeric":
                    coding[term][name] = "numeric"
                elif len(term) == 1 or frozenset(term) - {name} in present:
                    coding[term][name] = "contrast"
                else:
                    coding[term][name] = "full"

        return coding

    def __build(self) -> None:
        """
        Reads the response, variables and groups, and builds the centred design.

        Args:
            None

        Raises:
            ValueError: If the data cannot be used.
        """

        # the response
        self.n = None
        column = self.__column(self.response)
        if self.__iscategorical(column):
            raise ValueError("The response '%s' must be numeric, not categorical." % self.response)
        # the censoring of each row: kind 0 exact, 1 right, 2 left, 3 interval,
        # and the limits [lower, upper], NaN where there is none
        self.kind = None
        self.limits = None
        if self.censoringcolumns is None:
            self.y = self.__numeric(self.response, column)
        else:
            self.__censoring()
            self.y = self.__response(column)
        self.n = self.y.size

        if self.n < 2:
            raise ValueError("At least two observations are needed.")
        exact = np.isfinite(self.y)
        if np.any(exact) and not np.min(self.y[exact]) >= 0.0:
            raise ValueError("Every value of the response must be zero or positive.")

        # the columns of each variable, and their names
        self.info = {}
        bases, fullbases = {}, {}
        labels, fulllabels = {}, {}
        variables = []
        for term in self.terms:
            for name in term:
                if name not in variables:
                    variables.append(name)

        for name in variables:
            column = self.__column(name)
            if self.__iscategorical(column):
                levels, codes = self.__categorical(name, column)
                if len(levels) < 2:
                    raise ValueError("The categorical variable '%s' has only one level with observations." % name)
                fullbases[name] = (codes[:, None] == np.arange(len(levels))[None, :]).astype(float)
                bases[name] = fullbases[name][:, 1:]
                fulllabels[name] = ["%s[%s]" % (name, level) for level in levels]
                labels[name] = fulllabels[name][1:]
                self.info[name] = dict(kind="categorical", levels=levels, reference=levels[0])
            else:
                values = self.__numeric(name, column)
                if np.std(values) == 0.0:
                    raise ValueError("The variable '%s' is constant." % name)
                bases[name] = values[:, None]
                labels[name] = [name]
                self.info[name] = dict(kind="numeric")

        # how each variable of each effect is coded
        self.coding = self.__codings()

        # the design matrix and the names of its columns
        self.X = self.__products(bases, fullbases) if self.terms else np.zeros((self.n, 0))
        self.columns = []
        self.termcolumns = {}
        scaled = []
        for term in self.terms:
            names = None
            for name in term:
                own = fulllabels[name] if self.coding[term][name] == "full" else labels[name]
                names = own if names is None else [a + ":" + b for a in names for b in own]
            self.termcolumns[":".join(term)] = list(range(len(self.columns), len(self.columns) + len(names)))
            self.columns.extend(names)
            continuous = any(self.info[name]["kind"] == "numeric" for name in term)
            scaled.extend([continuous] * len(names))

        self.d = len(self.columns)
        if self.d:
            spread = np.ptp(self.X, axis=0)
            if np.any(spread == 0.0):
                empty = self.columns[int(np.flatnonzero(spread == 0.0)[0])]
                raise ValueError("The column '%s' of the design matrix is constant, e.g. an interaction of categorical variables with an empty combination of levels." % empty)

        self.center = np.mean(self.X, axis=0) if self.d else np.zeros(0)
        self.scale = np.ones(self.d)
        scaled = np.array(scaled, dtype=bool)
        if self.d and np.any(scaled):
            self.scale[scaled] = np.std(self.X[:, scaled], axis=0)
        self.Xtilde = (self.X - self.center) / self.scale if self.d else self.X.copy()

        if self.d >= self.n:
            raise ValueError("There must be more observations than coefficients.")
        if self.d and np.linalg.matrix_rank(self.Xtilde) < self.d:
            raise ValueError("The design matrix is rank deficient: Some columns are linear combinations of others.")

        # the groups of the random effect
        self.groupcodes = None
        self.grouplabels = None
        if self.group is not None:
            column = self.__column(self.group)
            if not self.__iscategorical(column):
                raise ValueError("The grouping variable '%s' must have the 'category' dtype, e.g. data['%s'] = data['%s'].astype('category')." % (self.group, self.group, self.group))
            levels, codes = self.__categorical(self.group, column)
            if len(levels) < 2:
                raise ValueError("The grouping variable '%s' has only one level with observations." % self.group)
            self.grouplabels = levels
            self.groupcodes = codes

        return None
