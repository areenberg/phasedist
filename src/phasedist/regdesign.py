import itertools
import re

import numpy as np


# a variable name as R and Python both accept it: a letter or an underscore,
# then letters, digits, underscores or dots
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


class _regformula:
    """
    Parses a regression formula in the notation of R's lm and lme4 packages,
    restricted to what fitreg supports:

        'y ~ x + z'            main effects of x and z
        'y ~ x * z'            main effects and their interaction, x + z + x:z
        'y ~ x + z + x:z'      the same, written out
        'y ~ x + z + (1|g)'    plus a Gaussian random intercept per level of g
        'y ~ 1 + (1|g)'        a random intercept and no fixed effects

    A '*' term expands to every main effect and interaction of its variables,
    as in R. An explicit intercept '1' is accepted and ignored, the intercept
    being the scale of the phase-type generator. Repeated terms are kept once.
    Every interaction must come with all its lower-order terms (x:z needs x
    and z), which keeps the coding of categorical variables unambiguous.
    Transformations, random slopes and more than one random intercept are
    refused with a message saying so.

    The terms are returned as tuples of variable names, main effects first,
    then two-way interactions, and so on.

    Warning: This class reports a malformed formula by raising ValueError
    with a message meant for the user; the caller prints it.
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
        Splits the formula into response, fixed-effect terms and grouping
        variable.

        Args:
            None

        Raises:
            ValueError: If the formula is malformed or not supported.
        """

        if not isinstance(self.formula, str):
            raise ValueError("The formula must be a string, e.g. 'y ~ x + z + (1|g)'.")

        sides = self.formula.split("~")
        if len(sides) != 2:
            raise ValueError("The formula must contain exactly one '~', e.g. 'y ~ x + z'.")

        response = sides[0].strip()
        if not _NAME.match(response):
            raise ValueError("The left-hand side of the formula must be the name of the response variable, not '%s'." % response)

        righthand = sides[1]

        # the random terms are the parenthesized ones
        groups = []
        for match in re.finditer(r"\(([^()]*)\)", righthand):
            inner = match.group(1)
            if "|" not in inner:
                raise ValueError("Parentheses are only used for random intercepts such as (1|g). Transformations such as log(x) and functions such as C(x) are not supported; categorical variables are marked by giving them the 'category' dtype in the DataFrame.")
            if "||" in inner:
                raise ValueError("The '||' notation is not supported; only random intercepts such as (1|g) are.")
            left, right = inner.split("|", 1)
            if left.strip() != "1":
                raise ValueError("Only random intercepts such as (1|g) are supported; '(%s)' asks for random slopes." % inner.strip())
            group = right.strip()
            if not _NAME.match(group):
                raise ValueError("The grouping variable of a random intercept must be a single variable name, not '%s'." % group)
            groups.append(group)

        fixedpart = re.sub(r"\([^()]*\)", " ", righthand)
        if "(" in fixedpart or ")" in fixedpart:
            raise ValueError("The formula has unbalanced or nested parentheses.")

        if len(groups) > 1:
            raise ValueError("fitreg supports one random intercept, but the formula has %d: %s. Keep one of them."
                             % (len(groups), ", ".join("(1|%s)" % group for group in groups)))

        # the fixed-effect terms, each a tuple of variable names
        order = []          # variables in order of first appearance
        found = []          # terms in order of appearance, before sorting
        for token in fixedpart.split("+"):
            term = token.strip()
            if term == "" or term == "1":
                continue
            if term == "0" or "-" in term:
                raise ValueError("The intercept cannot be removed: in a phase-type regression it is the scale of the phase-type generator.")
            if "*" in term and ":" in term:
                raise ValueError("'%s' mixes '*' and ':'; write such terms out with '+', e.g. x + z + x:z." % term)

            separator = "*" if "*" in term else ":"
            names = [name.strip() for name in term.split(separator)]
            for name in names:
                if not _NAME.match(name):
                    raise ValueError("'%s' is not a variable name. Terms are variable names joined by '+', with ':' or '*' for interactions." % name)
                if name not in order:
                    order.append(name)
            if len(set(names)) != len(names):
                raise ValueError("The term '%s' names a variable more than once." % term)

            if separator == "*":
                # every main effect and interaction of the variables
                for size in range(1, len(names) + 1):
                    for subset in itertools.combinations(names, size):
                        found.append(subset)
            else:
                found.append(tuple(names))

        # repeated terms are kept once, written in the order the variables
        # first appear, and sorted by degree
        terms = []
        seen = set()
        for term in found:
            canonical = tuple(sorted(term, key=order.index))
            if frozenset(canonical) not in seen:
                seen.add(frozenset(canonical))
                terms.append(canonical)
        terms.sort(key=len)

        # every interaction needs all its lower-order terms
        for term in terms:
            if len(term) < 2:
                continue
            for size in range(1, len(term)):
                for subset in itertools.combinations(term, size):
                    if frozenset(subset) not in seen:
                        raise ValueError("The interaction '%s' needs its lower-order term '%s' in the formula as well; '%s' adds them all." % (":".join(term), ":".join(subset), "*".join(term)))

        variables = [name for name in order if any(name in term for term in terms)]
        group = groups[0] if groups else None

        if response in variables or response == group:
            raise ValueError("The response '%s' cannot also appear on the right-hand side of the formula." % response)
        if group is not None and group in variables:
            raise ValueError("The grouping variable '%s' cannot also be a fixed effect." % group)
        if not terms and group is None:
            raise ValueError("The formula has neither covariates nor a random intercept. Use the class fit for a phase-type distribution without covariates.")

        self.response = response
        self.terms = terms
        self.variables = variables
        self.group = group

        return None


class _regdesign:
    """
    Builds the design of a phase-type regression from a pandas DataFrame: the
    response, the fixed-effect design matrix, and the group codes of a random
    intercept.

    The type of each variable is read from the DataFrame. A column with the
    'category' dtype is categorical and is coded by treatment dummies, the
    first category with observations being the reference level (reorder the
    categories to change it). A numeric column is continuous. Any other
    column (text, boolean, dates) is refused with a message saying how to
    convert it. The grouping variable of a random intercept must be
    categorical. An interaction's columns are the products of the columns of
    its variables.

    For the fitting, every column of the design matrix is centred, the dummies
    included, and the columns with a continuous variable in them are scaled
    to unit standard deviation. Centring every column makes the conditional
    standard errors of the coefficients valid (see the design note, Section
    9). backtransform() returns the estimates to the scale of the data.

    Warning: This class reports unusable data by raising ValueError with a
    message meant for the user; the caller prints it.
    """

    def __init__(
        self,
        data,
        response: str,
        terms: list,
        group: str = None,
    ) -> None:
        """
        Builds the design.

        Args:
            data (DataFrame): The data, one column per variable.
            response (str): Name of the response variable.
            terms (list): The fixed-effect terms, tuples of variable names.
            group (str, optional): Name of the grouping variable of a random
                intercept.

        Raises:
            ValueError: If the data cannot be used.
        """

        self.data = data
        self.response = response
        self.terms = [tuple(term) for term in terms]
        self.group = group

        self.__build()

        return None

    # ----------------------------------------------------------------------
    #   PUBLIC METHODS (used by fitreg)
    # ----------------------------------------------------------------------

    def backtransform(self, betatilde: np.array) -> tuple:
        """
        Returns the coefficients on the scale of the data, and the shift that
        the centring moved into the scale of the generator: with
        eta~ = sum_k beta~_k (x_k - c_k)/s_k = x'beta - delta, the baseline
        generator is exp(-delta) times the fitted one.

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
        Returns the design row, on the scale of the data, for one set of
        covariate values. Without covariates the row is the baseline: zero for
        the continuous variables and the reference level for the categorical
        ones.

        Args:
            covariates (dict, optional): Value of each variable, the level for
                a categorical variable.

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

        bases = {}
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
                bases[name] = np.array([[float(value == level) for level in info["levels"][1:]]])

        return self.__products(bases)[0]

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
        Returns whether a column has pandas' 'category' dtype.

        Args:
            column (Series): The column.

        Returns:
            bool: True for a categorical column.
        """

        return str(column.dtype) == "category"

    def __numeric(self, name: str, column) -> np.array:
        """
        Returns a numeric column as floating point.

        Args:
            name (str): Name of the variable.
            column (Series): The column.

        Returns:
            ndarray: The values as floats.

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
        Returns the levels of a categorical column (its categories in order,
        those without observations left out) and the code of each row.

        Args:
            name (str): Name of the variable.
            column (Series): The column.

        Returns:
            tuple: The levels (list) and the codes (ndarray of int).

        Raises:
            ValueError: If the column has missing values or a single level.
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

    def __products(self, bases: dict) -> np.array:
        """
        Returns the design matrix built from the columns of each variable:
        for every term, the products of the columns of its variables, the
        first variable's columns varying slowest.

        Args:
            bases (dict): The columns of each variable, (rows, columns).

        Returns:
            ndarray: The design matrix.
        """

        blocks = []
        for term in self.terms:
            block = bases[term[0]]
            for name in term[1:]:
                other = bases[name]
                block = (block[:, :, None] * other[:, None, :]).reshape(block.shape[0], -1)
            blocks.append(block)

        if not blocks:
            first = next(iter(bases.values())) if bases else np.zeros((self.n, 0))
            return np.zeros((first.shape[0], 0))

        return np.column_stack(blocks)

    def __build(self) -> None:
        """
        Reads the response, the variables and the groups from the data and
        builds the centred design matrix.

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
        self.y = self.__numeric(self.response, column)
        self.n = self.y.size

        if self.n < 2:
            raise ValueError("At least two observations are needed.")
        if not np.min(self.y) > 0.0:
            raise ValueError("Every value of the response must be positive.")

        # the columns of each variable, and their names
        self.info = {}
        bases = {}
        labels = {}
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
                bases[name] = (codes[:, None] == np.arange(1, len(levels))[None, :]).astype(float)
                labels[name] = ["%s[%s]" % (name, level) for level in levels[1:]]
                self.info[name] = dict(kind="categorical", levels=levels, reference=levels[0])
            else:
                values = self.__numeric(name, column)
                if np.std(values) == 0.0:
                    raise ValueError("The variable '%s' is constant." % name)
                bases[name] = values[:, None]
                labels[name] = [name]
                self.info[name] = dict(kind="numeric")

        # the design matrix and the names of its columns
        self.X = self.__products(bases) if self.terms else np.zeros((self.n, 0))
        self.columns = []
        self.termcolumns = {}
        scaled = []
        for term in self.terms:
            names = [labels[term[0]]]
            for name in term[1:]:
                names = [[a + ":" + b for a in names[0] for b in labels[name]]]
            self.termcolumns[":".join(term)] = list(range(len(self.columns), len(self.columns) + len(names[0])))
            self.columns.extend(names[0])
            continuous = any(self.info[name]["kind"] == "numeric" for name in term)
            scaled.extend([continuous] * len(names[0]))

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
            raise ValueError("The design matrix is rank deficient: some columns are linear combinations of others.")

        # the groups of the random intercept
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
