'''
UNIT TEST FOR THE _regdesign CLASS (the design matrices of fitreg).

Sub-tests:
    Case 1: Main effects and interactions up to four-way of continuous and
            categorical variables equal hand-built products, names included.
    Case 2: Centring and scaling: every column centred, continuous ones scaled;
            the linear predictor is unchanged by backtransform().
    Case 3: Prediction rows equal the design rows; bad covariates refused.
    Case 4: Levels and groups: the reference is the first category with
            observations, unused categories dropped, groups recoded 0..J-1.
    Case 5: Unusable data is refused with a message.
    Case 6: Column names with spaces and special symbols (backticks).
    Case 7: Censoring columns: kind and limits of each row; refusals.
    Case 8: Numeric variables mixed with categorical ones of 2, 3 and 5 levels,
            up to four-way: every column the hand-built product of its parts,
            the number of columns of each effect the product of the parts'
            column counts, and interactions written in any order.
'''

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from phasedist.regdesign import _regformula, _regdesign


TOL = 1e-12
SEED = 0
N = 120

rng = np.random.default_rng(SEED)
x = rng.normal(1.0, 2.0, N)
z = rng.uniform(-1.0, 3.0, N)
g = rng.choice(["a", "b", "c"], N)
h = rng.choice(["p", "q"], N)
k = rng.choice(["r", "s", "t"], N)
DATA = pd.DataFrame({"y": rng.exponential(size=N), "x": x, "z": z,
                     "g": pd.Categorical(g), "h": pd.Categorical(h), "k": pd.Categorical(k),
                     "u": pd.Categorical(rng.integers(0, 6, N))})
if set(g) != {"a", "b", "c"} or set(h) != {"p", "q"} or len(set(zip(g, h))) != 6 or len(set(zip(g, k))) != 9:
    sys.exit("Validation test failed at initialization: a level or a cell of g and h is empty.")


def design(formula, data=DATA, censoring=None):
    """Builds the design of a formula."""
    parsed = _regformula(formula)
    return _regdesign(data, parsed.response, parsed.terms, parsed.group, censoring)


def refused(formula, data, censoring=None):
    """Returns whether the design refuses the data with a message."""
    try:
        design(formula, data, censoring)
    except ValueError as error:
        return bool(str(error))
    return False


# ------------------------------------------------------------------
# CASE 1: Products of main effects and interactions up to four-way
# ------------------------------------------------------------------

gb, gc, hq = (g == "b") * 1.0, (g == "c") * 1.0, (h == "q") * 1.0
built = design("y ~ x*z*g*h")
hand = {
    "x": x, "z": z, "g[b]": gb, "g[c]": gc, "h[q]": hq,
    "x:z": x * z, "x:g[b]": x * gb, "x:g[c]": x * gc, "x:h[q]": x * hq,
    "z:g[b]": z * gb, "z:g[c]": z * gc, "z:h[q]": z * hq, "g[b]:h[q]": gb * hq, "g[c]:h[q]": gc * hq,
    "x:z:g[b]": x * z * gb, "x:z:g[c]": x * z * gc, "x:z:h[q]": x * z * hq,
    "x:g[b]:h[q]": x * gb * hq, "x:g[c]:h[q]": x * gc * hq, "z:g[b]:h[q]": z * gb * hq, "z:g[c]:h[q]": z * gc * hq,
    "x:z:g[b]:h[q]": x * z * gb * hq, "x:z:g[c]:h[q]": x * z * gc * hq,
}
if built.columns != list(hand):
    sys.exit("Validation test failed at case 1: the columns of 'x*z*g*h' are %s, not %s." % (built.columns, list(hand)))
if np.max(np.abs(built.X - np.column_stack(list(hand.values())))) > TOL:
    sys.exit("Validation test failed at case 1: the design of 'x*z*g*h' differs from the hand-built products.")
# two categorical variables with two dummies each: the columns run over the
# levels of the second fastest
ks, kt = (k == "s") * 1.0, (k == "t") * 1.0
crossed = design("y ~ g*k")
hand2 = {"g[b]": gb, "g[c]": gc, "k[s]": ks, "k[t]": kt,
         "g[b]:k[s]": gb * ks, "g[b]:k[t]": gb * kt, "g[c]:k[s]": gc * ks, "g[c]:k[t]": gc * kt}
if crossed.columns != list(hand2) or np.max(np.abs(crossed.X - np.column_stack(list(hand2.values())))) > TOL:
    sys.exit("Validation test failed at case 1: the design of 'g*k' differs from the hand-built products.")
expectedcolumns = {"x": [0], "g": [2, 3], "x:z:g": [14, 15], "x:z:g:h": [21, 22]}
for effect, index in expectedcolumns.items():
    if built.termcolumns[effect] != index:
        sys.exit("Validation test failed at case 1: the effect '%s' has the columns %s, not %s." % (effect, built.termcolumns[effect], index))


# ------------------------------------------------------------------
# CASE 2: Centring, scaling and the back-transformation
# ------------------------------------------------------------------

continuous = np.array([any(part.split("[")[0] in ("x", "z") for part in name.split(":")) for name in built.columns])
scale = np.where(continuous, built.X.std(axis=0), 1.0)
if np.max(np.abs(built.center - built.X.mean(axis=0))) > TOL or np.max(np.abs(built.scale - scale)) > TOL:
    sys.exit("Validation test failed at case 2: the centres or scales are not the column means and the standard deviations of the continuous columns.")
if np.max(np.abs(built.Xtilde - (built.X - built.X.mean(axis=0)) / scale)) > TOL:
    sys.exit("Validation test failed at case 2: the centred design is not (X - mean) / scale.")
betatilde = rng.normal(size=built.d)
beta, delta = built.backtransform(betatilde)
if np.max(np.abs((built.X @ beta - delta) - built.Xtilde @ betatilde)) > 1e-10:
    sys.exit("Validation test failed at case 2: the back-transformed coefficients change the linear predictor.")


# ------------------------------------------------------------------
# CASE 3: Prediction rows
# ------------------------------------------------------------------

for i in (0, 7, 51, N - 1):
    row = built.row({"x": x[i], "z": z[i], "g": g[i], "h": h[i]})
    if np.max(np.abs(row - built.X[i])) > TOL:
        sys.exit("Validation test failed at case 3: the prediction row of observation %d differs from its design row." % i)
if np.any(design("y ~ x + g").row() != 0.0):
    sys.exit("Validation test failed at case 3: the baseline row is not zero.")
for covariates in ({"x": 1.0, "z": 0.0, "g": "d", "h": "p"}, {"x": 1.0, "z": 0.0, "g": "a"},
                   {"x": 1.0, "z": 0.0, "g": "a", "h": "p", "w": 2.0}):
    try:
        built.row(covariates)
        sys.exit("Validation test failed at case 3: the covariates %s were accepted." % covariates)
    except ValueError as error:
        if not str(error):
            sys.exit("Validation test failed at case 3: the covariates %s were refused without a message." % covariates)


# ------------------------------------------------------------------
# CASE 4: Levels and groups
# ------------------------------------------------------------------

# the reference is the first category with observations; unused ones are dropped
reordered = DATA.assign(g=pd.Categorical(g, categories=["unused", "c", "a", "b", "other"]))
levels = design("y ~ g", reordered)
if levels.columns != ["g[a]", "g[b]"] or levels.info["g"]["reference"] != "c":
    sys.exit("Validation test failed at case 4: with categories (unused, c, a, b, other) the columns are %s." % levels.columns)
grouped = design("y ~ x + (1|u)")
if grouped.group != "u" or sorted(set(grouped.groupcodes)) != list(range(6)) or \
        not np.array_equal(grouped.grouplabels, sorted(set(DATA["u"]))):
    sys.exit("Validation test failed at case 4: the groups are not recoded 0..J-1 with their labels.")


# ------------------------------------------------------------------
# CASE 5: Refusals
# ------------------------------------------------------------------

bad = {
    "a text column": ("y ~ w", DATA.assign(w=g)),
    "a boolean column": ("y ~ w", DATA.assign(w=x > 0)),
    "a missing value": ("y ~ x", DATA.assign(x=np.where(np.arange(N) == 3, np.nan, x))),
    "a constant covariate": ("y ~ w", DATA.assign(w=1.0)),
    "a single level": ("y ~ w", DATA.assign(w=pd.Categorical(["a"] * N))),
    "an empty cell": ("y ~ g*k", DATA.assign(k=pd.Categorical(np.where(g == "a", "r", "s")))),
    "a rank-deficient design": ("y ~ x + w", DATA.assign(w=2.0 * x + 1.0)),
    "a negative response": ("y ~ x", DATA.assign(y=-DATA["y"])),
    "a categorical response": ("y ~ x", DATA.assign(y=pd.Categorical(g))),
    "a numeric grouping variable": ("y ~ x + (1|w)", DATA.assign(w=rng.integers(0, 5, N))),
    "an unknown variable": ("y ~ v", DATA),
}
for name, (formula, data) in bad.items():
    if not refused(formula, data):
        sys.exit("Validation test failed at case 5: %s was not refused with a message." % name)


# ------------------------------------------------------------------
# CASE 6: Names with spaces and special symbols
# ------------------------------------------------------------------

names = {"x": "body mass index (kg/m^2)", "z": "x-1", "g": "sex: m/f", "u": "hospital id", "y": "time to event"}
renamed = DATA.rename(columns=names)
quoted = design("`time to event` ~ `body mass index (kg/m^2)`*`x-1` + `sex: m/f` + (1|`hospital id`)", renamed)
plain = design("y ~ x*z + g + (1|u)")
expected = ["body mass index (kg/m^2)", "x-1", "sex: m/f[b]", "sex: m/f[c]", "body mass index (kg/m^2):x-1"]
if quoted.columns != expected or np.max(np.abs(quoted.X - plain.X)) > TOL or \
        not np.array_equal(quoted.groupcodes, plain.groupcodes) or not np.array_equal(quoted.y, plain.y):
    sys.exit("Validation test failed at case 6: the design with special names is %s, not that of the plain names." % quoted.columns)
row = quoted.row({"body mass index (kg/m^2)": 2.0, "x-1": 0.5, "sex: m/f": "c"})
if np.max(np.abs(row - np.array([2.0, 0.5, 0.0, 1.0, 1.0]))) > TOL:
    sys.exit("Validation test failed at case 6: the prediction row with special names is %s." % row)


# ------------------------------------------------------------------
# CASE 7: Censoring columns
# ------------------------------------------------------------------

lower = np.full(N, np.nan)
upper = np.full(N, np.nan)
kinds = np.arange(N) % 4                                  # exact, right, left, interval
lower[kinds == 1] = 2.0
upper[kinds == 2] = 0.5
lower[kinds == 3], upper[kinds == 3] = 0.5, 1.5
upper[np.flatnonzero(kinds == 1)[:3]] = np.inf            # an infinite upper limit is missing
censored = DATA.assign(lo=lower, hi=upper)              # the response of censored rows is ignored
built = design("y ~ x", censored, ("lo", "hi"))
if not np.array_equal(built.kind, kinds):
    sys.exit("Validation test failed at case 7: the kinds of the rows are wrong.")
expectedlimits = np.column_stack([lower, np.where(np.isinf(upper), np.nan, upper)])
if not np.array_equal(np.isnan(built.limits), np.isnan(expectedlimits)) or \
        np.any(built.limits[~np.isnan(expectedlimits)] != expectedlimits[~np.isnan(expectedlimits)]):
    sys.exit("Validation test failed at case 7: the censoring limits are wrong.")
if not np.all(np.isnan(built.y[kinds != 0])) or np.any(built.y[kinds == 0] != DATA["y"].to_numpy()[kinds == 0]):
    sys.exit("Validation test failed at case 7: the response is not NaN in the censored rows and the observation elsewhere.")
if design("y ~ x").kind is not None:
    sys.exit("Validation test failed at case 7: a design without censoring has kinds.")

badcensoring = {
    "a negative lower limit": (censored.assign(lo=np.where(kinds == 1, -1.0, lower)), ("lo", "hi")),
    "a lower limit above the upper": (censored.assign(lo=np.where(kinds == 3, 2.0, lower)), ("lo", "hi")),
    "an upper limit of zero": (censored.assign(hi=np.where(kinds == 2, 0.0, upper)), ("lo", "hi")),
    "a missing response in an exact row": (censored.assign(y=np.where(kinds == 0, np.nan, DATA["y"])), ("lo", "hi")),
    "a column of the formula": (censored, ("x", "hi")),
    "a categorical column": (censored.assign(lo=pd.Categorical(g)), ("lo", "hi")),
    "a missing column": (censored, ("lo", "upper")),
}
for name, (data, columns) in badcensoring.items():
    if not refused("y ~ x", data, columns):
        sys.exit("Validation test failed at case 7: %s was not refused with a message." % name)


# ------------------------------------------------------------------
# CASE 8: Numeric and categorical variables with 2, 3 and 5 levels
# ------------------------------------------------------------------

NMIX = 600
mixrng = np.random.default_rng([SEED, 8])
LEVELS = {"a": ["a1", "a2"], "b": ["b1", "b2", "b3"], "c": ["c1", "c2", "c3", "c4", "c5"]}
MIX = pd.DataFrame({"y": mixrng.exponential(size=NMIX), "x": mixrng.normal(size=NMIX), "z": mixrng.uniform(size=NMIX)})
for name, levels in LEVELS.items():
    MIX[name] = pd.Categorical(mixrng.choice(levels, NMIX), categories=levels)
if len(set(zip(MIX["a"], MIX["b"], MIX["c"]))) != 30:
    sys.exit("Validation test failed at initialization: a cell of a, b and c is empty.")


def parts(name):
    """The columns of one variable: itself if numeric, else its treatment dummies."""
    if name in LEVELS:
        return {"%s[%s]" % (name, level): (MIX[name].to_numpy() == level) * 1.0 for level in LEVELS[name][1:]}
    return {name: MIX[name].to_numpy()}


def handbuilt(effects):
    """Columns of the effects, each the products of its variables' columns, the last variable fastest."""
    columns = {}
    for effect in effects:
        block = {"": np.ones(NMIX)}
        for name in effect:
            block = {(key + ":" if key else "") + part: value * column
                     for key, value in block.items() for part, column in parts(name).items()}
        columns.update(block)
    return columns


cases = {
    "y ~ x*a + z*b": [("x",), ("a",), ("z",), ("b",), ("x", "a"), ("z", "b")],
    "y ~ x + b + c + x:b + b:c + c:x + b:c:x": [("x",), ("b",), ("c",), ("x", "b"), ("b", "c"), ("x", "c"), ("x", "b", "c")],
    "y ~ x*a*b*c": [term for size in range(1, 5) for term in itertools.combinations(["x", "a", "b", "c"], size)],
    "y ~ z*c + x:a + x + a + b": [("z",), ("c",), ("x",), ("a",), ("b",), ("z", "c"), ("x", "a")],
}
for formula, effects in cases.items():
    built = design(formula, MIX)
    hand = handbuilt(effects)
    if built.columns != list(hand) or np.max(np.abs(built.X - np.column_stack(list(hand.values())))) > TOL:
        sys.exit("Validation test failed at case 8: the design of '%s' differs from the hand-built products." % formula)
    for effect in effects:
        count = int(np.prod([len(parts(name)) for name in effect]))
        if len(built.termcolumns[":".join(effect)]) != count:
            sys.exit("Validation test failed at case 8: the effect '%s' of '%s' has %d columns, not %d."
                     % (":".join(effect), formula, len(built.termcolumns[":".join(effect)]), count))
if design("y ~ x*a*b*c", MIX).d != 59:
    sys.exit("Validation test failed at case 8: 'x*a*b*c' does not have 1 + 1 + 2 + 4 + 21 + 22 + 8 = 59 columns.")

# one column checked by hand: x times the indicators of b3 and c5
built = design("y ~ x*a*b*c", MIX)
column = built.columns.index("x:b[b3]:c[c5]")
if np.max(np.abs(built.X[:, column] - MIX["x"].to_numpy() * (MIX["b"] == "b3") * (MIX["c"] == "c5"))) > TOL:
    sys.exit("Validation test failed at case 8: the column x:b[b3]:c[c5] is not x times the two indicators.")

# interactions written in another order give the same columns, renamed
written = design("y ~ c:b:x + b:c + x + c + b + x:c + b:x", MIX)
reference = design("y ~ x + b + c + x:b + b:c + c:x + b:c:x", MIX)
if {frozenset(n.split(":")) for n in written.columns} != {frozenset(n.split(":")) for n in reference.columns}:
    sys.exit("Validation test failed at case 8: the interactions written in another order give other columns.")
order = [written.columns.index(next(m for m in written.columns if frozenset(m.split(":")) == frozenset(n.split(":"))))
         for n in reference.columns]
if np.max(np.abs(written.X[:, order] - reference.X)) > TOL:
    sys.exit("Validation test failed at case 8: the interactions written in another order give other values.")


print("_regdesign: All tests passed.")
