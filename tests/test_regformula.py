'''
UNIT TEST FOR THE _regformula CLASS (the formula parser of fitreg).

Sub-tests:
    Case 1: Main effects and interactions up to four-way: '*' expands to every
            lower-order effect, written-out formulas give the same effects,
            and an incomplete higher-order interaction is refused.
    Case 2: Spaces: None, many and mixed around '~', '+', ':', '*', '|' and
            parentheses give the same output (hand-written and random).
    Case 3: Long covariate names (up to 32 characters).
    Case 4: Names with spaces and special symbols in backticks, and names
            with letters of other alphabets; refusals that point to backticks.
    Case 5: Every effect once: 'x*y + z*w' and interactions written in any
            order pass; 'x*y + x*y*z', 'x + y + x*y' and repeated effects are
            refused, and fitreg prints an error and terminates.
'''

import contextlib
import io
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from phasedist.fitreg import fitreg
from phasedist.regdesign import _regformula


SEED = 0
NVARIANTS = 50       # random spacings per formula in case 2


def parsed(formula):
    """Returns (response, terms, group) of a formula."""
    result = _regformula(formula)
    return result.response, result.terms, result.group


def refused(formula, words=()):
    """Returns whether the formula is refused with a message containing the words."""
    try:
        _regformula(formula)
    except ValueError as error:
        return all(word in str(error) for word in words)
    return False


def expansion(names):
    """Every main effect and interaction of the names, by degree, as R orders them."""
    return [subset for size in range(1, len(names) + 1) for subset in itertools.combinations(names, size)]


# ------------------------------------------------------------------
# CASE 1: Main effects and interactions up to four-way
# ------------------------------------------------------------------

expected = {
    "y ~ a": ("y", [("a",)], None),
    "y ~ a + b + c + d": ("y", [("a",), ("b",), ("c",), ("d",)], None),
    "y ~ a*b": ("y", [("a",), ("b",), ("a", "b")], None),
    "y ~ a + b + a:b": ("y", [("a",), ("b",), ("a", "b")], None),
    "y ~ a*b*c": ("y", [("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c")], None),
    "y ~ a*b + c*d": ("y", [("a",), ("b",), ("c",), ("d",), ("a", "b"), ("c", "d")], None),
    "y ~ a*b*c + d": ("y", [("a",), ("b",), ("c",), ("d",), ("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c")], None),
    "y ~ a*b*c*d": ("y", expansion(["a", "b", "c", "d"]), None),
    "y ~ a*b*c*d + (1|g)": ("y", expansion(["a", "b", "c", "d"]), "g"),
    "y ~ 1 + a*b": ("y", [("a",), ("b",), ("a", "b")], None),
}
if len(expected["y ~ a*b*c*d"][1]) != 15 or expected["y ~ a*b*c*d"][1][-1] != ("a", "b", "c", "d"):
    sys.exit("Validation test failed at initialization: The four-way expansion is not the 15 effects.")
for formula, result in expected.items():
    if parsed(formula) != result:
        sys.exit("Validation test failed at case 1: '%s' was parsed as %s, not %s." % (formula, parsed(formula), result))

# the four-way model written out in a scrambled order: The same effects,
# sorted by degree, each named in the order its variables first appear
rng = np.random.default_rng(SEED)
effects = expansion(["a", "b", "c", "d"])
for _ in range(10):
    scrambled = [effects[k] for k in rng.permutation(len(effects))]
    formula = "y ~ " + " + ".join(":".join(rng.permutation(term)) for term in scrambled)
    response, terms, group = parsed(formula)
    first = []
    for name in formula.replace("y ~", "").replace("+", ":").replace(" ", "").split(":"):
        if name not in first:
            first.append(name)
    if sorted(map(frozenset, terms), key=sorted) != sorted(map(frozenset, effects), key=sorted) or \
            [len(term) for term in terms] != sorted(len(term) for term in terms) or \
            any(list(term) != sorted(term, key=first.index) for term in terms):
        sys.exit("Validation test failed at case 1: The written-out four-way model '%s' was parsed as %s." % (formula, terms))

# an interaction needs every lower-order effect from three variables on; the
# message names a missing effect and the '*' formula
for formula, missing, written in (("y ~ a*b*c + d + a:b:c:d", "a:d", "a*b*c*d"),
                                  ("y ~ a*b*c*d - 1", None, None),
                                  ("y ~ a + b + c + a:b + a:c + a:b:c", "b:c", "a*b*c")):
    words = () if missing is None else ("'%s'" % missing, written)
    if not refused(formula, words):
        sys.exit("Validation test failed at case 1: '%s' was not refused with a message naming %s and %s." % (formula, missing, written))


# ------------------------------------------------------------------
# CASE 2: Spaces around the operators
# ------------------------------------------------------------------

# hand-written: No spaces, many spaces and the mix a busy person types
spacings = {
    "y ~ x1*x2 + x3:x4 + x3 + x4 + (1|grp)": [
        "y~x1*x2+x3:x4+x3+x4+(1|grp)",
        "  y   ~   x1   *   x2   +   x3   :   x4   +   x3   +   x4   +   (   1   |   grp   )   ",
        "y ~x1* x2+  x3 :x4 +x3+   x4+( 1|grp )",
        "y~ x1 *x2 +x3:  x4+x3 + x4 +(1 |  grp)",
        "y ~ x1*x2+x3:x4 + x3+x4 + (1|grp)",
        "\ty ~ x1 * x2 +\tx3:x4 + x3 + x4 + (1|grp)\t",
    ],
    "y ~ a*b*c*d": [
        "y~a*b*c*d",
        "y ~ a * b * c * d",
        "y ~a*b *c*  d",
        "y~ a  *b* c *d ",
    ],
    "y ~ a + b + c + a:b + a:c + b:c + a:b:c": [
        "y~a+b+c+a:b+a:c+b:c+a:b:c",
        "y ~ a+b+ c+a :b+ a: c+b:c +a:b:c",
        "y   ~a   +b+c   +a:b+a:c+b:c+  a : b : c",
    ],
}
for base, variants in spacings.items():
    reference = parsed(base)
    for variant in variants:
        if parsed(variant) != reference:
            sys.exit("Validation test failed at case 2: '%s' was parsed as %s, not as '%s' (%s)." % (variant, parsed(variant), base, reference))


def respace(formula, rng):
    """Removes every space and puts 0-4 back at random around each operator and parenthesis."""
    out = []
    for char in formula.replace(" ", ""):
        if char in "~+:*|()":
            out.append(" " * rng.integers(0, 5) + char + " " * rng.integers(0, 5))
        else:
            out.append(char)
    return "".join(out)


for base in list(spacings) + ["y ~ x + z + w + x:z + x:w + z:w + x:z:w + (1|g)", "y ~ 1 + (1|g)"]:
    reference = parsed(base)
    for _ in range(NVARIANTS):
        variant = respace(base, rng)
        if parsed(variant) != reference:
            sys.exit("Validation test failed at case 2: '%s' was parsed as %s, not as '%s'." % (variant, parsed(variant), base))


# ------------------------------------------------------------------
# CASE 3: Long covariate names
# ------------------------------------------------------------------

long = {
    "y": "time_to_first_hospital_discharge",      # 32 characters
    "a": "systolic_blood_pressure_baseline",      # 32
    "b": "body_mass_index_at_first_visit",        # 30
    "c": "number_of_previous_admissions",         # 29
    "d": "treatment_arm.randomised_2019",         # 29
    "g": "recruiting_hospital_identifier",        # 30
}
if max(len(name) for name in long.values()) != 32:
    sys.exit("Validation test failed at initialization: The longest name is not 32 characters.")
for template, terms, group in (
        ("{y} ~ {a}*{b}*{c}*{d} + (1|{g})", expansion(["a", "b", "c", "d"]), "g"),
        ("{y} ~ {a} + {b} + {a}:{b} + {c} + (1|{g})", [("a",), ("b",), ("c",), ("a", "b")], "g"),
        ("{y} ~ {a}*{b} + {c}:{d} + {c} + {d}", [("a",), ("b",), ("c",), ("d",), ("a", "b"), ("c", "d")], None)):
    formula = template.format(**long)
    translated = (long["y"], [tuple(long[name] for name in term) for term in terms], None if group is None else long[group])
    for variant in (formula, respace(formula, rng)):
        if parsed(variant) != translated:
            sys.exit("Validation test failed at case 3: '%s' was parsed as %s, not %s." % (variant, parsed(variant), translated))


# ------------------------------------------------------------------
# CASE 4: Names with spaces, special symbols and other alphabets
# ------------------------------------------------------------------

special = {
    "y ~ `body mass index (kg/m^2)` + `x-1`":
        ("y", [("body mass index (kg/m^2)",), ("x-1",)], None),
    "`time to event (days)` ~ `dose + 1`*`age, years` + (1|`hospital id`)":
        ("time to event (days)", [("dose + 1",), ("age, years",), ("dose + 1", "age, years")], "hospital id"),
    "y ~ `a:b` + `p*q` + `a:b`:`p*q`":
        ("y", [("a:b",), ("p*q",), ("a:b", "p*q")], None),
    "y ~ `#visits` + `1st line` + `ω score` + `time~dependent` + `(1|g)`":
        ("y", [("#visits",), ("1st line",), ("ω score",), ("time~dependent",), ("(1|g)",)], None),
    "højde ~ ålder*größe + vægt_kg + (1|sygehus)":
        ("højde", [("ålder",), ("größe",), ("vægt_kg",), ("ålder", "größe")], "sygehus"),
    "y ~ `body mass index`*`sex (m/f)`*`x-1`*treatment":
        ("y", expansion(["body mass index", "sex (m/f)", "x-1", "treatment"]), None),
}
for formula, result in special.items():
    if parsed(formula) != result:
        sys.exit("Validation test failed at case 4: '%s' was parsed as %s, not %s." % (formula, parsed(formula), result))
    if "`" not in formula:
        variant = respace(formula, rng)
        if parsed(variant) != result:
            sys.exit("Validation test failed at case 4: '%s' was parsed as %s, not %s." % (variant, parsed(variant), result))

# hand-written spacing around names in backticks, symbols inside untouched
variants = {
    "y ~ `body mass index (kg/m^2)` + `x-1`": [
        "y~`body mass index (kg/m^2)`+`x-1`",
        "y ~  `body mass index (kg/m^2)`+   `x-1` ",
        " y~ `body mass index (kg/m^2)` +`x-1`"],
    "`time to event (days)` ~ `dose + 1`*`age, years` + (1|`hospital id`)": [
        "`time to event (days)`~`dose + 1`*`age, years`+(1|`hospital id`)",
        "`time to event (days)`  ~`dose + 1` *  `age, years`+ ( 1 |`hospital id` )"],
    "y ~ `body mass index`*`sex (m/f)`*`x-1`*treatment": [
        "y~`body mass index`*`sex (m/f)`*`x-1`*treatment",
        "y ~`body mass index` *`sex (m/f)`*  `x-1` * treatment "],
}
for formula, written in variants.items():
    for variant in written:
        if parsed(variant) != special[formula]:
            sys.exit("Validation test failed at case 4: '%s' was parsed as %s." % (variant, parsed(variant)))

# a name in backticks and the same name without them are one variable
if parsed("y ~ `x` + x:z + z") != ("y", [("x",), ("z",), ("x", "z")], None) or not refused("y ~ x + `x`"):
    sys.exit("Validation test failed at case 4: 'x' and '`x`' were not taken as one variable.")

# names with spaces or symbols must be in backticks, with a message saying so
for formula in ("y ~ body mass + age", "y ~ x-1 + z", "y ~ #visits", "y ~ x + (1|hospital id)"):
    if not refused(formula):
        sys.exit("Validation test failed at case 4: '%s' was accepted." % formula)
if not refused("y ~ body mass + age", ("backticks",)):
    sys.exit("Validation test failed at case 4: The refusal of a name with a space does not mention backticks.")
for formula, word in (("y ~ `body mass + age", "backtick"), ("y ~ `` + age", "empty"), ("y ~ ` ` + age", "empty")):
    if not refused(formula, (word,)):
        sys.exit("Validation test failed at case 4: '%s' was not refused with a message mentioning '%s'." % (formula, word))


# ------------------------------------------------------------------
# CASE 5: Every effect once
# ------------------------------------------------------------------

accepted = {
    "y ~ x*y2 + z*w": [("x",), ("y2",), ("z",), ("w",), ("x", "y2"), ("z", "w")],
    "y ~ x + y2 + z + x:y2 + y2:z + z:x + y2:z:x": [("x",), ("y2",), ("z",), ("x", "y2"), ("y2", "z"), ("x", "z"), ("x", "y2", "z")],
    # names follow the order in which the variables first appear (z, x, y2)
    "y ~ z:x:y2 + x:z + y2 + x + z:y2 + z + y2:x": [("y2",), ("x",), ("z",), ("z", "x"), ("z", "y2"), ("x", "y2"), ("z", "x", "y2")],
}
for formula, terms in accepted.items():
    if parsed(formula)[1] != terms:
        sys.exit("Validation test failed at case 5: '%s' was parsed as %s, not %s." % (formula, parsed(formula)[1], terms))
# the order of the variables in an interaction does not matter: The same model
if sorted(map(frozenset, parsed("y ~ x + y2 + z + x:y2 + y2:z + z:x + y2:z:x")[1]), key=sorted) != \
        sorted(map(frozenset, expansion(["x", "y2", "z"])), key=sorted):
    sys.exit("Validation test failed at case 5: Interactions written in mixed order are not the model x*y2*z.")

# an effect given twice, by "*" or written out, is refused with a message
# naming it
for formula, effect in (("y ~ x*y2 + x*y2*z", "x"), ("y ~ x + y2 + x*y2", "x"), ("y ~ x*y2 + y2:x", "x:y2"),
                        ("y ~ x + x", "x"), ("y ~ x + y2 + x:y2 + y2:x", "x:y2"), ("y ~ a*b + c*d + a*c", "a")):
    if not refused(formula, ("Infeasible formula", "'%s'" % effect)):
        sys.exit("Validation test failed at case 5: '%s' was not refused as giving '%s' twice." % (formula, effect))

# through fitreg: An error is printed and the program terminates
frame = pd.DataFrame({"y": rng.exponential(size=30), "x": rng.normal(size=30), "y2": rng.normal(size=30),
                      "z": rng.normal(size=30)})
for formula in ("y ~ x*y2 + x*y2*z", "y ~ x + y2 + x*y2"):
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            fitreg(formula=formula, data=frame, nphases=1, restarts=1)
        sys.exit("Validation test failed at case 5: fitreg accepted '%s'." % formula)
    except SystemExit as stop:
        if stop.code != 1 or not printed.getvalue().startswith("Error: Infeasible formula"):
            sys.exit("Validation test failed at case 5: fitreg did not print 'Error: Infeasible formula' for '%s'." % formula)


print("_regformula: All tests passed.")
