<img src="https://github.com/areenberg/phasedist/blob/development/images/phasedist_logo_dark.png" alt="PhaseDist logo" width="500">

[![PyPI](https://img.shields.io/pypi/v/phasedist?logo=pypi&logoColor=white)](https://pypi.org/project/phasedist/)
[![Python](https://img.shields.io/pypi/pyversions/phasedist?logo=python&logoColor=white)](https://pypi.org/project/phasedist/)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/areenberg/phasedist/blob/main/LICENSE)

PhaseDist is a Python package for fitting phase-type regressions and distributions.

## Features

* Fit phase-type regressions containing fixed and random effects.
* Fit continuous and discrete phase-type distributions.
* Fit to uncensored or censored observations.
* Approximate another distribution, e.g. a log-normal distribution, using a phase-type distribution.
* Use built-in methods to check your fit.
* Simulate observations from a phase-type distribution.

<img src="https://github.com/areenberg/phasedist/blob/development/images/phasedist_fit.gif" alt="A phase-type distribution being fitted to bimodal data" width="500">

*A bimodal phase-type distribution fitted using PhaseDist.*

## Installation

Install directly from PyPI with:

```
pip install phasedist
```

## Quick start guide

### Fitting a phase-type regression

The following shows how to use PhaseDist for fitting a phase-type regression, where the covariates scale the phase-type generator matrix.

(1) Start by loading PhaseDist (and pandas) and reading the data. The data is a clinical trial dataset consisting of a continuous variable `dose`, a categorical variable `group` (`control` vs. `treatment`), and a response variable `time`.

```python
import phasedist as ph
import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/areenberg/phasedist/development/examples/regression_data.csv")
data["group"] = pd.Categorical(data["group"], categories=["control", "treatment"])
```

(2) Fit the regression `time ~ dose + group` with a Coxian distribution containing 2 phases.

```python
fit = ph.fitreg(formula="time ~ dose + group",
                data=data,
                nphases=2,
                dtype="coxian")
```

(3) Test the coefficients. A positive coefficient means shorter times.

```python
fit.coeftest()
#                     Estimate    Std.Err   LR stat  Pr(>Chi)
#dose                   0.1639     0.0410     15.44    <0.001
#group[treatment]       0.9902     0.2205     18.39    <0.001
```

(4) Store the fitted distribution of a patient with dose 5 in the treatment group in the object `phdist` and compute the mean, variance, and 95% quantile.

```python
phdist = fit.getdist(covariates={"dose": 5.0, "group": "treatment"})

print(phdist.getmean())
#3.210571

print(phdist.getvar())
#13.545293

print(phdist.getquantile(p=0.95))
#10.648819
```

Find the complete example, including a PP plot to check the fit, in the file `examples/regression_example.py`.

### Fitting a phase-type distribution

The following shows how to use PhaseDist for fitting a continuous phase-type distribution from observed data.

(1) Start by loading PhaseDist (and NumPy) and defining the observed data.

```python
import phasedist as ph
import numpy as np

obs = np.array([1.48246359,1.13468709,0.66779536,0.61823347,0.8888217,1.10124776,0.1424737,2.1228061,
1.73924933,0.9849647,1.4828275,1.97188842,2.56132465,1.58038807,1.27567082,2.7754917,1.42516854,0.4602795,
1.93701091,2.50633135,1.92906099,1.60935023,1.41949599,1.14870169,0.79146146,1.31530543,1.81352371,1.17079096,
0.78948314,1.39528837,1.62003755,1.52143826,0.46665594,1.37913488,3.10066725,0.76942733,1.42849783,1.61511175,
2.94617609,1.53719196,1.01144357,2.00466269,0.56886361,1.62237618,0.41023332,0.78733512,4.01849928,1.27761144,
1.09426382,1.36946933])
```

(2) Fit the data to a generalized Erlang distribution with 3 phases.

```python
fit = ph.fit(obs=obs,
             nphases=3,
             dtype="generlang")
```

(3) Store the fitted distribution in the object `phdist` and compute the mean, variance, and 95% quantile.

```python
phdist = fit.getdist()

print(phdist.getmean())
#1.455814

print(phdist.getvar())
#0.706465

print(phdist.getquantile(p=0.95))
#3.055168
```

Find the complete example in the file `examples/distribution_example.py`.

## Documentation

The documentation can be found in the [wiki for PhaseDist](https://github.com/areenberg/phasedist/wiki).

## License

PhaseDist is released under the [MIT License](https://github.com/areenberg/phasedist/blob/main/LICENSE).
