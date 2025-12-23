
<img src="https://github.com/areenberg/phasedist/blob/main/images/PhaseDist_logo.png" alt="Example" width="300">

PhaseDist is a Python package for fitting continuous and discrete phase-type distributions.

## Features

* Fit continuous and discrete phase-type distributions using EM algorithms.
* Use built-in methods to easily check the fitted distribution.
* Evaluate various metrics, such as the mean, density, quantile function, AIC and more.
* Simulate observations from phase-type distributions.
* Approximate another distribution, e.g. a log-normal distribution, using a phase-type distribution.

# Installation

Install directly from PyPI with:

```
pip install phasedist
```

# Quick start guide

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

(3) Compare the CDF of the fitted distribution to the empirical CDF.

```python
fit.plot()
```

<img src="https://github.com/areenberg/phasedist/blob/main/images/quickstart_example_CDF.png" alt="Example" width="500">

(4) Store the fitted distribution in the object `phdist` and compute the mean, variance, and 95% quantile.

```python
phdist = fit.getdist()

print(phdist.getmean())
#1.455814

print(phdist.getvar())
#0.706466

print(phdist.getquantile(p=0.95))
#3.055169
```

Find the complete example in the file `quickstart_example.py`.

# Documentation

The documentation can be found in the [wiki for PhaseDist](https://github.com/areenberg/phasedist/wiki).