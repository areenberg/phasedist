'''
BENCHMARK TEST: CPH Erlang-3

In this benchmark test, we compare the 'Phasedist' package to
the 'matrixdist' package (from R) by fitting Erlang-3
distributions to the data in the CSV-file 'erlang3_data.csv'
and comparing the fitted distributions.
'''

# Load packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyreadr
import subprocess
import shutil
from pathlib import Path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "..", "src")
sys.path.append(os.path.abspath(src_path))
import phasedist as ph

# Fit an Erlang-3 CPH distribution using Phasedist

obs = np.loadtxt('benchmarks/erlang3_data.csv',skiprows=1)
fit = ph.fit(obs=obs,nphases=3,dtype="generlang",discrete=False,verbose=True)
phdist = fit.getdist()

# Fit an Erlang-3 CPH distribution using matrixdist in R (installs the matrixdist package first)

subprocess.run(["Rscript","benchmarks/erlang3_in_R.R"], check=True)
matdist_result = pyreadr.read_r('benchmarks/ph_parameters.RData')
matdist = ph.dist(discrete=False,initdist=matdist_result['initdist'].values.flatten(),phgen=matdist_result['phgen'].values)

# Compare the estimated means and variances

print('Phasedist estimated mean:',phdist.getmean())
print('Phasedist estimated variance:',phdist.getvar())
print('matrixdist estimated mean:',matdist.getmean())
print('matrixdist estimated variance:',matdist.getvar())

# Compare the estimated densities

x = np.linspace(0,7,100)
denph = phdist.getdensity(x)
denmatdist = matdist.getdensity(x)

plt.figure(figsize=(8, 5))
plt.plot(x, denph, label="Phasedist", color="blue")
plt.plot(x, denmatdist, label="matrixdist", color="red", linestyle='--')
plt.title("Density Comparison")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Clean up R-files

file = Path("benchmarks/ph_parameters.RData")
file.unlink()
shutil.rmtree("benchmarks/R_lib/cli")
shutil.rmtree("benchmarks/R_lib/glue")
shutil.rmtree("benchmarks/R_lib/lifecycle")
shutil.rmtree("benchmarks/R_lib/magrittr")
shutil.rmtree("benchmarks/R_lib/matrixdist")
shutil.rmtree("benchmarks/R_lib/plyr")
shutil.rmtree("benchmarks/R_lib/Rcpp")
shutil.rmtree("benchmarks/R_lib/RcppArmadillo")
shutil.rmtree("benchmarks/R_lib/reshape2")
shutil.rmtree("benchmarks/R_lib/rlang")
shutil.rmtree("benchmarks/R_lib/stringi")
shutil.rmtree("benchmarks/R_lib/stringr")
shutil.rmtree("benchmarks/R_lib/vctrs")
