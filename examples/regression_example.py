"""
REGRESSION EXAMPLE

This file contains a simple example of fitting a phase-type
regression model to data using the `fitreg` class.
"""

#Load the PhaseDist and pandas packages
import os
import phasedist as ph
import pandas as pd

#Load the observed data from the CSV-file next to this script
data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression_data.csv"))

#Mark the variable named "group" as a categorical variable, with "control" as the reference level
data["group"] = pd.Categorical(data["group"], categories=["control", "treatment"])

#Fit a phase-type regression of the form "time ~ dose + group"
#based on a Coxian distribution with 2 phases
fit = ph.fitreg(formula="time ~ dose + group",
                data=data,
                nphases=2,
                dtype="coxian")

#Print a summary of the model, with a test of each effect
fit.summary()

#Check the fit with a PP plot
fit.plot()

#Store the fitted distribution of a patient with dose 5 in the treatment group
phdist = fit.getdist(covariates={"dose": 5.0, "group": "treatment"})

#Compute and print the mean
print("Mean:", phdist.getmean())

#Compute and print the variance
print("Variance:", phdist.getvar())

#Compute and print the 95% quantile
print("95% Quantile:", phdist.getquantile(p=0.95))
