from fit import fit
from fitcph2dist import fitcph2dist
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

seed=123
np.random.seed(seed)


#----------------------------------------
# EMPIRICAL APPROACH
#----------------------------------------

#simulate some data
#obs = np.random.geometric(0.2,size=100)
#obs = np.random.negative_binomial(2,0.5,size=100)+2 #convert to trials by adding two
#obs = np.random.exponential(scale=1.0,size=100)
#obs = np.random.lognormal(0,0.25,size=100)

#fit the distribution
#ph = fit(obs=obs,nphases=4,dtype="coxian",discrete=True,verbose=True)

#check the fitted parameters
#print(ph.getinitdist())
#print(ph.getphasegen())
#print(ph.getexitrates())

#check metrics of the fitted PH distribution
#print("mean =",ph.getmean())
#print("variance =",ph.getvar())
#print("density =",ph.getdensity(1))
#print("P(X<=x) =",ph.getcumprob(1))
#print("LogLik =",ph.getloglik())
#print("AIC =",ph.getaic())
#print("BIC =",ph.getbic())

#make a visual check
#ph.plot()

#----------------------------------------
# PARAMETRIC APPROACH
#----------------------------------------

#create object dedicated for approximating a parametric distribution
ph = fitcph2dist(nphases=3,
                 dtype="coxian",
                 randominit=True,
                 seed=seed,
                 tolerance=1e-3,
                 truncation=0.99,
                 steps=50,
                 verbose=True)

#select true distribution
ph.chisq(df=2)

#make the fit
ph.fit()

#make a visual comparison between the approximate and true distribution
ph.plot()

#check metrics of the approximate distribution
print(ph.getmean())
print(ph.getvar())


