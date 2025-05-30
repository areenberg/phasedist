import phasedist as ph
import numpy as np

seed=25681
np.random.seed(seed)

#----------------------------------------
# EMPIRICAL APPROACH
#----------------------------------------

#simulate some data
#obs = np.random.geometric(0.2,size=100)
#obs = np.random.negative_binomial(2,0.5,size=100)+2 #convert to trials by adding two
#obs = np.random.exponential(scale=1.0,size=100)
obs = np.random.lognormal(0,0.25,size=100)

#fit the distribution
fit = ph.fit(obs=obs,nphases=9,dtype="coxian",discrete=False,verbose=True)

#check the fitted parameters
print(fit.getinitdist())
print(fit.getphasegen())
print(fit.getexitrates())

#check metrics of the fitted PH distribution
print("mean =",fit.getmean())
print("variance =",fit.getvar())
print("density =",fit.getdensity(1))
print("P(X<=x) =",fit.getcumprob(1))
print("LogLik =",fit.getloglik())
print("AIC =",fit.getaic())
print("BIC =",fit.getbic())

#make a visual check
fit.plot()

#----------------------------------------
# PARAMETRIC APPROACH
#----------------------------------------

#create object for approximating a parametric distribution
apx = ph.fitcph2dist(nphases=3,
                 dtype="coxian",
                 randominit=True,
                 seed=seed,
                 tolerance=1e-3,
                 truncation=0.99,
                 steps=50,
                 verbose=True)

#select true distribution
apx.chisq(df=2)

#make the fit
apx.fit()

#make a visual comparison between the approximate and true distribution
apx.plot()

#check metrics of the approximate distribution
print(apx.getmean())
print(apx.getvar())
