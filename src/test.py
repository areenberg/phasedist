from fit import fit
from fitcph2dist import fitcph2dist
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

seed=123
np.random.seed(seed)

#obs = np.random.geometric(0.2,size=100)
#obs = np.random.negative_binomial(2,0.5,size=100)+2 #convert to trials by adding two
#obs = np.random.exponential(scale=1.0,size=100)
#obs = np.random.lognormal(0,0.25,size=100)

#ph = fit(obs=obs,nphases=4,dtype="coxian",discrete=True,verbose=True)

#print(ph.getinitdist())
#print(ph.getphasegen())
#print(ph.getexitrates())

#print(ph.getmean())
#print(obs.mean())

#print(ph.getvar())
#print(obs.var())

#ph.plot()


ph = fitcph2dist(nphases=3,
                 dtype="coxian",
                 randominit=True,
                 seed=seed,
                 tolerance=1e-3,
                 truncation=0.99,
                 steps=50,
                 verbose=True)


ph.chisq(df=2)
ph.fit()
ph.plot()

print(ph.getmean())
print(ph.getvar())

#print("mean =",ph.getmean())
#print("variance =",ph.getvar())
#print("density =",ph.getdensity(1))
#print("P(X<=x) =",ph.getcumprob(1))
#print("LogLik =",ph.getloglik())
#print("AIC =",ph.getaic())
#print("BIC =",ph.getbic())

