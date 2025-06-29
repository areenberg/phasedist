import phasedist as ph
import numpy as np

#----------------------------------------
# EMPIRICAL APPROACH
#----------------------------------------


#simulate some data
sample_size = 50

d = ph.dist(discrete=False,initdist=np.array([1.0,0.0,0.0]),phgen=np.matrix([[-2.0,2.0,0.0],[0.0,-2.0,2.0],[0.0,0.0,-2.0]])) #continuous
#d = ph.dist(discrete=True,initdist=np.array([1.0,0.0,0.0]),phgen=np.matrix([[0.5,0.5,0.0],[0.0,0.5,0.5],[0.0,0.0,0.5]])) #discrete
obs = d.getrandom(sample_size)



#fit the distribution
fit = ph.fit(obs=obs,nphases=3,dtype="generlang",discrete=False,verbose=True)

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

print(fit.getquantile(0.25))
print(fit.getquantile(0.50))
print(fit.getquantile(0.75))
print(fit.getquantile(0.99))

#make a visual check
fit.plot()


#----------------------------------------
# PARAMETRIC APPROACH
#----------------------------------------

'''
#create object for approximating a parametric distribution
apx = ph.fitcph2dist(nphases=3,
                 dtype="generlang",
                 randominit=True,
                 tolerance=1e-3,
                 truncation=0.99,
                 steps=50,
                 verbose=True)

#select true distribution
#apx.chisq(df=2)
apx.phasedist(initdist=np.array([1.0,0.0,0.0]),phgen=np.matrix([[-2.0,2.0,0.0],[0.0,-2.0,2.0],[0.0,0.0,-2.0]]))

#make the fit
apx.fit()

#make a visual comparison between the approximate and true distribution
apx.plot()

#check metrics of the approximate distribution
print(apx.getmean())
print(apx.getvar())
'''