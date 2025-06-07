import phasedist as ph
import numpy as np

#----------------------------------------
# EMPIRICAL APPROACH
#----------------------------------------

#simulate some data
sample_size = 500
obs = np.zeros(sample_size)

#continuous

df = np.zeros(1000000)
for i in range(df.size):
    initdist = np.random.uniform(size=3)
    initdist /= np.sum(initdist)
    a = np.random.uniform(size=9)*5
    d = ph.dist(discrete=False,initdist=initdist,phgen=np.matrix([[-(a[0]+a[1]+a[2]),a[0],a[1]],[a[3],-(a[3]+a[4]+a[5]),a[4]],[a[6],a[7],-(a[6]+a[7]+a[8])]]))
    x = np.round(np.random.uniform()*3.0+0.001,8)
    p = d.getcumprob(x)
    newx = d.getquantile(p)
    df[i] = abs(newx-x)/x
print(df.max())
print(np.quantile(df,0.9999))
print(np.quantile(df,0.999))
print(np.quantile(df,0.99))

'''
obs = d.getrandom(sample_size)

#discrete
#d = ph.dist(discrete=True,initdist=np.array([1.0,0.0,0.0]),phgen=np.matrix([[0.5,0.5,0.0],[0.0,0.5,0.5],[0.0,0.0,0.5]]))
#obs = d.getrandom(sample_size)

#fit the distribution
fit = ph.fit(obs=obs,nphases=3,dtype="gencoxian",discrete=False,verbose=True)

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
'''
#----------------------------------------
# PARAMETRIC APPROACH
#----------------------------------------
'''
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
'''