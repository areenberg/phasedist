from fitdph import __fitdph
from fitcph import __fitcph
from fitcph2dist import __fitcph2dist
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

seed=456
np.random.seed(seed)

#obs = np.random.geometric(0.2,size=100)
#obs = np.random.negative_binomial(2,0.2,size=100)+2 #convert to trials by adding two
#obs = np.random.exponential(scale=1.0,size=100)
#obs = np.random.lognormal(0,0.25,size=100)

#print(obs)


#print(np.mean(obs))
#print(np.var(obs))

#ph = __fitdph(obs=obs,
#            initpi=np.matrix([[1,0]]),
#            initphgen=np.matrix([[1,1],
#                                 [0,1]]),initexitrates=np.transpose(np.matrix([[0,1]])),randominit=True,seed=seed)

#ph = __fitcph(obs=obs,
#            initpi=np.matrix([[1,1,1]]),
#            initphgen=np.matrix([[1,1,1],
#                                 [1,1,1],
#                                 [1,1,1]]),initexitrates=np.transpose(np.matrix([[1,1,1]])),randominit=True,seed=seed)



ph = __fitcph2dist(initpi=np.matrix([[1,0,0,0,0]]),
            initphgen=np.matrix([[1,1,0,0,0],
                                 [0,1,1,0,0],
                                 [0,0,1,1,0],
                                 [0,0,0,1,1],
                                 [0,0,0,0,1]]),
            initexitrates=np.transpose(np.matrix([[1,1,1,1,1]])),
            randominit=True,
            seed=seed,
            tolerance=1e-3,
            truncation=0.99,
            steps=50)



ph.lognorm(mu=1,sigma=1)
ph.fit()


print("pi =",ph.pi)
print("phgen =",ph.phgen)
print("exitrates =",ph.exitrates)

print("mean =",ph.getmean())
print("variance =",ph.getvar())
print("density =",ph.getdensity(1))
print("P(X<=x) =",ph.getcumprob(1))
#print("LogLik =",ph.getloglik())
#print("AIC =",ph.getaic())
#print("BIC =",ph.getbic())



x = np.linspace(0.001, 15, 500)
lognorm_pdf = np.zeros(len(x))
ph_pdf = np.zeros(len(x))
for i in range(len(x)):
    lognorm_pdf[i] = ph.lognormdensity(x[i])
    ph_pdf[i] = ph.getdensity(x[i])
    

plt.figure(figsize=(10, 6))
plt.plot(x, lognorm_pdf, label='Log-norm. dist.', color='blue')
plt.plot(x, ph_pdf, label='PH dist.', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('PH distribution')
plt.legend()
plt.grid(True)
plt.show()