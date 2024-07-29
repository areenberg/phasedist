from fitdph import __fitdph
from fitcph import __fitcph
import numpy as np

seed=123
np.random.seed(seed)

#obs = np.random.geometric(0.2,size=100)
#obs = np.random.negative_binomial(2,0.2,size=100)+2 #convert to trials by adding two
obs = np.random.exponential(scale=1.0,size=100)

#print(obs)


print(np.mean(obs))
print(np.var(obs))

#ph = __fitdph(obs=obs,
#            initpi=np.matrix([[1,0]]),
#            initphgen=np.matrix([[1,1],
#                                 [0,1]]),initexitrates=np.transpose(np.matrix([[0,1]])),randominit=True,seed=seed)

ph = __fitcph(obs=obs,
            initpi=np.matrix([[1,1,1]]),
            initphgen=np.matrix([[1,1,1],
                                 [1,1,1],
                                 [1,1,1]]),initexitrates=np.transpose(np.matrix([[1,1,1]])),randominit=True,seed=seed)


ph.fit()


print("pi =",ph.pi)
print("phgen =",ph.phgen)
print("exitrates =",ph.exitrates)

print("mean =",ph.getmean())
print("variance =",ph.getvar())
print("density =",ph.getdensity(2))
print("P(X<=x) =",ph.getcumprob(2))
print("LogLik =",ph.getloglik())
print("AIC =",ph.getaic())
print("BIC =",ph.getbic())