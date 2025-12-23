#Load the PhaseDist and NumPy packages
import phasedist as ph
import numpy as np

#Load the observed data
obs = np.array([1.48246359,1.13468709,0.66779536,0.61823347,0.8888217,1.10124776,0.1424737,2.1228061,
1.73924933,0.9849647,1.4828275,1.97188842,2.56132465,1.58038807,1.27567082,2.7754917,1.42516854,0.4602795,
1.93701091,2.50633135,1.92906099,1.60935023,1.41949599,1.14870169,0.79146146,1.31530543,1.81352371,1.17079096,
0.78948314,1.39528837,1.62003755,1.52143826,0.46665594,1.37913488,3.10066725,0.76942733,1.42849783,1.61511175,
2.94617609,1.53719196,1.01144357,2.00466269,0.56886361,1.62237618,0.41023332,0.78733512,4.01849928,1.27761144,
1.09426382,1.36946933])

#Fit the data to a generalized Erlang distribution with 3 phases
fit = ph.fit(obs=obs,
             nphases=3,
             dtype="generlang")

#Compare the fitted CDF to the observed data
fit.plot()

#Store the fitted distribution in `phdist` 
phdist = fit.getdist()

#Compute and print the mean
print(phdist.getmean())

#Compute and print the variance
print(phdist.getvar())

#Compute and print the 95% quantile
print(phdist.getquantile(p=0.95))
