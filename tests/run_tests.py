'''
TEST SCRIPT FOR THE PHASEDIST PACKAGE
'''

# Load phasedist from src-folder including dependencies
import os
import sys
import numpy as np
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "..", "src")
sys.path.append(os.path.abspath(src_path))
import phasedist as ph

#------------------------------------------------------------------
# TEST 1: Fit an Erlang-3 CPH distribution using simulated data
#------------------------------------------------------------------

obs = np.loadtxt('tests/erlang3_data.csv',skiprows=1)

initdist = np.array([1.0,0.0,0.0])
phasegen = np.array([[-1.82,1.82,0.0],
                     [0.0,-4.49,4.49],
                     [0.0,0.0,-12.2]])
exitrates = np.array([0.0,0.0,12.2])

fit = ph.fit(obs=obs,
             nphases=3,
             dtype="generlang",
             initdist=initdist,
             initphgen=phasegen,
             initexitrates=exitrates,
             randominit=False,
             discrete=False,
             verbose=False,
             tolerance=1e-9)
phdist = fit.getdist()

# Compare estimated parameters to the expected results

if not np.array_equal(
    np.round(phdist.getinitdist(), 3),
    np.round(np.array([[1.0,0.0,0.0]]), 3),
):
    sys.exit("Validation test failed at estimation of initial distribution for Erlang-3 distribution.")
    
if not np.array_equal(
    np.round(phdist.getphasegen(), 6),
    np.round(np.array([[-2.06070328,2.06070328,0.0],
                      [0.0,-2.06070328,2.06070328],
                      [0.0,0.0,-2.06070328]]), 6),
):
    sys.exit("Validation test failed at estimation of phase-type generator for Erlang-3 distribution.")
    
if not np.array_equal(
    np.round(phdist.getexitrates(), 6),
    np.round(np.array([[0.0],[0.0],[2.06070328]]), 6),
):
    sys.exit("Validation test failed at estimation of exit rates for Erlang-3 distribution.")

#---------------------------------------------------------------------------
# TEST 2: Fit a negative binomial-3 DPH distribution using simulated data
#---------------------------------------------------------------------------

obs = np.loadtxt('tests/negbinom3_data.csv',skiprows=1)

initdist = np.array([1.0,0.0,0.0])
phasegen = np.array([[0.52,0.48,0.0],
                     [0.0,0.10,0.90],
                     [0.0,0.0,0.84]])
exitrates = np.array([0.0,0.0,0.16])

fit = ph.fit(obs=obs,
             nphases=3,
             dtype="generlang",
             initdist=initdist,
             initphgen=phasegen,
             initexitrates=exitrates,
             randominit=False,
             discrete=True,
             verbose=False,
             tolerance=1e-9)
phdist = fit.getdist()

# Compare estimated parameters to the expected results

if not np.array_equal(
    np.round(phdist.getinitdist(), 3),
    np.round(np.array([[1.0,0.0,0.0]]), 3),
):
    sys.exit("Validation test failed at estimation of initial distribution for negative binomial-3 distribution.")
    
if not np.array_equal(
    np.round(phdist.getphasegen(), 6),
    np.round(np.array([[0.46996466,0.53003534,0.0],
                      [0.0,0.46996466,0.53003534],
                      [0.0,0.0,0.46996466]]), 6),
):
    sys.exit("Validation test failed at estimation of phase-type generator for negative binomial-3 distribution.")
    
if not np.array_equal(
    np.round(phdist.getexitrates(), 6),
    np.round(np.array([[0.0],[0.0],[0.53003534]]), 6),
):
    sys.exit("Validation test failed at estimation of exit rates for negative binomial-3 distribution.")
    
#---------------------------------------------------------------------------
# TEST 3: Test the distribution object
#---------------------------------------------------------------------------    

initdist = np.array([0.1,0.5,0.4])
phasegen = np.array([[-1.0,0.0,1.0],
                     [0.0,-2.0,0.0],
                     [0.0,4.0,-5.0]])

phdist = ph.dist(initdist=initdist,
              phgen=phasegen,
              seed=123)

if not phdist.getmean()==0.65:
    sys.exit("Validation test failed at estimation of mean.")
if not round(phdist.getvar(),2)==0.47:
    sys.exit("Validation test failed at estimation of variance.")
if not round(phdist.getcumprob(x=1.953),3)==0.950:
    sys.exit("Validation test failed at estimation of distribution function.")
if not round(phdist.getquantile(p=0.95),3)==1.953:
    sys.exit("Validation test failed at estimation of quantile.")
if not round(phdist.getdensity(x=1),3)==0.322:
    sys.exit("Validation test failed at estimation of density.")
if phdist.getrandom(size=1)>6.0:
    sys.exit("Validation Test 3 failed since an unlikely high number was sampled. Try running the test again. If the problem persist, check your installation.")

#---------------------------------------------------------------------------
# TEST 4: Test the approximation of continuous density to PH method
#---------------------------------------------------------------------------    

initdist = np.array([1.0,0.0,0.0])
phasegen = np.array([[-1.50,1.0,0.0],
                     [0.0,-4.0,3.0],
                     [0.0,0.0,-12.0]])
exitrates = np.array([0.5,1.0,12.0])

fit = ph.fitcph2dist(nphases = 3,
                    initdist=initdist,
                    initphgen=phasegen,
                    initexitrates=exitrates,
                    dtype = "coxian",
                    tolerance=0.01,
                    verbose=False)

fit.chisq(df = 3)
fit.fit()

phdist = fit.getdist()

# Compare estimated parameters to the expected results

if not np.array_equal(
    np.round(phdist.getinitdist(), 3),
    np.round(np.array([[1.0,0.0,0.0]]), 3),
):
    sys.exit("Validation test failed at approximation of Chi-squared distribution.")
    
if not np.array_equal(
    np.round(phdist.getphasegen(), 6),
    np.round(np.array([[-0.82841462,0.7202343,0.0],
                      [0.0,-0.89193789,0.4978001],
                      [0.0,0.0,-0.58758287]]), 6),
):
    sys.exit("Validation test failed at approximation of Chi-squared distribution.")
    
if not np.array_equal(
    np.round(phdist.getexitrates(), 6),
    np.round(np.array([[0.10818032],[0.39413778],[0.58758287]]), 6),
):
    sys.exit("Validation test failed at approximation of Chi-squared distribution.")

#---------------------------------------------------------------------------
# FINAL VALIDATION
#---------------------------------------------------------------------------

print("All tests completed successfully!")