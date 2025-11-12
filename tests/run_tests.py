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
             verbose=True,
             tolerance=1e-9)
phdist = fit.getdist()

# Compare estimated parameters to the expected results

if not np.array_equal(
    np.round(phdist.getinitdist(), 3),
    np.round(np.array([[1.0,0.0,0.0]]), 3),
):
    sys.exit("Validation test failed at estimation of Erlang-3 distribution.")
    
if not np.array_equal(
    np.round(phdist.getphasegen(), 6),
    np.round(np.array([[-2.06070328,2.06070328,0.0],
                      [0.0,-2.06070328,2.06070328],
                      [0.0,0.0,-2.06070328]]), 6),
):
    sys.exit("Validation test failed at estimation of Erlang-3 distribution.")
    
if not np.array_equal(
    np.round(phdist.getexitrates(), 6),
    np.round(np.array([[0.0],[0.0],[2.06070328]]), 6),
):
    sys.exit("Validation test failed at estimation of Erlang-3 distribution.")

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
             verbose=True,
             tolerance=1e-9)
phdist = fit.getdist()

# Compare estimated parameters to the expected results

if not np.array_equal(
    np.round(phdist.getinitdist(), 3),
    np.round(np.array([[1.0,0.0,0.0]]), 3),
):
    sys.exit("Validation test failed at estimation of negative binomial-3 distribution.")
    
if not np.array_equal(
    np.round(phdist.getphasegen(), 6),
    np.round(np.array([[0.46996466,0.53003534,0.0],
                      [0.0,0.46996466,0.53003534],
                      [0.0,0.0,0.46996466]]), 6),
):
    sys.exit("Validation test failed at estimation of negative binomial-3 distribution.")
    
if not np.array_equal(
    np.round(phdist.getexitrates(), 6),
    np.round(np.array([[0.0],[0.0],[0.53003534]]), 6),
):
    sys.exit("Validation test failed at estimation of negative binomial-3 distribution.")