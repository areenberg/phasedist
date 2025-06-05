import sys
import numpy as np
from scipy.linalg import expm
from scipy.stats import lognorm
import matplotlib.pyplot as plt

class dist:
    #Phase-type distribution object for computing various metrics
    #and performing simulated samples

    def __init__(self,discrete=False,initdist=None,phgen=None,seed=None):

        self.discrete = discrete
        self.initdist = initdist
        self.phgen = phgen
        self.nphases = self.phgen.shape[0]
        self.seed = seed

        if self.__checkinputs(): #check inputs
            self.__initialize()
        else:
            sys.exit(1) #terminate the program

    def getmean(self):
        #returns the mean
        if self.discrete:
            return np.sum(np.matmul(self.initdist,np.linalg.inv(np.subtract(np.eye(self.nphases),self.phgen))))    
        else:
            return -np.sum(np.matmul(self.initdist,np.linalg.inv(self.phgen)))

    def getvar(self):
        #returns the variance
        if self.discrete:
            Tinv = np.linalg.inv(np.subtract(np.eye(self.nphases),self.phgen))
            return np.sum(np.matmul(np.matmul(self.pi,Tinv),np.subtract((2*Tinv),np.eye(self.nphases))))-self.getmean()**2
        else:
            phinv = np.linalg.inv(self.phgen)
            return 2*np.sum(np.matmul(self.pi,np.linalg.matrix_power(phinv,2)))-np.power(np.sum(np.matmul(self.pi,phinv)),2)

    def getdensity(self,x):
        #returns the density
        if self.discrete:
            if int(x)!=x:
                print("Error: 'x' is not an integer.")
                return np.nan
            else:
                return np.matmul(self.pi,np.matmul(np.linalg.matrix_power(self.phgen,(x-1)),self.exitrates)).item()
        else:
            return np.matmul(self.pi,np.matmul(expm(self.phgen*x),self.exitrates)).item()

    def getcumprob(self,x):
        #returns the cumulated probability P(X<=x)
        if self.discrete:
            if int(x)!=x:
                print("Error: 'x' is not an integer.")
                return np.nan
            else:
                return 1-np.sum(np.matmul(self.initdist,np.linalg.matrix_power(self.phgen,x)))
        else:
            return 1-np.sum(np.matmul(self.initdist,expm(self.phgen*x)))
        
    def getquantile(self,p,tolerance=1e-12):
        #returns the x that ensures P(X<=x)=p,
        #i.e. the quantile function of the
        #PH distribution
        if p==1.0:
            return np.inf
        elif p<0.0 or p>1.0:
            return np.nan
        elif p==0.0:
            return 0.0
        elif self.discrete:
            return self.__dphtruncquantfun(prob=p,idst=self.initdist,phg=self.phgen,itermax=1000000)
        else:
            return self.__cphtruncquantfun(self,prob=p,idst=self.initdist,phg=self.phgen,tol=tolerance,itermax=1000000)
        
    def getsample(self):
        #samples a random value from the PH distribution
        if self.discrete:
            return self.__dphsample()
        else:
            return self.__cphsample()
            
    def __initialize(self):
        #overall initialization
                
        #compute exit rate vector
        if self.discrete:
            self.exitrates = 1-np.sum(self.phgen,axis=1)
        else:
            self.exitrates = abs(np.sum(self.phgen,axis=1))    
        
        #set a pre-defined seed    
        if self.seed is not None:
            np.random.seed(self.seed)

    def __checkinputs(self):
        #check the feasibility of all input parameters
        #before proceeding
        
        #check data types and convert if necesarry
        if not isinstance(self.discrete,bool):
            print("Error: The argument 'discrete' needs to be of type 'bool'.")
        if self.initdist is not None and (isinstance(self.initdist,np.ndarray) or isinstance(self.initdist,list)):
            self.initdist = np.matrix(self.initdist)
        elif self.initdist is not None and not isinstance(self.initdist,np.matrix):
            print("Error: The initial distribution can only be specified as a list, NumPy array, or a NumPy matrix.")
            return False
        if self.phgen is not None and (isinstance(self.phgen,np.ndarray) or isinstance(self.phgen,list)):
            self.phgen = np.matrix(self.phgen)
        elif self.phgen is not None and not isinstance(self.phgen,np.matrix):
            print("Error: The PH generator can only be specified as a list or a NumPy matrix.")
            return False
        if self.seed is not None and not isinstance(self.seed,int):
            print("Error: The seed can only be specified as an integer.")
            return False
        return True #if all correct
        
    def __cphsample(self):
        #samples a random value from the
        #continuous phase-type (CPH) distribution
        
        p = len(self.initdist)
        t = 0.0
        s = np.random.choice(p, size=1, p=self.initdist)[0]
        while True:
            t += np.random.exponential(scale=1/(-self.phgen[s, s]))
            a = self.phgen[s, :]/(-self.phgen[s,s])
            a[a < 0] = 0
            a_cumsum = np.cumsum(a)
            r = np.random.uniform(0,1)
            l = np.where(a_cumsum > r)[0]
            if len(l) == 0:
                return t
            else:
                s = l[0]
    
    def __dphsample(self):
        #samples a random value from the
        #discrete phase-type (DPH) distribution
        
        t = 0  # Time in discrete steps
        s = np.random.choice(self.nphases, size=1, p=self.initdist)[0]
        
        while True:
            t += 1
            a = np.array([self.phgen[s, :],self.exitrates[s].item()])
            s = np.random.choice((self.nphases+1),size=1,p=a)[0]
            if s == self.nphases:
                return t
            
    def __cphtruncquantfun(self,prob,tol=1e-12,itermax=1000000):
        #numerical quantile function for the
        #CPH distribution
        
        #some initial preparation
        phinv = np.linalg.inv(self.phgen)
        var = 2*np.sum(np.matmul(self.initdist,np.linalg.matrix_power(phinv,2)))-np.power(np.sum(np.matmul(self.initdist,phinv)),2)
        mean = -np.sum(np.matmul(self.initdist,phinv))
        
        #generate initial guess
        param1 = np.log(np.power(mean,2)/np.sqrt(np.power(mean,2)+var))
        param2 = np.log(1 + var/np.power(mean,2))
        x=lognorm.ppf(prob,param2,scale=np.exp(param1)) 
        
        #improve x until convergence
        trc=1-np.sum(np.matmul(self.initdist,expm(self.phgen*x)))
        dd = np.sqrt(var)*tol
        iter=0
        while np.abs(trc-prob)>tol and iter<itermax:
                x1=x+dd
                f1 = 1-np.sum(np.matmul(self.initdist,expm(self.phgen*x1)))
                grad = (f1-trc)/dd
                x = x - (trc-tol)/grad
                trc = 1-np.sum(np.matmul(self.initdist,expm(self.phgen*x)))
                iter+=1
        if iter==itermax:
            print("Warning: Algorithm terminated with iter==itermax. Results might be misleading.")
        return x

    def __dphtruncquantfun(self,prob,itermax=1000000):
        #numerical quantile function for the
        #DPH distribution

        #initialization
        x = int(-1)
        trc = 0.0
        iter = 0
        
        #improve x until convergence
        while trc<prob and iter<itermax:
            x += 1
            trc = 1-np.sum(np.matmul(self.initdist,np.linalg.matrix_power(self.phgen,x)))
            iter += 1
        if iter==itermax:
            print("Warning: Algorithm terminated with iter==itermax. Results might be misleading.")
        return x