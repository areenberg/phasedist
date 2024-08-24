import numpy as np
from scipy.linalg import expm

# REFERENCES

#Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in Applied Probability. 
#Springer. https://doi.org/10.1007/978-1-4939-7049-0

class fitcph:
    #fit continuous-time phase-type distributions using the
    #EM algorithm from p. 678 Bladt and Nielsen (2017).

    def __init__(self,obs=None,initpi=None,initphgen=None,initexitrates=None,randominit=True,seed=None,tolerance=1e-6,itermax=1e6,verbose=False):
        self.obs = obs #observed realizations of the PH distribution
        self.initpi = initpi
        self.initphgen = initphgen
        self.initexitrates = initexitrates
        self.nphases = self.initphgen.shape[0]
        self.randominit = randominit
        self.seed = seed
        self.itermax = itermax
        self.tolerance=tolerance
        self.verbose=verbose
        self.__initialize()

    def fit(self):
        #fit the CPH distribution
        iter=0
        eps = np.inf
        loglik0 = -np.inf
        while iter<self.itermax and eps>self.tolerance:
            self.__estep()
            self.__mstep()
            eps = self.loglikelihood-loglik0 #loglik is evaluated within the E-step 
            #print(self.loglikelihood,loglik0,eps)
            loglik0 = self.loglikelihood
            iter += 1
            if self.verbose and iter%5==0:
                print("iter =",iter,"  eps =",eps,"  mean =",self.getmean(),"  var =",self.getvar())
            
        self.__updatelikelihood() #evaluate final loglik

    def getinitdist(self):
        #returns the initial distribution
        return self.pi
    
    def getphasegen(self):
        #returns the phase-type generator
        return self.phgen
    
    def getexitrates(self):
        #returns the exit rate vector
        return self.exitrates 
            
    def getmean(self):
        #returns the mean of the CPH
        return -np.sum(np.matmul(self.pi,np.linalg.inv(self.phgen)))

    def getvar(self):
        #returns the variance of the CPH
        phinv = np.linalg.inv(self.phgen)
        return 2*np.sum(np.matmul(self.pi,np.linalg.matrix_power(phinv,2)))-np.power(np.sum(np.matmul(self.pi,phinv)),2)

    def getdensity(self,x):
        #returns the density of the CPH
        return np.matmul(self.pi,np.matmul(expm(self.phgen*x),self.exitrates)).item()

    def getcumprob(self,x):
        #returns the cumulated probability P(X<=x) of the CPH
        return 1-np.sum(np.matmul(self.pi,expm(self.phgen*x)))

    def getloglik(self):
        return self.loglikelihood

    def getaic(self):
        #returns the Akaike's Information Criteria (AIC)
        return -2*self.loglikelihood+2*self.nparam

    def getbic(self):
        #returns the Bayesian Information Criteria (BIC)
        return -2*self.loglikelihood+self.nparam*np.log(len(self.obs))

    def __initialize(self):    
        if self.seed is not None:
            np.random.seed(self.seed)
        
        #convert data types
        self.obs = self.obs.astype(float) #observations must be float
        self.initpi = self.initpi.astype(float)
        self.initphgen = self.initphgen.astype(float)
        self.initexitrates = self.initexitrates.astype(float)
        self.identity = np.eye(self.nphases)
        
        #copy to output parameters    
        self.pi = self.initpi      
        self.phgen = self.initphgen
        self.exitrates = self.initexitrates    
        
        #initialize with random parameters
        #accounting for the specified structure
        if self.randominit:
            self.__initrandom()
        #self.__updatelikelihood() #compute the log-likelihood
        self.__countParameters()    
            
    def __initrandom(self):
        #initialize with a random CPH distribution
        
        #make a random initial distribution (pi)
        nzidx = np.nonzero(self.pi)[1]
        u = np.random.uniform(low=0.0,high=1.0,size=len(nzidx))
        u = u/np.sum(u)
        self.pi[0,nzidx] = u
        
        #make a random exit vector
        nzidx = np.nonzero(self.exitrates)[0]
        u = np.random.uniform(low=0.0, high=10.0, size=len(nzidx))
        self.exitrates[nzidx,0] = u
        
        #make a random PH generator
        for i in range(self.nphases):
            nzidx = np.nonzero(self.phgen[i,:])[1]
            msk = nzidx != i
            nzidx = nzidx[msk]
            u = np.random.uniform(low=0.0, high=10.0, size=len(nzidx))
            self.phgen[i,nzidx] = u
            self.phgen[i,i] = -(np.sum(u)+self.exitrates[i,0])
        
    def __estep(self):
        #performs the E-step
        
        self.bi = np.zeros(self.nphases)
        self.zi = np.zeros(self.nphases)
        self.ni = np.zeros(self.nphases)
        self.nij = np.zeros((self.nphases,self.nphases))
        self.loglikelihood = 0.0
        
        for y in self.obs:
            self.__Jmatrix(y) #update J matrix and exp(T*y)
            eTyt = np.matmul(self.eTy,self.exitrates)
            pieTy = np.matmul(self.pi,self.eTy)
            pieTyt = np.matmul(pieTy,self.exitrates)
            self.loglikelihood += np.log(pieTyt)
            for i in range(self.nphases):
                #compute B_i (expected number of times starting in phase i)
                self.bi[i] += (self.pi[0,i]*eTyt[i,0])/pieTyt
                #compute Z_i (expected time spend in phase i)
                self.zi[i] += self.Jmat[i,i]/pieTyt
                #compute N_ij (expected transitions between phase i and j)
                for j in range(self.nphases):
                    if j!=i:
                        self.nij[i,j] += (self.phgen[i,j]*self.Jmat[j,i])/pieTyt
                #compute N_i (expected number of transitions to absorbing state from phase i)
                self.ni[i] += (pieTy[0,i]*self.exitrates[i,0])/pieTyt
        
    def __mstep(self):
        #performs the M-step
        
        #update the initial distribution
        for i in range(self.nphases):
            self.pi[0,i] = self.bi[i]/len(self.obs)
            
        #update the exit rates
        for i in range(self.nphases):
            self.exitrates[i,0] = self.ni[i]/self.zi[i]
        
        #update the PH generator
        for i in range(self.nphases):
            sm=self.exitrates[i,0]
            for j in range(self.nphases):
                if j!=i:
                    self.phgen[i,j] = self.nij[i,j]/self.zi[i]
                    sm += self.phgen[i,j]
            self.phgen[i,i] = -sm
            
    def __updatelikelihood(self):
        self.loglikelihood = 0.0
        for y in self.obs:
            self.loglikelihood += np.log(self.getdensity(y))
    
    def __Jmatrix(self,y):
        #computes the matrix-function 'J' as well as
        #the matrix 'exp(T*y)' for the observations 'y'
        
        mat = expm(np.block([[self.phgen,np.matmul(self.exitrates,self.pi)],
                        [np.zeros((self.nphases,self.nphases)),self.phgen]])*y)
        
        self.eTy = mat[:self.nphases,:self.nphases]
        self.Jmat = mat[:self.nphases, self.nphases:2*self.nphases]
        

    def __countParameters(self):
        #count the number of independent parameters
        phg = 0
        for i in range(self.nphases): #independent parameters in each phase of the PH generator and exit vector
            phg += np.count_nonzero(self.phgen[i,:])+np.count_nonzero(self.exitrates[i,0])-1 #subtract the diagonal element
        self.nparam = phg+(np.count_nonzero(self.pi)-1) #add the number of independent parameters in the initial distribution