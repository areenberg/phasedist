import numpy as np

# REFERENCES

#Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in Applied Probability. 
#Springer. https://doi.org/10.1007/978-1-4939-7049-0

class fitdph:
    #fit discrete-time phase-type distributions using the
    #EM algorithm from p. 675 Bladt and Nielsen (2017). 

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
        #fit the DPH distribution
        iter=0
        eps = np.inf
        loglik0 = self.loglikelihood
        while iter<self.itermax and eps>self.tolerance:
            self.__estep()
            self.__mstep()
            self.__updatelikelihood()
            eps = self.loglikelihood-loglik0
            loglik0 = self.loglikelihood
            iter += 1
            if self.verbose and iter%25==0:
                print("iter =",iter,"  eps =",eps.item(),"  mean =",self.getmean(),"  var =",self.getvar())
    
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
        #returns the mean of the DPH
        return np.sum(np.matmul(self.pi,np.linalg.inv(np.subtract(np.eye(self.nphases),self.phgen))))

    def getvar(self):
        #returns the variance of the DPH
        Tinv = np.linalg.inv(np.subtract(np.eye(self.nphases),self.phgen))
        return np.sum(np.matmul(np.matmul(self.pi,Tinv),np.subtract((2*Tinv),np.eye(self.nphases))))-self.getmean()**2

    def getdensity(self,x):
        #returns the density of the DPH
        return np.matmul(self.pi,np.matmul(np.linalg.matrix_power(self.phgen,(x-1)),self.exitrates)).item()

    def getcumprob(self,x):
        #returns the cumulated probability P(X<=x) of the DPH
        return 1-np.sum(np.matmul(self.pi,np.linalg.matrix_power(self.phgen,x)))

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
        self.obs = self.obs.astype(int) #observations must be integer
        self.obs = np.sort(self.obs) #ensure observations are sorted
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
        self.__updatelikelihood() #compute the log-likelihood
        self.__countParameters()    
            
    def __initrandom(self):
        #initialize with a random DPH distribution
        
        #make a random initial distribution (pi)
        nzidx = np.nonzero(self.pi)[1]
        u = np.random.uniform(low=0.0,high=1.0,size=len(nzidx))
        u = u/np.sum(u)
        self.pi[0,nzidx] = u
        
        #make a random exit vector
        nzidx = np.nonzero(self.exitrates)[0]
        u = np.random.uniform(low=0.0, high=1.0, size=len(nzidx))
        self.exitrates[nzidx,0] = u
        
        #make a random PH generator
        for i in range(self.nphases):
            nzidx = np.nonzero(self.phgen[i,:])[1]
            u = np.random.uniform(low=0.0, high=1.0, size=len(nzidx))
            u = (u / np.sum(u)) * (1 - self.exitrates[i, 0])
            self.phgen[i,nzidx] = u
        
    def __estep(self):
        #performs the E-step
        
        self.bi = np.zeros(self.nphases)
        self.ni = np.zeros(self.nphases)
        self.nij = np.zeros((self.nphases,self.nphases))
        self.phgeninv = np.linalg.inv(self.phgen) #for the computation of 'K'
        
        y0 = -1
        for y in self.obs:    
            if y!=y0: #reuse computations if possible
                if y==1:
                    Tpow = self.identity
                    Ttprod = self.exitrates
                else:
                    Tpow = np.linalg.matrix_power(self.phgen,(y-1))
                    Ttprod = np.matmul(Tpow,self.exitrates)
                piTtprod = np.matmul(self.pi,Ttprod)
                
                if y>=2:
                    #pre-compute the matrix function 'K'
                    self.__Kmatrix(y)
            y0=y #update y0
            
            if piTtprod!=0.0:
                for i in range(self.nphases):
                
                    #compute B_i (expected number of times starting in phase i)
                    self.bi[i] += self.pi[0,i]*Ttprod[i,0] / piTtprod
                    #compute N_i (expected number of transitions to absorbing state from phase i)
                    piTpow = np.matmul(self.pi,Tpow)
                    self.ni[i] += piTpow[0,i]*self.exitrates[i,0] / piTtprod
                
                    if y>=2:
                        #compute N_ij (expected transitions between phase i and j)
                        for j in range(self.nphases):
                            self.nij[i,j] += self.phgen[i,j]*self.Kmat[j,i] / piTtprod
        
    def __mstep(self):
        #performs the M-step
        
        #update the initial distribution
        for i in range(self.nphases):
            self.pi[0,i] = self.bi[i]/len(self.obs)
            
        #update the exit rates
        for i in range(self.nphases):
            sm = self.ni[i]
            for j in range(self.nphases):
                sm += self.nij[i,j]
            self.exitrates[i,0] = self.ni[i]/sm

        #update the PH generator
        for i in range(self.nphases):
            for j in range(self.nphases):
                sm = self.ni[i]
                for k in range(self.nphases):
                    sm += self.nij[i,k]
                self.phgen[i,j] = self.nij[i,j]/sm
    
    def __updatelikelihood(self):
        self.loglikelihood = 0.0
        for y in self.obs:
            self.loglikelihood += np.log(self.__getProbMass(y)).item()
    
    def __getProbMass(self,y):
        return np.matmul(self.pi,np.matmul(np.linalg.matrix_power(self.phgen,y),self.exitrates))
    
    def __Kmatrix(self,y):
        #computes the matrix-function 'K' for
        #the observations 'y'
        
        if y<2:
            return

        self.Kmat = np.zeros((self.nphases, self.nphases))

        Ty = np.linalg.matrix_power(self.phgen,y-2)
        exit_prod = np.matmul(Ty,self.exitrates)
        pi_mat = self.pi
        
        for k in range(y-1):
            self.Kmat += np.outer(exit_prod,pi_mat)
            if k<y-2:
                Ty = np.matmul(Ty,self.phgeninv)  
                exit_prod = np.matmul(Ty,self.exitrates)
                pi_mat = np.matmul(pi_mat,self.phgen)  

    def __countParameters(self):
        #count the number of independent parameters
        phg = 0
        for i in range(self.nphases): #independent parameters in each phase of the PH generator and exit vector
            phg += np.count_nonzero(self.phgen[i,:])+np.count_nonzero(self.exitrates[i,0])-1
        self.nparam = phg+(np.count_nonzero(self.pi)-1) #add the number of independent parameters in the initial distribution