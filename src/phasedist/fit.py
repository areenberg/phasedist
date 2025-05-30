import numpy as np
import matplotlib.pyplot as plt
from phasedist.fitcph import fitcph
from phasedist.fitdph import fitdph

class fit:
    #fit continuous or discrete-time phase-type
    #distributions 

    def __init__(self,obs=None,nphases=2,dtype="general",discrete=False,
                 initdist=None,initphgen=None,initexitrates=None,
                 randominit=True,seed=None,tolerance=1e-6,itermax=1e6,verbose=False):
        
        #set parameters
        self.obs=obs
        self.nphases=nphases
        self.dtype=dtype
        self.discrete=discrete
        self.initdist=initdist
        self.initphgen=initphgen
        self.initexitrates=initexitrates
        self.randominit=randominit
        self.seed=seed
        self.tolerance=tolerance
        self.itermax=itermax
        self.verbose=verbose
        
        #checking and fitting
        if self.__checkinputs():
            self.__fitdist() #fit the parameters


    def getinitdist(self):
        #returns the initial distribution
        return self.d.getinitdist()
    
    def getphasegen(self):
        #returns the phase-type generator
        return self.d.getphasegen()
    
    def getexitrates(self):
        #returns the exit rate vector
        return self.d.getexitrates() 
            
    def getmean(self):
        #returns the mean
        return self.d.getmean()

    def getvar(self):
        #returns the variance
        return self.d.getvar()

    def getdensity(self,x):
        #returns the density
        return self.d.getdensity(x)

    def getcumprob(self,x):
        #returns the cumulated probability P(X<=x)
        return self.d.getcumprob(x)

    def getloglik(self):
        return self.d.getloglik()

    def getaic(self):
        #returns the Akaike's Information Criteria (AIC)
        return self.d.getaic()

    def getbic(self):
        #returns the Bayesian Information Criteria (BIC)
        return self.d.getbic()
    
    def plot(self):
        #compares the empirical and theoretical CDFs in a plot
        
        obssorted = np.sort(self.obs)
        empcdf = np.arange(1,len(self.obs) + 1)/len(self.obs)
        
        if not self.discrete:
            res = 1000
            theocdf = np.zeros(res)
            x = np.linspace(np.min(obssorted),np.max(obssorted),res)
            for i in range(res):
                theocdf[i] = self.getcumprob(x[i])
        else:
            x = np.arange(np.min(obssorted),np.max(obssorted)+1)
            theocdf = np.zeros(x.size)
            for i in range(x.size):
                theocdf[i] = self.getcumprob(x[i])
            
        plt.figure(figsize=(8, 6))
        
        if not self.discrete:
            plt.plot(x,theocdf,label="Fitted CDF",lw=1,linestyle='-',color='blue')
            plt.plot(obssorted,empcdf,label="Empirical CDF",lw=1,linestyle='-',color='red')
        else:
            plt.scatter(x,theocdf,label="Fitted CDF",lw=1,marker='o',color='blue')
            plt.plot(obssorted,empcdf,label="Empirical CDF",lw=1,linestyle='-',color='red')
            
        plt.xlabel("x")
        plt.ylabel("CDF")
        plt.title("Empirical and Fitted CDFs")
        plt.legend()
        plt.grid()
        plt.show()
        
    def __checkinputs(self):
        #check the feasibility of all input parameters
        #before proceeding
        
        #check data types and convert if necesarry
        if isinstance(self.obs,list):
            self.obs = np.array(self.obs)
        elif not isinstance(self.obs,np.ndarray): 
            print("Error: Observations can only be specified as a list or a NumPy array.")
            return False
        if not isinstance(self.nphases,int) or self.nphases<1:
            print("Error: The number of phases can only be specified as an integer larger than 0.")
            return 0    
        if not isinstance(self.dtype,str):
            print("Error: The distribution type can only be specified as a string.")
        if not isinstance(self.discrete,bool):
            print("Error: The argument 'discrete' needs to be of type 'bool'.")
        if self.initdist is not None and (isinstance(self.initdist,np.ndarray) or isinstance(self.initdist,list)):
            self.initdist = np.matrix(self.initdist)
        elif self.initdist is not None and not isinstance(self.initdist,np.matrix):
            print("Error: The initial distribution can only be specified as a list, NumPy array, or a NumPy matrix.")
            return False
        if self.initphgen is not None and (isinstance(self.initphgen,np.ndarray) or isinstance(self.initphgen,list)):
            self.initphgen = np.matrix(self.initphgen)
        elif self.initphgen is not None and not isinstance(self.initphgen,np.matrix):
            print("Error: The PH generator can only be specified as a list or a NumPy matrix.")
            return False
        if self.initexitrates is not None and (isinstance(self.initexitrates,np.ndarray) or isinstance(self.initexitrates,list)):
            self.initexitrates = np.transpose(np.matrix(self.initexitrates))
        elif self.initexitrates is not None and not isinstance(self.initexitrates,np.matrix):
            print("Error: The exit rate vector can only be specified as a list, NumPy array, or a NumPy matrix.")
            return False
        if not isinstance(self.randominit,bool):
            print("Error: The argument 'randominit' needs to be of type 'bool'.")
            return False        
        if self.seed is not None and not isinstance(self.seed,int):
            print("Error: The seed can only be specified as an integer.")
            return False
        if not isinstance(self.tolerance,float):
            print("Error: The argument 'tolerance' needs to be of type 'float'.")
            return False        
        if not isinstance(self.itermax,float) and not isinstance(self.itermax,int):
            print("Error: The argument 'itermax' needs to be of type 'float' or 'int'.")
            return False        
        if not isinstance(self.verbose,bool):
            print("Error: The argument 'verbose' needs to be of type 'bool'.")
            return False
        if self.discrete and self.dtype=="general":
            print("Error: The 'general' option is currently inactivated for discrete PH distributions.")
            return False
        
        #check observations are equal to or greather than zero
        if np.any(self.obs<0):
            print("Error: The array of observations contains negative values.")
            return False
        
        #check PH generator and exit rates in case of no random initialization
        if self.dtype=="custom" or not self.randominit:
            self.nphases = self.initphgen.shape[0]
        if not self.randominit:
            if not self.__correctphgen(self.initphgen,self.initexitrates) or not self.__correctinitdist(self.d.getinitdist):
                return False
        
        return True
    
    def __fitdist(self):
        
        #set distribution type
        if self.dtype=="general":
            self.__general()
        elif self.dtype=="generlang":
            self.__generlang()
        elif self.dtype=="hyperexp":
            self.__hyperexp()
        elif self.dtype=="coxian":
            self.__coxian()
        elif self.dtype=="gencoxian":
            self.__gencoxian()
        elif self.dtype!="custom":
            print("Error: Unknown distribution type.")
            return 1
        
        #check for zeros in observations
        obsnonzero,fraczero = self.__checkzeros()
        
        #fit parameters
        if self.discrete:
            self.d = fitdph(obs=obsnonzero,initpi=self.initdist,initphgen=self.initphgen,
                            initexitrates=self.initexitrates,randominit=self.randominit,seed=self.seed,
                            tolerance=self.tolerance,itermax=self.itermax,verbose=self.verbose)
        else:
            
            self.d = fitcph(obs=obsnonzero,initpi=self.initdist,initphgen=self.initphgen,
                            initexitrates=self.initexitrates,randominit=self.randominit,seed=self.seed,
                            tolerance=self.tolerance,itermax=self.itermax,verbose=self.verbose)
        self.d.fit()
        
        #check fitted parameters
        self.__checkfit()
        
        #adjust for zeros in observations
        self.d.initpi = self.d.initpi*(1-fraczero)
        
        if self.fitaccepted:    
            return 0
        else:
            print("The PH distribution might contain infeasible or inaccurate parameters.")
            return 1

    def __general(self):
        #general phase-type distribution
        self.initdist=np.matrix(np.ones((1,self.nphases)))
        self.initphgen=np.matrix(np.ones((self.nphases,self.nphases)))
        self.initexitrates=np.matrix(np.ones((self.nphases,1)))

    def __generlang(self):
        #generalized Erlang distribution        
        self.initdist=np.matrix(np.zeros((1,self.nphases)))
        self.initdist[0,0]=1
        
        self.initexitrates=np.matrix(np.zeros((self.nphases,1)))
        self.initexitrates[self.nphases-1,0]=1
        
        self.initphgen=np.matrix(np.zeros((self.nphases,self.nphases)))
        for i in range(self.nphases):
            self.initphgen[i,i]=1
            if i<(self.nphases-1):
                self.initphgen[i,i+1]=1
            
    def __hyperexp(self):
        #hyper-exponential distribution
        self.initdist=np.matrix(np.ones((1,self.nphases)))
        
        self.initphgen=np.matrix(np.zeros((self.nphases,self.nphases)))
        self.initexitrates=np.matrix(np.ones((self.nphases,1)))
        for i in range(self.nphases):
            self.initphgen[i,i]=1
        
    def __coxian(self):
        #Coxian distribution
        self.initdist=np.matrix(np.zeros((1,self.nphases)))
        self.initdist[0,0]=1
        
        self.initexitrates=np.matrix(np.ones((self.nphases,1)))
        
        self.initphgen=np.matrix(np.zeros((self.nphases,self.nphases)))
        for i in range(self.nphases):
            self.initphgen[i,i]=1
            if i<(self.nphases-1):
                self.initphgen[i,i+1]=1

    def __gencoxian(self):
        #generalized Coxian distribution
        self.initdist=np.matrix(np.ones((1,self.nphases)))
        self.initexitrates=np.matrix(np.ones((self.nphases,1)))
        
        self.initphgen=np.matrix(np.zeros((self.nphases,self.nphases)))
        for i in range(self.nphases):
            self.initphgen[i,i]=1
            if i<(self.nphases-1):
                self.initphgen[i,i+1]=1
             
    def __checkzeros(self):
        #check for zeros in observations
        nz = np.nonzero(self.obs)[0]
        return self.obs[nz],1-(nz.size/self.obs.size)
    
    def __checkfit(self):
        #check feasibility of fitted parameters
        self.fitaccepted = True
        if self.__correctphgen(self.d.getphasegen(),self.d.getexitrates()) and self.__correctinitdist(self.d.getinitdist()):
            self.fitaccepted = True
        else:
            self.fitaccepted = False
            
    def __correctphgen(self,phasegen,exitrates):
        #returns True if the PH generator and
        #exit rates are feasible
        
        #check PH generator
        if phasegen.shape[0]!=phasegen.shape[1] or phasegen.shape[0]!=self.nphases:
            print("Error: The dimensions of the PH generator does not match the number of phases.")
            return False
        if np.where(np.isnan(phasegen))[0].size>0 or np.where(np.isinf(phasegen))[0].size>0 or np.where(np.isneginf(phasegen))[0].size>0:
            print("Error: The PH generator contains NaN or/and infinity values.")
            return False
        if np.any(phasegen[~np.eye(self.nphases,dtype=bool)]<0):
            print("Error: The PH generator contains negative off-diagonal values.")
            return False
        if not self.discrete and np.any(phasegen[np.eye(self.nphases,dtype=bool)]>0):
            print("Error: The PH generator contains positive diagonal values.")
            return False
        if not self.discrete and np.max(abs(np.add(np.sum(phasegen,axis=1),exitrates)))>1e-6:
            print("Warning: An element of the exit rate vector deviates at least 1e-6 from the absolute row sum of the PH generator.")
            return True
        #check exit rates
        if self.nphases!=exitrates.size:
            print("Error: The size of the exit rate vector does not match the number of phases.")
            return False
        if np.where(np.isnan(exitrates))[0].size>0 or np.where(np.isinf(exitrates))[0].size>0 or np.where(np.isneginf(exitrates))[0].size>0:
            print("Error: The exit rate vector contains NaN or/and infinity values.")
            return False
        if np.any(exitrates<0):
            print("Error: The exit rate vector contains negative values.")
            return False
        return True

    def __correctinitdist(self,initdist):
        #returns True if the initial distribution
        #is feasible
        
        if initdist.size!=self.nphases:
            print("Error: The size of the initial distribution does not match the number of phases.")
            return False
        if np.where(np.isnan(initdist))[0].size>0 or np.where(np.isinf(initdist))[0].size>0 or np.where(np.isneginf(initdist))[0].size>0:
            print("Error: The initial distribution contains NaN or/and infinity values.")
            return False
        if np.any(initdist<0.0):
            print("Error: The initial distribution contains negative values.")
            return False
        if np.abs(np.sum(initdist)-1.0)>1e-14:
            print("Warning: Prior to adjusting for zeros in the observations the initial distribution summed to " + str(np.sum(initdist)))
            return False
        return True