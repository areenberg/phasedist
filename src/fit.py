import numpy as np

class fit:
    #fit continuous or discrete-time phase-type
    #distributions 

    def __init__(self,obs=None,nphases=2,dtype="general",discrete=False,
                 initdist=None,initphgen=None,initexitrates=None,
                 randominit=True,seed=None,tolerance=1e-6,itermax=1e6):
        
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
        
        #checking and fitting
        self.__checkparameters()
        self.__fitdist()
        
    def __checkparameters(self):
        return 0
    
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
        
        #fit parameters
        self.d =         
        
        return 0

    def __general(self):
        #general phase-type distribution
        self.initdist=np.ones((1,self.nphases))
        self.initphgen=np.ones((self.nphases,self.nphases))
        self.initexitrates=np.ones((self.nphases,1))

    def __generlang(self):
        #generalized Erlang distribution        
        self.initdist=np.zeros((1,self.nphases))
        self.initdist[0,0]=1
        
        self.initexitrates=np.zeros((self.nphases,1))
        self.initexitrates[self.nphases-1,0]=1
        
        self.initphgen=np.zeros((self.nphases,self.nphases))
        for i in range(self.nphases):
            self.initphgen[i,i]=1
            if i<(self.nphases-1):
                self.initphgen[i,i+1]=1
            
    def __hyperexp(self):
        #hyper-exponential distribution
        self.initdist=np.ones((1,self.nphases))
        
        self.initphgen=np.zeros((self.nphases,self.nphases))
        self.initexitrates=np.ones((self.nphases,1))
        for i in range(self.nphases):
            self.initphgen[i,i]=1
        
    def __coxian(self):
        #Coxian distribution
        self.initdist=np.zeros((1,self.nphases))
        self.initdist[0,0]=1
        
        self.initexitrates=np.ones((self.nphases,1))
        
        self.initphgen=np.zeros((self.nphases,self.nphases))
        for i in range(self.nphases):
            self.initphgen[i,i]=1
            if i<(self.nphases-1):
                self.initphgen[i,i+1]=1

    def __gencoxian(self):
        #generalized Coxian distribution
        self.initdist=np.ones((1,self.nphases))
        self.initexitrates=np.ones((self.nphases,1))
        
        self.initphgen=np.zeros((self.nphases,self.nphases))
        for i in range(self.nphases):
            self.initphgen[i,i]=1
            if i<(self.nphases-1):
                self.initphgen[i,i+1]=1
        