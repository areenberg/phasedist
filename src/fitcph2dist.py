import sys
import numpy as np
from scipy.linalg import expm
from scipy.stats import lognorm, norm, gamma, weibull_min, chi2
import matplotlib.pyplot as plt

# REFERENCES

#Bladt, M., & Nielsen, B. F. (2017). Matrix-Exponential Distributions in Applied Probability. 
#Springer. https://doi.org/10.1007/978-1-4939-7049-0

class fitcph2dist:
    #fit a continuous-time phase-type distributions to
    #a distribution with a continuous density using
    #the EM algorithm from p. 681 Bladt and Nielsen (2017).

    def __init__(self,initdist=None,initphgen=None,initexitrates=None,randominit=True,seed=None,tolerance=1e-3,truncation=0.99,steps=50,itermax=1e9,verbose=False):
        self.initpi = initdist
        self.initphgen = initphgen
        self.initexitrates = initexitrates
        self.nphases = self.initphgen.shape[0]
        self.randominit = randominit
        self.seed = seed
        self.itermax = itermax
        self.tolerance=tolerance
        self.truncation=truncation
        self.disttype = None
        self.verbose=verbose
        self.steps = steps #number of steps in the numerical integration
        
    def lognorm(self,mu=None,sigma=None,mean=None,var=None):
        #approximate a log-normal distribution
        if mu is None and mean is not None:
            if mean<=0:
                print("Error: 'mean<=0' is infeasible for the lognormal distribution.")
                sys.exit(1)
            self.param1 = np.log(np.power(mean,2)/np.sqrt(np.power(mean,2)+var))
            self.param2 = np.log(1 + var/np.power(mean,2))
        elif mu is not None and mean is None:
            self.param1 = mu
            self.param2 = np.power(sigma,2)
        self.disttype="lognorm"
        self.__initialize()
        
    def norm(self,mu=None,sigma=None):
        #approximate a (truncated) normal distribution
        self.param1=mu
        self.param2=sigma
        self.disttype="norm"
        self.__initialize()
        
    def gamma(self,shape=None,scale=None,rate=None):
        #approximate a gamma distribution
        self.param1=shape
        if scale is None:
            self.param2=1/rate
        elif rate is None:
            self.param2=scale
        self.disttype="gamma"
        self.__initialize()
    
    def chisq(self,df=None):
        #approximate a chi-squared distribution
        self.param1=df
        self.disttype="chisq"
        self.__initialize()
    
    def weibull(self,shape=None,scale=None):
        #approximate a weibull distribution
        self.param1=shape
        self.param2=scale
        self.disttype="weibull"
        self.__initialize()
        
    def phasedist(self,initdist,phgen):
        #approximate another phase-type distribution
        self.param1=initdist
        self.param2=phgen
        self.disttype="ph"
        self.__initialize()

    def percentiles(self,cumprobs=None,x=None):
        #create a PH approximation based on the
        #cumulated probabilities (could be empirically
        #determined) in the numpy array 'cumprobs' and
        #the corresponding response values in 'x'.
        self.param1=cumprobs
        self.param2=x
        self.disttype="per"
        self.__initialize()
        
    def plot(self):
        #conduct a visual comparison of the approximate and true
        #distributions    
 
        #compute densities for approximate and true distributions        
        x = np.linspace(1e-6,self.y.max(),500)
        dist_pdf = np.zeros(len(x))
        ph_pdf = np.zeros(len(x))
        for i in range(len(x)):
            if self.disttype=="lognorm":
                dist_pdf[i] = lognorm.pdf(x[i],self.param2,scale=np.exp(self.param1))
            elif self.disttype=="gamma":
                dist_pdf[i] = gamma.pdf(x[i],self.param1,scale=self.param2)
            elif self.disttype=="weibull":
                dist_pdf[i] = weibull_min.pdf(x[i],self.param1,scale=self.param2)
            elif self.disttype=="chisq":
                dist_pdf[i] = chi2.pdf(x[i],self.param1)
            elif self.disttype=="ph":
                dist_pdf[i] = np.matmul(self.param1,np.matmul(expm(self.param2*x),abs(np.sum(self.param2,axis=1)))).item()
            ph_pdf[i] = self.getdensity(x[i])
        
        #make plot    
        plt.figure(figsize=(10, 6))
        if not self.disttype=="norm" and not self.disttype=="per":
            plt.plot(x, dist_pdf, label='True density', color='blue')
        plt.plot(x, ph_pdf, label='Approx. density', color='red', linestyle='--')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.title('Approximation validation')
        plt.legend()
        plt.grid(True)
        plt.show()        
        
        return None
        
    def fit(self):
        #approximate the CPH distribution
        if self.disttype is None:
            print("Select a distribution for the approximation.")
            return 0
        iter=0
        self.eps = np.inf
        self.pi0 = np.copy(self.pi)
        self.phgen0 = np.copy(self.phgen)
        while iter<self.itermax and self.eps>self.tolerance:
            self.__estep()
            self.__mstep()
            self.__updateEpsilon()
            self.pi0 = np.copy(self.pi)
            self.phgen0 = np.copy(self.phgen)
            iter += 1
            if self.verbose and iter%5==0:
                print("iter =",iter,"  eps =",self.eps,"  mean =",self.getmean(),"  var =",self.getvar())
            
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

    def __initialize(self):    
        if self.seed is not None:
            np.random.seed(self.seed)
        
        #convert data types
        self.steps = int(self.steps)
        self.initpi = self.initpi.astype(float)
        self.initphgen = self.initphgen.astype(float)
        self.initexitrates = self.initexitrates.astype(float)
        
        #copy to output parameters    
        self.pi = self.initpi      
        self.phgen = self.initphgen
        self.exitrates = self.initexitrates    
        
        #initialize with random parameters
        #accounting for the specified structure
        if self.randominit:
            self.__initrandom()
                    
        #create the cumulated probabilities and evaluation points
        self.__computeyvector()    
            
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
        
        #pre-compute eTy and pieTyt
        self.eTy = [None] * self.steps
        self.eTyut = [[None] * (self.steps+1) for _ in range(self.steps)]
        self.pieTu = [[None] * (self.steps+1) for _ in range(self.steps)]
        self.pieTyt = np.zeros(self.steps)
        for k in range(self.steps):
            self.eTy[k] = expm(self.phgen*self.y[k])
            self.pieTyt[k] = np.matmul(self.pi,np.matmul(self.eTy[k],self.exitrates))
            
        for i in range(self.nphases):
            
            #bi
            for k in range(self.steps):
                eTyt = np.matmul(self.eTy[k],self.exitrates) 
                Gy = ((self.pi[0,i]*eTyt[i,0])/self.pieTyt[k])
                self.bi[i] += Gy*self.hy[k]

            #zi (denominator used in calculation of PH generator and exit rates)
            for k in range(self.steps):
                #inner integral
                innerint=0.0
                if self.y[k]>0:
                    u = np.linspace(0,self.y[k],self.steps+1)
                    for l in range(0,len(u)-2,2):
                        #inner fa
                        self.pieTu[k][l] = np.matmul(self.pi,expm(self.phgen*u[l]))
                        self.eTyut[k][l] = np.matmul(expm(self.phgen*(self.y[k]-u[l])),self.exitrates)
                        inner_fa = self.pieTu[k][l][0,i]*self.eTyut[k][l][i,0]
                        #inner fmid
                        self.pieTu[k][l+1] = np.matmul(self.pi,expm(self.phgen*u[l+1]))
                        self.eTyut[k][l+1] = np.matmul(expm(self.phgen*(self.y[k]-u[l+1])),self.exitrates)
                        inner_fmid = self.pieTu[k][l+1][0,i]*self.eTyut[k][l+1][i,0]
                        #inner fb
                        self.pieTu[k][l+2] = np.matmul(self.pi,expm(self.phgen*u[l+2]))
                        self.eTyut[k][l+2] = np.matmul(expm(self.phgen*(self.y[k]-u[l+2])),self.exitrates)
                        inner_fb = self.pieTu[k][l+2][0,i]*self.eTyut[k][l+2][i,0]
                        
                        innerint += self.__simpsonsrule(u[l],u[l+2],inner_fa,inner_fmid,inner_fb)
                #outer integral    
                Gy = innerint/self.pieTyt[k]
                self.zi[i] += Gy*self.hy[k]
                        
            #ni (numerator used in calculation of exit rates)
            for k in range(self.steps):
                pieTy = np.matmul(self.pi,self.eTy[k])
                Gy = (pieTy[0,i]/self.pieTyt[k])*self.exitrates[i,0]
                self.ni[i] += Gy*self.hy[k]
                    
            for j in range(self.nphases):
                #nij (numerator used in calculation of PH generator)
                if j!=i and self.phgen[i,j]>0.0:
                    for k in range(self.steps):
                        #inner integral
                        innerint=0.0
                        if self.y[k]>0:
                            u = np.linspace(0,self.y[k],self.steps+1)
                            for l in range(0,len(u)-2,2):
                                #inner fa
                                inner_fa = self.pieTu[k][l][0,i]*self.eTyut[k][l][j,0]
                                #inner fmid
                                inner_fmid = self.pieTu[k][l+1][0,i]*self.eTyut[k][l+1][j,0]
                                #inner fb
                                inner_fb = self.pieTu[k][l+2][0,i]*self.eTyut[k][l+2][j,0]

                                innerint += self.__simpsonsrule(u[l],u[l+2],inner_fa,inner_fmid,inner_fb)
                        #outer integral
                        Gy = (self.phgen[i,j]/self.pieTyt[k])*innerint
                        self.nij[i,j] += Gy*self.hy[k]
        
    def __mstep(self):
        #performs the M-step
        
        #update the initial distribution
        for i in range(self.nphases):
            self.pi[0,i] = self.bi[i]
        self.pi = self.pi/np.sum(self.pi)
            
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
                    
        
    def __updateEpsilon(self):
    #get the largest absolute difference in parameters
        #eps1 = np.max(np.abs(np.divide(np.subtract(self.pi[np.nonzero(self.pi0)],self.pi0[np.nonzero(self.pi0)]),self.pi0[np.nonzero(self.pi0)])))
        #eps2 = np.max(np.abs(np.divide(np.subtract(self.phgen[np.nonzero(self.phgen0)],self.phgen0[np.nonzero(self.phgen0)]),self.phgen0[np.nonzero(self.phgen0)])))
        eps1 = np.max(np.abs(np.subtract(self.pi[np.nonzero(self.pi0)],self.pi0[np.nonzero(self.pi0)])))
        eps2 = np.max(np.abs(np.subtract(self.phgen[np.nonzero(self.phgen0)],self.phgen0[np.nonzero(self.phgen0)])))
        self.eps = np.max(np.array([eps1,eps2]))
   
    def __computeyvector(self):
        #pre-compute evaluation points for the
        #numerical integral
        
        #make evaluation points
        if self.disttype=="lognorm":
            self.y = np.linspace(0,lognorm.ppf(self.truncation,self.param2,scale=np.exp(self.param1)),self.steps+1)
        elif self.disttype=="gamma":
            self.y = np.linspace(0,gamma.ppf(self.truncation,self.param1,scale=self.param2),self.steps+1)
        elif self.disttype=="norm":
            self.y = np.linspace(0,self.__normtruncquantfun(self.param1,self.param2),self.steps+1)
        elif self.disttype=="weibull":
            self.y = np.linspace(0,weibull_min.ppf(self.truncation,self.param1,scale=self.param2),self.steps+1)
        elif self.disttype=="chisq":
            self.y = np.linspace(0,chi2.ppf(self.truncation,self.param1),self.steps+1)
        elif self.disttype=="ph":
            self.y = np.linspace(0,self.__phtruncquantfun(self.param1,self.param2),self.steps+1)
        elif self.disttype=="per":
            self.y = np.linspace(0,np.max(self.param2),self.steps+1)
        
        
        #compute cumulated probability segments
        self.hy = np.zeros(self.steps)
        for i in range(self.steps):
            if self.disttype=="lognorm":
                self.hy[i] = self.__lognorm_dcdf(self.y[i],self.y[i+1])
            elif self.disttype=="gamma":
                self.hy[i] = self.__gamma_dcdf(self.y[i],self.y[i+1])
            elif self.disttype=="norm":
                self.hy[i] = self.__normtrunc_dcdf(self.y[i],self.y[i+1])
            elif self.disttype=="weibull":
                self.hy[i] = self.__weib_dcdf(self.y[i],self.y[i+1])
            elif self.disttype=="chisq":
                self.hy[i] = self.__chisq_dcdf(self.y[i],self.y[i+1])
            elif self.disttype=="ph":
                self.hy[i] = self.__ph_dcdf(self.y[i],self.y[i+1])
            elif self.disttype=="per":
                self.hy[i] = self.__per_dcdf(self.y[i],self.y[i+1])
        
        #adjust y's for the E-step
        self.y = self.y[1:]
        
    def __simpsonsrule(self,a,b,fa,fmid,fb):
       #returns Simpson's 1/3 rule
       #note: fmid = f((a+b)/2)
       return ((b-a)/6)*(fa+4*fmid+fb)
       
    def lognormdensity(self,x):
        #PDF of the log-normal distribution
       return (1/(x*np.sqrt(self.param2)*np.sqrt(2*np.pi)))*np.exp(-(np.power(np.log(x)-self.param1,2)/(2*self.param2)))

    def __lognorm_dcdf(self,x0,x1):
        #Cumulated probability *between* x0 and x1
        #of the log-normal distribution
        if x0==0:
            return norm.cdf((np.log(x1)-self.param1)/np.sqrt(self.param2))
        else:
            return norm.cdf((np.log(x1)-self.param1)/np.sqrt(self.param2))-norm.cdf((np.log(x0)-self.param1)/np.sqrt(self.param2))

    def __gamma_dcdf(self,x0,x1):
        #Cumulated probability *between* x0 and x1
        #of the gamma distribution
        if x0==0:
            return gamma.cdf(x1,self.param1,scale=self.param2)
        else:
            return gamma.cdf(x1,self.param1,scale=self.param2)-gamma.cdf(x0,self.param1,scale=self.param2)

    def __normtrunc_dcdf(self,x0,x1):
        #Cumulated probability *between* x0 and x1
        #of the truncated normal distribution
        
        return (norm.cdf((x1-self.param1)/self.param2)-norm.cdf((x0-self.param1)/self.param2))/(1-norm.cdf((-self.param1)/self.param2))

    def __weib_dcdf(self,x0,x1):
        #Cumulated probability *between* x0 and x1
        #of the Weibull distribution
        
        if x0==0:
            return weibull_min.cdf(x1,self.param1,scale=self.param2)
        else:
            return weibull_min.cdf(x1,self.param1,scale=self.param2)-weibull_min.cdf(x0,self.param1,scale=self.param2)
    
    def __chisq_dcdf(self,x0,x1):
        #Cumulated probability *between* x0 and x1
        #of the Chi-square distribution
        
        if x0==0:
            return chi2.cdf(x1,self.param1)
        else:
            return chi2.cdf(x1,self.param1)-chi2.cdf(x0,self.param1)
        
    def __ph_dcdf(self,x0,x1):
        #Cumulated probability *between* x0 and x1
        #of the PH distribution
        
        if x0==0:
            return 1-np.sum(np.matmul(self.param1,expm(self.param2*x1)))
        else:
            return np.sum(np.matmul(self.param1,expm(self.param2*x0)))-np.sum(np.matmul(self.param1,expm(self.param2*x1)))
        
    def __per_dcdf(self,x0,x1):
        #Cumulated probability *between* x0 and x1
        #for percentiles provided by the user
        if x0==0:
            return self.__per_cdf(x1)
        else:
            return self.__per_cdf(x1)-self.__per_cdf(x0)
    
    def __per_cdf(self,x):
        #Cumulated probability for percentiles
        #provided by the user
        idx1 = np.min(np.where(self.param2>=x))
        if idx1>0:
            idx0=idx1-1
            return self.param1[idx0]+(self.param1[idx1]-self.param1[idx0])*((x-self.param2[idx0])/(self.param2[idx1]-self.param2[idx0]))
        else:
            return self.param1[idx1]*(x/self.param2[idx1])

    def __normtruncquantfun(self,mu,sigma):
        #numerical quantile function for the truncated
        #normal distribution
        cmp=1-norm.cdf(-mu/sigma)
        x=np.max(np.array([sigma*self.tolerance,mu]))
        trc=(norm.cdf((x-mu)/sigma)-norm.cdf(-mu/sigma))/cmp
        dd = sigma*self.tolerance
        iter=0
        while np.abs(trc-self.truncation)>self.tolerance and iter<self.itermax:
                x1=x+dd
                f1 = (norm.cdf((x1-mu)/sigma)-norm.cdf(-mu/sigma))/cmp
                grad = (f1-trc)/dd
                x = x - (trc-self.truncation)/grad
                trc = (norm.cdf((x-mu)/sigma)-norm.cdf(-mu/sigma))/cmp
                iter+=1          
        return x            
            
    def __phtruncquantfun(self,idst,phg):
        #numerical quantile function for the
        #PH distribution
        phinv = np.linalg.inv(phg)
        sigma = np.sqrt(2*np.sum(np.matmul(idst,np.linalg.matrix_power(phinv,2)))-np.power(np.sum(np.matmul(idst,phinv)),2))
        mu = -np.sum(np.matmul(idst,phinv))
        x=np.max(np.array([sigma*self.tolerance,mu]))
        trc=1-np.sum(np.matmul(idst,expm(phg*x)))
        dd = sigma*self.tolerance
        iter=0
        while np.abs(trc-self.truncation)>self.tolerance and iter<self.itermax:
                x1=x+dd
                f1 = 1-np.sum(np.matmul(idst,expm(phg*x1)))
                grad = (f1-trc)/dd
                x = x - (trc-self.truncation)/grad
                trc = 1-np.sum(np.matmul(idst,expm(phg*x)))
                iter+=1
        return x            
           
            
            
            