import numpy as np


class rndcph:
    """
    Generates a random Continuous-time Phase-type (CPH) Distribution with
    a specified structure.
    Note: This class has no input checks.
    """

    def __init__(
        self,
        nphases: int,
        initdist: np.array,
        phgen: np.array,
        exitrates: np.array
    ) -> None:
        """
        Initializes the class and specifies the structure of the CPH distribution to be generated.
        The structure is specified by setting the elements intended to be zero to zero in the
        input parameters and vice versa for the intended non-zero elements.

        For instance, to generate a three-phase hyper-exponential distribution, use:

        nphases = 3
        initdist = [1,1,1]
        phgen = [[1,0,0],
                 [0,1,0],
                 [0,0,1]]
        exitrates = [1,1,1]         

        Args:
            nphases (int): Number of phases in the CPH distribution.
            initdist (ndarray): Specifies structure of the initial distribution vector.
            phgen (ndarray): Specifies structure of the phase-type generator.
            exitrates (ndarray): Specifies structure of the exit-rate vector. 
        """
        self.nphases=nphases
        self.initdist = initdist
        self.phgen = phgen 
        self.exitrates = exitrates

        return None

    def run(self):
        """
        Generates all parameters of the CPH distribution.

        Args:
            None

        Returns:
            self.new_initdist (ndarray): The generated random initial distribution vector.
            self.new_phgen (ndarray): The generated random phase-type generator.
            self.new_exitrates (ndarray): The generated random exit-rate vector.
        """

        #generate the initial distribution
        self.__geninitdist()

        #generate the exitrate vector
        self.__genexitrates()

        #generate the phase-type distribution
        self.__genphgen(self.exitrates)

        return self.new_initdist, self.new_phgen, self.new_exitrates
    
    def __geninitdist(self):
        """
        Generates the initial distribution.

        Args:
            None

        Returns:
            None
        """

        #initialize the output vector
        self.new_initdist = np.copy(self.initdist)

        #get indices of the non-zero elements
        nzidx = np.nonzero(self.pi)

        #sample numbers
        u = np.random.uniform(low=0.0, high=1.0, size=len(nzidx))

        #normalize
        u = u / np.sum(u)
        self.new_initdist[nzidx] = u

        return None
    
    def __genphgen(self,exitrates):
        """
        Generates the phase-type generator.

        Args:
            exitrates (ndarray): An exit-rate vector.

        Returns:
            None
        """

        #initialize the output matrix
        self.new_phgen = np.copy(self.phgen)

        #generate matrix one row (i.e. phase) at a time
        for i in range(self.nphases):

            #get indices of the non-zero elements
            nzidx = np.nonzero(np.ravel(self.phgen[i, :]))[0]
            msk = nzidx != i
            nzidx = nzidx[msk]

            #sample numbers
            u = np.random.uniform(low=0.0, high=1.0, size=len(nzidx))
            self.new_phgen[i, nzidx] = u

            #adjust diagonal to the exitrate vector
            self.new_phgen[i, i] = -(np.sum(u) + exitrates[i])

        return None
    
    def __genexitrates(self):
        """
        Generates the exit-rate vector.

        Args:
            None

        Returns:
            None
        """

        #initialize the output vector
        self.new_exitrates = np.copy(self.exitrates)

        #get indices of the non-zero elements
        nzidx = np.nonzero(self.exitrates)
        
        #sample numbers
        u = np.random.uniform(low=0.0, high=1.0, size=len(nzidx))
        self.new_exitrates[nzidx] = u

        return None