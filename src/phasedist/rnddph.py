import numpy as np


class rnddph:
    """
    Generates a random Discrete-time Phase-type (DPH) Distribution with
    a specified structure.
    """

    def __init__(
        self,
        nphases: int,
        initdist: np.array,
        phgen: np.array,
        exitrates: np.array
    ) -> None:
        """
        Initializes the class and specifies the structure of the DPH distribution to be generated.
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
            nphases (int): Number of phases in the DPH distribution.
            initdist (ndarray): Specifies structure of the initial distribution vector.
            phgen (ndarray): Specifies structure of the phase-type generator.
            exitrates (ndarray): Specifies structure of the exit-rate vector.

        Raises:
            ValueError: If no DPH can be generated from the structure.
        """
        self.nphases=nphases
        self.initdist = initdist
        self.phgen = phgen
        self.exitrates = exitrates

        self.__checkinputs()

        return None

    def __checkinputs(self) -> None:
        """
        Checks the structure and converts it to floating-point arrays.

        Args:
            None

        Raises:
            ValueError: If no DPH can be generated from the structure.
        """

        if not isinstance(self.nphases, (int, np.integer)) or self.nphases < 1:
            raise ValueError("The number of phases must be an integer larger than zero.")

        self.initdist = np.asarray(self.initdist, dtype=float).reshape(-1)
        self.phgen = np.asarray(self.phgen, dtype=float)
        self.exitrates = np.asarray(self.exitrates, dtype=float).reshape(-1)

        if (self.initdist.size != self.nphases
                or self.exitrates.size != self.nphases
                or self.phgen.shape != (self.nphases, self.nphases)):
            raise ValueError("The structure must be specified at dimension %d throughout." % self.nphases)

        if not (self.initdist.min() >= 0.0 and self.initdist.max() < np.inf
                and self.phgen.min() >= 0.0 and self.phgen.max() < np.inf
                and self.exitrates.min() >= 0.0 and self.exitrates.max() < np.inf):
            raise ValueError("The structure must be non-negative and finite.")

        # the generated vector is normalized over its non-zero elements
        if not self.initdist.max() > 0.0:
            raise ValueError("The initial distribution must have at least one non-zero element.")

        # a row is scaled to one minus the exit probability, so a phase with
        # neither is never left. The whole row counts, since a self-loop is a
        # transition in the discrete case
        if np.any((self.phgen.sum(axis=1) == 0.0) & (self.exitrates == 0.0)):
            raise ValueError("Every phase must have either a transition or a non-zero exit probability.")

        return None

    def run(self):
        """
        Generates all parameters of the DPH distribution.

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

        #generate the phase-type generator using the newly generated exit rates
        self.__genphgen(self.new_exitrates)

        return self.new_initdist, self.new_phgen, self.new_exitrates
    
    def __geninitdist(self):
        """
        Generates the initial distribution.

        Args:
            None

        Returns:
            None
        """

        #initialize the output vector. The structure may be specified with integers,
        #so the copy is made floating point to avoid truncating the sampled values
        self.new_initdist = np.array(self.initdist, dtype=float)

        #get indices of the non-zero elements
        nzidx = np.nonzero(self.initdist)

        #sample numbers. np.nonzero returns a tuple of index arrays, so the number of
        #non-zero elements is nzidx[0].size and not len(nzidx)
        u = np.random.uniform(low=0.0, high=1.0, size=nzidx[0].size)

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

        #initialize the output matrix. The structure may be specified with integers,
        #so the copy is made floating point to avoid truncating the sampled values
        self.new_phgen = np.array(self.phgen, dtype=float)

        #generate matrix one row (i.e. phase) at a time
        for i in range(self.nphases):

            #get indices of the non-zero elements
            nzidx = np.nonzero(np.ravel(self.phgen[i, :]))

            #a phase with no transitions to other transient phases leaves an empty row
            if nzidx[0].size == 0:
                continue

            #sample numbers
            u = np.random.uniform(low=0.0, high=1.0, size=nzidx[0].size)

            #scale to value in exitrate vector
            u = (u / np.sum(u)) * (1.0 - exitrates[i])

            #insert in matrix
            self.new_phgen[i, nzidx] = u

        return None
    
    def __genexitrates(self):
        """
        Generates the exit-rate vector.

        Args:
            None

        Returns:
            None
        """

        #initialize the output vector. The structure may be specified with integers,
        #so the copy is made floating point to avoid truncating the sampled values
        self.new_exitrates = np.array(self.exitrates, dtype=float)

        #get indices of the non-zero elements
        nzidx = np.nonzero(self.exitrates)

        #sample numbers. np.nonzero returns a tuple of index arrays, so the number of
        #non-zero elements is nzidx[0].size and not len(nzidx)
        u = np.random.uniform(low=0.0, high=1.0, size=nzidx[0].size)
        self.new_exitrates[nzidx] = u

        return None