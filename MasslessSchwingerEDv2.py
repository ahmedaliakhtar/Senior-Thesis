import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg as sc_la
from bitstring import BitArray

class MasslessSchwingerEDv2(object) :

    #N=system size, szt=total spin sector we are interested in,
    #alpha = background field, x = coefficient on hopping term,
    #lam = coefficient on interaction terms, w/ random onsite potential
    #chosen from uniform distribution [-theta, theta].

    def __init__(self, N, szt, alpha, x, lam, theta, V = None) :
        #Collection of BitArrays
        self.basis = []

        self.N = N
        self.szt = szt
        self.x = x
        self.lam = lam
        self.alpha = alpha
        self.theta = theta
        if V is None :
            self.V = theta * (2 * np.random.rand(N, 1) - 1)
        else :
            self.V = V
        k = (2 * szt + N) // 2
        #construct the basis as a set of bit strings
        startingbit = BitArray('0b0')
        startingbit.clear()
        self.makeString(startingbit, N - k, k)
        self.dim = len(self.basis)

    def makeString(self, curr, a, b) :
        if a == 0 and b == 0 :
            self.basis.append(curr)
        else :
            if a > 0 :
                self.makeString(curr + [0], a - 1, b)
            if b > 0 :
                self.makeString(curr + [1], a, b - 1)

    def printBasis(self) :
        print(self.basis)

    #find the index of a state using Binary Search
    def findIndex(self, state) :
        hi, lo = self.dim - 1, 0
        while (lo <= hi) :
            mid = lo + (hi - lo) // 2
            compareVal = self.compareBitArrays(state, self.basis[mid])
            if  compareVal < 0 :
                hi = mid - 1
            elif compareVal > 0 :
                lo = mid + 1
            else :
                return mid

        return -1

    def compareBitArrays(self, state1, state2) :
        for i in range(0, self.N) :
            if state1[i] == False and state2[i] == True :
                return -1
            elif state1[i] == True and state2[i] == False :
                return 1
        return 0

    def getDimension(self) :
        return self.dim

    def innerProd(self, state1, state2) :
        if self.compareBitArrays(state1, state2) == 0 :
            return 1.0
        else :
            return 0.0

    #given a state, compute ZjZi on the state
    def ZjZi(self, state, j, i) :
        if state[i] == state[j] :
            return 1.0
        return -1.0

    #given a state, compute PjMi
    def PjMi(self, state, j, i) :
        if state[i] == False or state[j] == True :
            return 0.0, state
        new_state = MasslessSchwingerED.swap(state, j, i)
        return 1.0, new_state

    #return the state with bits i and j swapped
    @staticmethod
    def swap(state, i, j) :
        c1 = state[i]
        c2 = state[j]
        state = BitArray(state)
        state.set(c1, j)
        state.set(c2, i)
        return state

    #given a state, compute Zi
    @staticmethod
    def Zi(state, i) :
        if state[i] == False :
            return -1.0
        return 1.0

    def buildHamiltonian(self) :
        #print('Started building matrix.')
        minusH = sp.lil_matrix((self.dim, self.dim))
        #construct matrix elements
        N = self.N

        # TWO BODY TERMS
        for n in range(0, self.dim) :
            state = self.basis[n]
            for place in reversed(range(1, N)) :
                if state[place] == True and state[place - 1] == False :
                    m = self.findIndex(MasslessSchwingerEDv2.swap(state, place, place - 1))
                    minusH[m,n] = -self.x

        for n in range(0, self.dim) :
            state = self.basis[n]
            for place in range(0, N - 1) :
                if state[place] == True and state[place + 1] == False :
                    m = self.findIndex(MasslessSchwingerEDv2.swap(state, place, place + 1))
                    minusH[m,n] = minusH[m,n] - self.x

        # DIAGONAL TERMS
        for m in range(0, self.dim) :
            factor, disorder_term = 0.0, 0.0
            state = self.basis[m]
            for n in range(0, N) :
                disorder_term = disorder_term + self.V[n] * MasslessSchwingerEDv2.Zi(state, n)
                if n < N - 1 :
                    term = self.alpha
                    for k in range(0, N) :
                        if k <= n :
                            term += 0.25 * MasslessSchwingerEDv2.Zi(state, k)
                        else :
                            term -= 0.25 * MasslessSchwingerEDv2.Zi(state, k)
                    factor += term * term
            minusH[m,m] = minusH[m,m] - self.lam * factor - 0.5 * disorder_term

        self.minusH = minusH
        self.minusH.tocsc()
        #print('-H computed.')

    def newDisorder(self) :
        #print('use different disorder')
        #construct a new disorder
        new_V = self.theta * (2 * np.random.rand(self.N, 1) - 1)

        #update matrix
        for m in range(0, self.dim) :
            disorder_term = 0.0
            state = self.basis[m]
            for n in range(0, self.N) :
                disorder_term = disorder_term + (new_V[n] - self.V[n]) * MasslessSchwingerEDv2.Zi(state, n)
            self.minusH[m,m] = self.minusH[m,m] - 0.5 * disorder_term

        #print('done')
        self.V = new_V

    def partialSpectrum(self, num_of_eigenVals = 10, save_spectrum = False, folder = 'massless_schwinger_ed_data_2', quantifier = '') :

        #print('Started computing spectrum.')
        w = -1 * sc_la.eigsh(self.minusH, k = num_of_eigenVals, which = 'LA', 
            return_eigenvectors = False)
        #print('Done computing spectrum.')
        if save_spectrum :
            file = MasslessSchwingerEDv2.titleString(self.N, self.szt, self.alpha, 
                self.x, self.lam, self.theta)
            np.save(folder + '/' + file + ' ' + quantifier, w)
        return w
    
    def getGroundState(self) :
        
        w, v = -1 * sc_la.eigsh(self.minusH, k = 1, which = 'LA', return_eigenvectors = True)
        return v[:,0]

    def calculateSzi(self, state) :
        
        Szi = np.zeros(self.N)
        for n in range(0, self.N) :
            for m in range(0, self.dim) :
                Szi[n] += self.basis[m][n] * state[m] * np.conj(state[m])
        
        return Szi

    def fullSpectrum(self, save_spectrum = False, folder = 'massless_schwinger_ed_data_2', quantifier = '') :

        #print('Started computing full spectrum.')
        w = -1 * np.linalg.eigvalsh(self.minusH.todense())
        #print('Done computing full spectrum.')
        if save_spectrum :
            file = MasslessSchwingerEDv2.titleString(self.N, self.szt, self.alpha, 
                self.x, self.lam, self.theta)
            np.save(folder + '/' + file + ' ' + quantifier, w)
        return w

    @staticmethod
    def titleString(N, szt, alpha, x, lam, theta) :
        return 'N = %d szt = %d alpha = %f x = %f lam = %f theta = %f' % (N, szt, alpha, x, lam, theta)
