import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg as sc_la
from bitstring import BitArray

class MasslessSchwingerED(object) :

    #N=system size, szt=total spin sector we are interested in,
    #l0 = field to the left of system, x = coefficient on hopping term,
    #lam = coefficient on interaction terms, random onsite potential
    #chosen from uniform distribution [-theta, theta].

    def __init__(self, N, szt, l0, x, lam, theta, V = None) :
        #Collection of BitArrays
        self.basis = []

        self.N = N
        self.szt = szt
        self.x = x
        self.lam = lam
        self.l0 = l0
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

    def innerProd(state1, state2) :
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
    def Zi(self, state, i) :
        if state[i] == False :
            return -1.0
        return 1.0

    def buildHamiltonian(self) :
        #print('Started building matrix.')
        minusH = sp.lil_matrix((self.dim, self.dim))
        #construct matrix elements
        N = self.N
        q, p, k = np.zeros(N), np.zeros(N), np.zeros(N)
        Neven = N - 2
        if (N - 2) % 2 == 1 :
            Neven = Neven - 1

        q[0] = Neven / 2 + 1 
        p[0] = N - 1  
        for j in range(1, N - 1) :
            k[j] = N - 1 - j
            p[j] = N - 1 - j
            if (j % 2) == 0 :
                q[j] = q[j - 1]
            else :
                q[j] = q[j - 1] - 1

        if Neven < N - 2 :
            q[N - 2] = 0 

        s = 0.5 * q + self.l0 * p

        # TWO BODY TERMS
        for n in range(0, self.dim) :
            state = self.basis[n]
            for place in reversed(range(1, N)) :
                if state[place] == True and state[place - 1] == False :
                    m = self.findIndex(MasslessSchwingerED.swap(state, place, place - 1))
                    minusH[m,n] = -self.x

        for n in range(0, self.dim) :
            state = self.basis[n]
            for place in range(0, N - 1) :
                if state[place] == True and state[place + 1] == False :
                    m = self.findIndex(MasslessSchwingerED.swap(state, place, place + 1))
                    minusH[m,n] = minusH[m,n] - self.x

        # DIAGONAL TERMS
        for m in range(0, self.dim) :
            factor, disorder_term = 0.0, 0.0
            state = self.basis[m]
            for n in range(0, N) :
                factor = factor + s[n] * self.Zi(state, n)
                disorder_term = disorder_term + self.V[n] * self.Zi(state, n)
            for j in range(1, N - 1) :
                for i in range(0, j) :
                    factor = factor + 0.5 * k[j] * self.ZjZi(state, i, j)
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
                disorder_term = disorder_term + (new_V[n] - self.V[n]) * self.Zi(state, n)
            self.minusH[m,m] = self.minusH[m,m] - 0.5 * disorder_term
        #print('done')
        self.V = new_V

    def partialSpectrum(self, num_of_eigenVals = 10, save_spectrum = False, folder = 'massless_schwinger_ed_data', 
                        compute_states = False, quantifier = '') :
        
        if compute_states :
            w, v = sc_la.eigsh(self.minusH, k = num_of_eigenVals, which = 'LA',
                               return_eigenvectors = True)
            w = -1.0 * w
            if save_spectrum :
                file = MasslessSchwingerED.titleString(self.N, self.szt, self.l0,
                                                       self.x, self.lam, self.theta)
                np.savez(folder + '/' + file + ' ' + quantifier, w, v)
            
            return w, v
        
        w = -1 * sc_la.eigsh(self.minusH, k = num_of_eigenVals, which = 'LA', 
            return_eigenvectors = False)
        #print('Done computing spectrum.')
        if save_spectrum :
            file = MasslessSchwingerED.titleString(self.N, self.szt, self.l0, 
                self.x, self.lam, self.theta)
            np.save(folder + '/' + file + ' ' + quantifier, w)
        return w

    def fullSpectrum(self, save_spectrum = False, folder = 'massless_schwinger_ed_data', quantifier = '') :

        #print('Started computing full spectrum.')
        w = -1 * np.linalg.eigvalsh(self.minusH.todense())
        #print('Done computing full spectrum.')
        if save_spectrum :
            file = MasslessSchwingerED.titleString(self.N, self.szt, self.l0, 
                self.x, self.lam, self.theta)
            np.save(folder + '/' + file + ' ' + quantifier, w)
        return w

    @staticmethod
    def titleString(N, szt, l0, x, lam, theta) :
        return 'N = %d szt = %d l0 = %f x = %f lam = %f theta = %f' % (N, szt, l0, x, lam, theta)
