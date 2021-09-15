#!/usr/bin/env python3

import numpy as np
import numpy.linalg as la
import numpy.random as random

# FIXME: this routine needs to distribute the
# choice of K centers among processors.
def choose_K(X, K, Ntot):
    #if C.rank == 0:
    #else:
    #    m
    ind = random.choice(Ntot, size=K, replace=False)
    return X[ind]

class Kmeans:
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.N = X.shape[0]
        self.Ntot = self.allsum(self.N)
        self.M = X.shape[1]
        self.z = None # category labels
        self.occ = None # total in each category
        self.ctr = choose_K(self.X, K, self.Ntot)
        # total L2 distance within each category
        # (filled in by categorize)
        self.err = None

    def allsum(self, x):
        return x

    def categorize(self, calc_err=True):
        dist = self.X[:,None,:] - self.ctr[None,:,:]
        d2 = np.sum(dist*dist,2)
        self.z = np.argmin(d2, 1)
        if calc_err:
            self.occ = np.zeros(self.K, int)
            self.err = np.zeros(self.K)
            for k in range(self.K):
                self.occ[k] = np.sum(self.z == k)
                self.err[k] = np.sum(d2[self.z == k]**0.5)
            self.occ = self.allsum(self.occ)
            self.err = self.allsum(self.err)

    def estimate(self):
        # column 0 are the occupancies
        loc_ctr = np.zeros((self.K,1+self.M))
        for k in range(self.K):
            m = self.z == k
            loc_ctr[k,1:] = np.dot(m, self.X)
            loc_ctr[k,0] = np.sum(m)
        loc_ctr = self.allsum(loc_ctr)

        self.occ = loc_ctr[:,0]
        self.ctr = loc_ctr[:,1:]/loc_ctr[:,0,None]

    def step(self):
        # On the first iteration, determines labels and exits.
        #
        # For all other iterations, assumes categories (aka labels)
        # are already determined, and does:
        #  1. adjust centroids
        #  2. re-label
        #
        # Returns True when converged within tol
        if self.err is None:
            self.categorize(True)
            return False
        z0 = self.z.copy()
        err = self.err.sum()
        
        self.estimate()
        self.categorize(True)
        err2 = self.err.sum()
        # how many categories changed?
        dz = self.allsum( np.sum(self.z != z0) )
        #if np.abs(err - err2) < tol:
        #    return True
        return dz == 0

    def show(S):
        print(f"Ctr = {S.ctr}")
        print(f"categories = {S.z}")
        if S.occ is not None:
            print(f"Cluster occupancy: {S.occ}")
        if S.err is not None:
            print(f"Average distance to centers: {S.err/(S.occ+(S.occ == 0))}")

def test1(N, M, K):
    sigma = 0.6
    ctrs = np.array([[0.0, 0.0, 0.0],
                     [1.0, 1.0, 1.0],
                     [-1.0, 1.0, 1.0],
                     [-1.0, -0.5, -0.5]])
    assert K <= ctrs.shape[0]
    assert M <= ctrs.shape[1]
    assert N >= K
    z = np.zeros(N)
    X = np.zeros((N,M))
    for i in range(K):
        d = ((i+1)*N)//K - (i*N)//K
        z[(i*N)//K : ((i+1)*N)//K] = i
        X[(i*N)//K : ((i+1)*N)//K] = sigma*random.standard_normal((d,M)) \
                                   + ctrs[i,:M]

    S = Kmeans(X, K)
    for i in range(10):
        done = S.step()
        if done:
            print(f"Converged in {i} steps.")
            break
    else:
        print("Convergence not obtained.")
    S.show()

if __name__=="__main__":
    test1(10, 2, 3)
    test1(100, 3, 3)
    test1(1000, 3, 4)
