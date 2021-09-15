#!/usr/bin/env python3
import numpy as np
import numpy.random as random
import numpy.linalg as la

class PCA:
    def __init__(self, X):
        self.N = X.shape[0]
        self.M = X.shape[1]
        self.Ntot = self.allsum(self.N)
        # mean of the data
        self.m = self.allsum( X.sum(0) ) / self.Ntot

        # variance-covariance matrix
        C = np.dot((X-self.m).T, (X-self.m))
        self.C = self.allsum(C) / self.Ntot
        val, vec = la.eigh(self.C)

        # scale of each direction
        self.scale = np.sqrt(val)
        # variance-covariance directions
        self.vec = vec
    def allsum(self, x):
        return x

def randrot(i,j,N):
    assert i >= 0 and j >= 0 and i < N and j < N
    assert i != j
    c = random.uniform()
    s = np.sqrt(1.0-c*c)
    R = np.identity(N)
    R[i,i] = c
    R[i,j] = s
    R[j,i] = -s
    R[j,j] = c
    return R

def test1(N, M):
    # Cholesky decompose of covariance matrix
    # positive eigenvalues uniformly in [10,90]
    val = random.random(M)*90.0 + 10.0
    L = np.diag( val )
    val = np.sort(val)
    print(val)
    for i in range(M):
        for j in range(i+1,M):
            L = np.dot(randrot(i,j,M), L)

    # local samples
    X = np.dot(random.standard_normal((N,M)), L.T)
    S = PCA(X)
    # These 2 should be similar
    if M <= 5:
        print(S.C)
        print(np.dot(L, L.T))
    print(S.scale)

if __name__=="__main__":
    test1(1000, 5)
    #test1(1000, 100)
