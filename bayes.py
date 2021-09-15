#!/usr/bin/env python3
# Bayesian SVM regression

import numpy as np
import numpy.linalg as la
import numpy.random as random

# Asymptotics:
#   as x ->  infty, softplus(x) -> x      [infty]
#   as x -> -infty, softplus(x) -> exp(x) [0]
def softplus(x):
    y = x.copy()
    m = np.where(y < 10.0)
    if len(m[0]) > 0:
        y[m] = np.log1p(np.exp(x[m]))
    return y

class SVM:
    # X is the (samples x features) matrix of data
    # and z is the categorization vector (len(z)=samples).
    # z[i] should be 0 or 1, depending on the category of i.
    #
    # This class can be called in parallel.
    # In this case, it holds only local rows of X, z.
    # and (intermediate variable) s.
    # Everything else is global.
    def __init__(self, X, z, alpha=0.2, beta=0.8):
        # problem instance
        self.X = X
        self.z = z > 0 # turn into 2-category information
        self.N = X.shape[0]
        self.M = X.shape[1]
        self.Ntot = self.allsum(self.N)

        # solution variables (updated during step)
        # start off pointing at data centroid
        c1 = self.allsum(np.dot(self.z, self.X))/self.Ntot # z1 centroid
        c0 = self.allsum(np.sum(self.X,0))/self.Ntot # data centroid
        c1 = 2*c1-c0 # hyperplane direction
        self.w = c1 / np.dot(c1,c1)**0.5
        self.b = np.dot(self.w, c0)

        # algorithm parameters affecting line-search:
        assert 0.0 < alpha and alpha < 0.5
        assert 0.0 < beta and beta < 1.0
        self.alpha = alpha
        self.beta = beta

        self.Dval = None

        N1 = self.allsum(self.z.sum())
        N0 = self.Ntot - N1
        print(N0, N1)
        # constant shift to log-posterior likelihood
        self.log_const = N0*np.log(N0 + (N0 == 0)) \
                       + N1*np.log(N1 + (N1 == 0)) \
                       - self.Ntot*np.log(self.Ntot)
        self.N0 = N0
        self.N1 = N1

    @property
    def L(self):
        # Report the log-likelihood in per-sample units.
        if self.Dval is None:
            return None
        ans = (self.Dval+self.log_const) / self.Ntot
        #print(ans)
        return ans

    def D(self, w, p):
        # Return the current value of the objective function
        #
        ans = np.dot(p, self.z) \
            - softplus(p).sum()
        ans = self.allsum(ans)

        return ans + 0.5*np.log(np.dot(w,w))

    def calc_p(self, b, w):
        return np.dot(self.X, w) - b

    def calc_ps(self):
        p = self.calc_p(self.b, self.w)
        t = p.clip(None, 0.0)
        s = np.exp(t) / (1.0 + np.exp(2*t-p))
        return p, s

    def step(self, rtol=1e-6, Ltol=1e-9, verb=False):
        # Take a single step of unconstrained
        # Newton's method with backtracking.
        #
        # On output, updates self.b and self.w.
        # Also caches the value of D in self.Dval.
        # Returns True if minimization is complete,
        # and False if not.
        # Completion is achieved if |r| < rtol
        # or |L-L'| < Ltol.
        #
        p, s = self.calc_ps()

        if self.Dval is None:
            self.Dval = self.D(self.w, p)
            if verb:
                print(f"Initial objective = {self.L}")

        # Note: computing r and J each do 1 allsum,
        # so they be combined.
        r = self.r(s) # residual
        # check convergence based on |r|
        rnorm2 = self.allsum(np.vdot(r, r))
        if verb:
            print(f"Residual norm = {rnorm2**0.5}")
        if rnorm2 < rtol**2:
            return True
        # search direction
        d = -la.solve(self.calc_J(s), r)

        # linear extrapolation, D(t) ~ D(0) + t*slope/alpha
        slope = self.alpha*np.dot(r,d)
        if verb:
            # scale the slope to input units
            scale = self.Ntot*self.alpha
            print(f"Residual slope along d = {slope/scale}")
        t = 1.0 # full step
        while True:
            b = self.b + t*d[0]
            w = self.w + t*d[1:]
            D = self.D(w, self.calc_p(b, w))
            if D >= self.Dval + t*slope:
                break
            t = t*self.beta # scale down step
        if verb and t != 1.0:
            print(f"Back-tracked to t = {t}")

        # check change in objective
        done = np.abs(D-self.Dval) < Ltol*self.Ntot
        self.Dval = D
        self.b = b
        self.w = w
        return done

    def allsum(self, x):
        return x

    def r(self, s):
        r = np.zeros(1+self.M)
        r[0] = np.sum(s)
        r[1:] = np.dot(self.z-s, self.X)
        r = self.allsum(r)

        w2 = np.dot(self.w, self.w)
        r[0] -= self.N1 # sum z
        r[1:] += self.w/w2
        return r

    def calc_J(self, s):
        J = np.zeros((1+self.M,1+self.M))
        Q = s*(1.0-s)
        J[0,0] = -Q.sum()
        J[0,1:] = np.dot(Q,self.X)
        J[1:,0] = J[0,1:]
        J[1:,1:] = -np.dot(self.X.T*Q, self.X)

        w2 = np.dot(self.w, self.w)
        J = self.allsum(J)
        #J[1:,1:] += (1.0/w2) * np.identity(self.M) \
        #        - (2.0/w2**1.5) * self.w[:,None]*self.w[None,:]
        J[1:,1:] += (1.0/w2) * np.identity(self.M) \
                - (2.0*w2**-2) * self.w[:,None]*self.w[None,:]

        #if np.vdot(J[0],J[0]) < 1e-8:
        #    J[0,0] = 0.001

        return J

    def decision_function(self, X, tol=0.2):
        assert tol > 0.0 and tol <= 0.5
        print(self.b, self.w)
        p = np.dot(X, self.w) - self.b
        a = np.log(tol/(1.0-tol))
        b = np.log((1.0-tol)/tol)
        print(a, b)
        return p/b
        # clamp ranges
        #p[p < a] = -1.0
        #p[p >= b] = 1.0
        #p[(p >= b)*(p < a)] = 0.0
        #return p

def Ndiff(f, b, x0):
    h = 1e-7
    ih = 1.0/h
    h = 1.0/ih
    x1 = x0.copy()
    f0 = f(b, x0)
    f1 = np.zeros((1+len(x0),)+f0.shape)
    f1[0] = f(b+h,x0)
    for i in range(len(x0)):
        x1[i] = x0[i]+h
        f1[i+1] = f(b,x1)
        x1[i] = x0[i]
    return ih*(f1-f0)

def test1(N=10, M=4):
    y = np.arange(N) < N/2 # 0 or 1 for now
    x = random.standard_normal((N,M)) + 1.0 + y[:,None]
    y = 2*y-1 # -1 or 1
    S = SVM(x, y)
    for i in range(10):
        done = S.step(verb=True)
        if i == 2:
            check_diff(S)
        print(f"Step {i} objective = {S.L}. Converged = {done}.")
        if done:
            break

    print(S.b)
    print(S.w)

def check_diff(S):
    b0 = S.b
    w0 = S.w
    def calcD(b,w):
        return S.D(w, S.calc_p(b,w))
    def calcr(b,w):
        S.b = b
        S.w = w
        p, s = S.calc_ps()
        return S.r(s)
    
    r1 = Ndiff(calcD, b0, w0)
    r0 = calcr(b0, w0)
    err = np.abs(r1-r0).max()
    print(f"Residual err = {err}")
    if err > 1e-4:
        print(r0)
        print(r1)

    J1 = Ndiff(calcr, b0, w0)
    S.b = b0
    S.w = w0
    p, s = S.calc_ps()
    J0 = S.calc_J(s)
    err = np.abs(J1-J0).max()
    print(f"J err = {err}")
    if err > 1e-4:
        print(J0)
        print(J1)

def read_data(fname):
    # reads a data file with raw lines like:
    # -1 3 11 14 19 39 42 55 64 67 73 75 76 80 83 
    x = []
    y = []
    with open(fname, encoding='utf-8') as f:
        for line in f:
            tok = line.split()
            if len(tok) < 15:
                continue
            x.append([float(t) for t in tok[1:]])
            y.append(float(tok[0]))
    return np.array(x), np.array(y)

def test2(fname):
    x, y = read_data(fname)
    print(x.shape, y.shape)
    S = SVM(x, y)
    for i in range(50):
        done = S.step(verb=True)
        print(f"Step {i} objective = {S.L}. Converged = {done}.")
        if done:
            break
    print(S.b)
    print(S.w)

if __name__=="__main__":
    test1(1000, 2)
    #test2("a1a.t")
