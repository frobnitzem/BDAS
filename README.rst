=========================================
Big Data Analytics Suite - Python Version
=========================================

This package implements three core algorithms
for big data analytics:
 * Principal Component Analysis (PCA)
 * K-Means Regression (Kmeans)
 * Support Vector Machine Classifier (SVM)

Each of the algorithms is packaged as a class
that accepts an N by M matrix of data points, X.
N is the number of points belonging to the local MPI
rank, and M is the number of features.
While N can vary between MPI ranks, M must be constant.

The SVM classifier also requires an array of N labels,
(for the local data points). The labels are
turned into 0 or 1 using the comparison (label > 0).


References:
 * [Refence Benchmark Results](https://arxiv.org/pdf/1811.02287.pdf)
 * [Coral2 Summary](https://asc.llnl.gov/sites/asc/files/2020-09/BDAS_Summary_b4bcf27_0.pdf)

Quick Start
===========

.. code-block::

    from bdas import SVM

    X, y = read_data("local_file.npy")
    S = SVM(X, y)
    for i in range(50):
        done = S.step(verb=True)
        print(f"Step {i} objective = {S.L}. Converged = {done}.")
        if done:
            break
    print(S.b)
    print(S.w)

When mpi4py support is enabled, you should
launch your program with `mpirun python3 my_prog.py`.

If you're using a supercomputer, consider installing
`spindle <https://computing.llnl.gov/projects/spindle/software>`_,
and then use `spindle mpirun python3 my_prog.py`.

---
SVM
---

The present implementation of SVM uses the maximum likelihood
estimation of the following observation model,

\[ P(x_i | w, b, z_i) =
  \frac{e^{z_i(w^T x_i - b)}}{1 + e^{w^T x_i - b}} 
\]
For simplicity, define
\[
s_i = \frac{e^{z_i(w^T x_i - b)}}{1 + e^{w^T x_i - b}}
.
\]

This model provides a sigmoidal probability of assigning
label $z_i$ to observation, $i$,
\[ P(z_i | w, b, x_i) =
  \frac{p_{z_i} e^{z_i(w^T x_i - b)}}{p_0 + p_1 e^{w^T x_i - b}}
\]
when $p_0$ and $p_1$ are the prior probabilities for
assigning categories 0 or 1 without knowing $x_i$.

Note that the distance from the separating plane to
point $x_i$ is $(w^T x_i - b)/|w| $.  That makes $ 1/|w| $
a scale parameter, which should properly have a scale-independent
(Jeffreys) prior,
\[
P(|w| | I) \propto |w|
\]

Combining the observation model with the prior,
taking the logarithm, and ignoring addititive constants
yields the posterior log-probability,
\[
L(b,w) = \log |w| + z^T (X w - b) - \sum_i \log(1 + e^{w^T x_i - b})
\]

This function is convex, and has a unique maximizer.

The interior-point optimization makes use of the
first and second derivatives of the objective,
(ordering variables b, then w).
\[
r(b,w) = \begin{bmatrix}
-1^T (z-s) \\
w/|w| ^2  + X^T (z-s)
\end{bmatrix}
\]
\[
J(b,w) = \begin{bmatrix}
-\mathrm{trace}(Q) & 1^T Q X \\
X^T Q 1 & I/|w| ^2 - 2w w^T/|w| ^4 - X^T Q X
\end{bmatrix}
\]
defining $Q = diag(s_i(1-s_i))$.
