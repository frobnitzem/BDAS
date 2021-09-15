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

