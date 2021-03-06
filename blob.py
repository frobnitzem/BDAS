# Example SVM solution plotting code from scikit-learn
# modified to show results of bayes.SVM.
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from bayes import SVM

# we create 40 separable points
X, y = make_blobs(n_samples=40, centers=2, random_state=6)
print(X.shape)

# fit the model, don't regularize for illustration purposes
#clf = svm.SVC(kernel='linear', C=1000)
#clf.fit(X, y)
print(y)
clf = SVM(X, y)
for i in range(50):
    done = clf.step(verb=True)
    print(f"Step {i} objective = {clf.L}. Converged = {done}.")
    if done:
        break

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 60)
yy = np.linspace(ylim[0], ylim[1], 60)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
#ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
#           linewidth=1, facecolors='none', edgecolors='k')
plt.show()
