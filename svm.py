import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plot_results(models, titles, X, y, plot_sv=False):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()


def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def svm_kernels(C, X, Y, coef0):
    #models
    lin_kernel = svm.SVC(C=C, kernel="linear", gamma="auto")
    poly_2_kernel = svm.SVC(C=C, kernel="poly", degree=2, gamma="auto", coef0=coef0)
    poly_3_kernel = svm.SVC(C=C, kernel="poly", degree=3, gamma="auto", coef0=coef0)

    # fit
    lin_kernel.fit(X, Y)
    poly_2_kernel.fit(X, Y)
    poly_3_kernel.fit(X, Y)

    # plot
    models = [lin_kernel, poly_2_kernel, poly_3_kernel]
    plot_results(models, ["linear", "poly2", "poly3"], X, Y)


def part_c(C, X, Y, gamma):
    # Random relabel -1 with 1 w.p 0.1
    relabel = np.random.choice([-1, 1], np.shape(Y)[0], p=[0.1, 0.9])
    Y = np.maximum(Y, relabel)

    # fit
    poly_2_kernel = svm.SVC(C=C, kernel="poly", degree=2, gamma="auto", coef0=1)
    rbf_kernel = svm.SVC(C=C, kernel="rbf", gamma=gamma)
    poly_2_kernel.fit(X, Y)
    rbf_kernel.fit(X, Y)

    # plot
    models = [poly_2_kernel, rbf_kernel]
    plot_results(models, ["poly2", "rbf"], X, Y)


def main():
    C_hard = 1000000.0  # SVM regularization parameter
    C = 10
    n = 100

    # Data is labeled by a circle

    radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
    angles = 2 * math.pi * np.random.random(2 * n)
    X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
    X2 = (radius * np.sin(angles)).reshape((2 * n, 1))

    X = np.concatenate([X1, X2], axis=1)
    Y = np.concatenate([np.ones((n, 1)), -np.ones((n, 1))], axis=0).reshape([-1])

    svm_kernels(C, X, Y, 0)  # homogeneous
    svm_kernels(C, X, Y, 1)  # non-homogeneous
    part_c(C, X, Y, 10)


if __name__ == "__main__":
    main()
