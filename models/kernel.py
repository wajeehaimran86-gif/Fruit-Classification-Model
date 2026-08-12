"""
kernel.py
----------
Part D: Kernelized SVM.

Contains:
  - polynomial_kernel(), rbf_kernel(): the kernel functions
  - KernelBinarySVM: single binary kernel SVM trained with a
    Kernel-Pegasos-style stochastic gradient method
  - KernelOvRSVM: One-vs-Rest wrapper for multi-class classification
"""

import numpy as np


# ---------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------

def polynomial_kernel(X1, X2, degree=3, c=1.0):
    """K(x,z) = (x.z + c)^degree"""
    return (X1.dot(X2.T) + c) ** degree


def rbf_kernel(X1, X2, sigma=1.0):
    """K(x,z) = exp(-||x-z||^2 / (2*sigma^2))"""
    sq1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    sq2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    sq_dists = sq1 + sq2 - 2 * X1.dot(X2.T)
    sq_dists = np.maximum(sq_dists, 0)
    return np.exp(-sq_dists / (2 * sigma ** 2))


# ---------------------------------------------------------------------
# Kernel Binary SVM (Kernel Pegasos-style training)
# ---------------------------------------------------------------------

class KernelBinarySVM:
    def __init__(self, kernel="rbf", C=1.0, n_epochs=5, sigma=1.0,
                 degree=3, poly_c=1.0):
        self.kernel_name = kernel
        self.C = C
        self.n_epochs = n_epochs
        self.sigma = sigma
        self.degree = degree
        self.poly_c = poly_c

        self.alpha = None
        self.X_train = None
        self.y_train = None
        self.b = 0.0
        self.loss_history = []

    def _kernel(self, X1, X2):
        if self.kernel_name == "rbf":
            return rbf_kernel(X1, X2, sigma=self.sigma)
        elif self.kernel_name == "poly":
            return polynomial_kernel(X1, X2, degree=self.degree, c=self.poly_c)
        else:
            raise ValueError("kernel must be 'rbf' or 'poly'")

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.X_train = X
        self.y_train = y
        self.alpha = np.zeros(n_samples)
        self.b = 0.0

        K = self._kernel(X, X)  # precompute full kernel matrix once

        rng = np.random.default_rng(42)
        t = 0

        for epoch in range(self.n_epochs):
            order = rng.permutation(n_samples)
            epoch_loss = 0.0

            for i in order:
                t += 1
                lr = 1.0 / (self.C * t + 1)

                score_i = np.sum(self.alpha * self.y_train * K[i, :]) + self.b
                margin_i = y[i] * score_i

                if margin_i < 1:
                    self.alpha[i] += lr * self.C
                    self.b += lr * self.C * y[i]
                    epoch_loss += (1 - margin_i)

            self.loss_history.append(epoch_loss / n_samples)
            print(f"    epoch {epoch + 1}/{self.n_epochs}  avg_hinge_loss={self.loss_history[-1]:.4f}")

    def decision_function(self, X):
        K = self._kernel(X, self.X_train)
        return K.dot(self.alpha * self.y_train) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


# ---------------------------------------------------------------------
# One-vs-Rest wrapper
# ---------------------------------------------------------------------

class KernelOvRSVM:
    def __init__(self, kernel="rbf", C=1.0, n_epochs=5, sigma=1.0,
                 degree=3, poly_c=1.0):
        self.kwargs = dict(kernel=kernel, C=C, n_epochs=n_epochs,
                            sigma=sigma, degree=degree, poly_c=poly_c)
        self.classifiers = {}
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        y = np.array(y)

        for cls in self.classes_:
            print(f"  Training kernel SVM for class: {cls}")
            binary_labels = np.where(y == cls, 1, -1)
            clf = KernelBinarySVM(**self.kwargs)
            clf.fit(X, binary_labels)
            self.classifiers[cls] = clf

    def decision_scores(self, X):
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            scores[:, i] = self.classifiers[cls].decision_function(X)
        return scores

    def predict(self, X):
        scores = self.decision_scores(X)
        best_idx = np.argmax(scores, axis=1)
        return np.array([self.classes_[i] for i in best_idx])

    def save(self, filepath):
        save_dict = {"classes": self.classes_}
        for cls in self.classes_:
            clf = self.classifiers[cls]
            save_dict[f"alpha_{cls}"] = clf.alpha
            save_dict[f"y_train_{cls}"] = clf.y_train
            save_dict[f"b_{cls}"] = clf.b
        any_cls = self.classes_[0]
        save_dict["X_train"] = self.classifiers[any_cls].X_train
        np.savez(filepath, **save_dict)
        print(f"Model saved to {filepath}")

    def load(self, filepath, kernel="rbf", sigma=1.0, degree=3, poly_c=1.0):
        data = np.load(filepath, allow_pickle=True)
        self.classes_ = list(data["classes"])
        X_train = data["X_train"]
        self.classifiers = {}
        for cls in self.classes_:
            clf = KernelBinarySVM(kernel=kernel, sigma=sigma, degree=degree, poly_c=poly_c)
            clf.alpha = data[f"alpha_{cls}"]
            clf.y_train = data[f"y_train_{cls}"]
            clf.b = float(data[f"b_{cls}"])
            clf.X_train = X_train
            self.classifiers[cls] = clf
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Sanity check on synthetic non-linearly-separable data (circles)
    np.random.seed(0)
    n = 100
    theta = np.random.uniform(0, 2 * np.pi, n)
    r_inner = np.random.normal(1.0, 0.1, n)
    r_outer = np.random.normal(3.0, 0.1, n)
    X_inner = np.stack([r_inner * np.cos(theta), r_inner * np.sin(theta)], axis=1)
    X_outer = np.stack([r_outer * np.cos(theta), r_outer * np.sin(theta)], axis=1)
    X = np.vstack([X_inner, X_outer])
    y = np.array(["inner"] * n + ["outer"] * n)

    model = KernelOvRSVM(kernel="rbf", C=1.0, n_epochs=3, sigma=1.0)
    model.fit(X, y)
    preds = model.predict(X)
    acc = np.mean(preds == y)
    print(f"Sanity-check (circles, RBF kernel) accuracy: {acc:.2f} (should be high, ~0.9+)")