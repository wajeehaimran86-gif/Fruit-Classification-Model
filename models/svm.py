# Linear Soft Margin SVM
"""
svm.py
-------
Soft-margin linear SVM implemented from scratch using gradient descent
on the hinge-loss objective. Extended to multi-class via One-vs-Rest (OvR).

Objective (per binary classifier):
    minimize  (1/2)*||theta||^2 + C * sum(hinge_loss)
    hinge_loss(x,y) = max(0, 1 - y*(theta.x + b))
"""

import numpy as np


class BinarySVM:
    """
    A single binary soft-margin linear SVM.
    Labels must be -1 or +1.
    """

    def __init__(self, learning_rate=0.001, C=1.0, n_epochs=1000):
        self.lr = learning_rate
        self.C = C
        self.n_epochs = n_epochs
        self.theta = None
        self.b = 0.0
        self.loss_history = []

    def fit(self, X, y):
        """
        X: (n_samples, n_features)
        y: (n_samples,) with values -1 or +1
        """
        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.n_epochs):
            # decision score for every sample: theta.x + b
            scores = X.dot(self.theta) + self.b
            margins = y * scores  # if >=1, correctly classified with margin

            # indices where hinge loss is active (margin < 1)
            violating = margins < 1

            # Gradient of (1/2)||theta||^2 is theta.
            # Gradient of hinge loss w.r.t theta, for violating samples:
            #   -C * y_i * x_i   (summed), else 0
            grad_theta = self.theta - self.C * np.sum(
                (y[violating])[:, None] * X[violating], axis=0
            )
            grad_b = -self.C * np.sum(y[violating])

            # gradient descent update
            self.theta -= self.lr * grad_theta
            self.b -= self.lr * grad_b

            # track loss for the training curve (report requirement)
            hinge = np.maximum(0, 1 - margins)
            loss = 0.5 * np.dot(self.theta, self.theta) + self.C * np.sum(hinge)
            self.loss_history.append(loss)

    def decision_function(self, X):
        """Raw score theta.x + b (higher = more confident positive class)."""
        return X.dot(self.theta) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


class OvRSVM:
    """
    One-vs-Rest wrapper: trains one BinarySVM per class.
    At prediction time, runs all K classifiers and picks the
    class with the highest decision score.
    """

    def __init__(self, learning_rate=0.001, C=1.0, n_epochs=1000):
        self.learning_rate = learning_rate
        self.C = C
        self.n_epochs = n_epochs
        self.classifiers = {}   # class_name -> BinarySVM
        self.classes_ = []

    def fit(self, X, y):
        """
        X: (n_samples, n_features)
        y: (n_samples,) array of class name strings, e.g. "apple", "banana"
        """
        self.classes_ = sorted(set(y))
        y = np.array(y)

        for cls in self.classes_:
            print(f"  Training binary SVM for class: {cls}")
            binary_labels = np.where(y == cls, 1, -1)

            clf = BinarySVM(
                learning_rate=self.learning_rate,
                C=self.C,
                n_epochs=self.n_epochs,
            )
            clf.fit(X, binary_labels)
            self.classifiers[cls] = clf

    def decision_scores(self, X):
        """
        Returns a (n_samples, n_classes) matrix of raw decision scores,
        one column per class, in the order of self.classes_.
        """
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            scores[:, i] = self.classifiers[cls].decision_function(X)
        return scores

    def predict(self, X):
        """Returns predicted class name for each sample."""
        scores = self.decision_scores(X)
        best_idx = np.argmax(scores, axis=1)
        return np.array([self.classes_[i] for i in best_idx])

    def save(self, filepath):
        """Save all learned parameters (theta, b per class) to a .npz file."""
        save_dict = {"classes": self.classes_}
        for cls in self.classes_:
            save_dict[f"theta_{cls}"] = self.classifiers[cls].theta
            save_dict[f"b_{cls}"] = self.classifiers[cls].b
        np.savez(filepath, **save_dict)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load parameters from a .npz file saved by save()."""
        data = np.load(filepath, allow_pickle=True)
        self.classes_ = list(data["classes"])
        self.classifiers = {}
        for cls in self.classes_:
            clf = BinarySVM()
            clf.theta = data[f"theta_{cls}"]
            clf.b = float(data[f"b_{cls}"])
            self.classifiers[cls] = clf
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Tiny sanity check with synthetic 2D data (2 classes)
    np.random.seed(0)
    X_pos = np.random.randn(50, 2) + np.array([2, 2])
    X_neg = np.random.randn(50, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg])
    y = np.array(["pos"] * 50 + ["neg"] * 50)

    model = OvRSVM(learning_rate=0.01, C=1.0, n_epochs=200)
    model.fit(X, y)
    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    print(f"Sanity-check training accuracy: {accuracy:.2f} (should be close to 1.0)")