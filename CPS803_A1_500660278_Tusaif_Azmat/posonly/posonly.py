import numpy as np
import util
import sys

### NOTE : You need to complete logreg implementation first!
class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape
        if self.theta is None:
            self.theta = np.zeros(d, dtype=np.float32)
        for i in range(self.max_iter):
            grad = self._gradient(x, y)
            hess = self._hessian(x)
            prev_theta = np.copy(self.theta)
            self.theta -= self.step_size * np.linalg.inv(hess).dot(grad)
            loss = self._loss(x, y)
            if self.verbose:
                print('[iter: {:02d}, loss: {:.7f}]'.format(i, loss))
            if np.max(np.abs(prev_theta - self.theta)) < self.eps:
                break
        if self.verbose:
            print('Final theta (logreg): {}'.format(self.theta))
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        y_hat = self._sigmoid(x.dot(self.theta))
        return y_hat
        # *** END CODE HERE ***

    def _gradient(self, x, y):
        """Get gradient of J.
        Returns:
            grad: The gradient of J with respect to theta. Same shape as theta.
        """
        n, _ = x.shape
        probs = self._sigmoid(x.dot(self.theta))
        grad = 1 / n * x.T.dot(probs - y)
        return grad

    def _hessian(self, x):
        """Get the Hessian of J given theta and x.
        Returns:
            hess: The Hessian of J. Shape (dim, dim), where dim is dimension of theta.
        """
        n, _ = x.shape
        probs = self._sigmoid(x.dot(self.theta))
        diag = np.diag(probs * (1. - probs))
        hess = 1 / n * x.T.dot(diag).dot(x)
        return hess

    def _loss(self, x, y):
        """Get the empirical loss for logistic regression."""
        eps = 1e-10
        hx = self._sigmoid(x.dot(self.theta))
        loss = -np.mean(y * np.log(hx + eps) + (1 - y) * np.log(1 - hx + eps))
        return loss

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))


# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
def add_intercept(x):
    """Add intercept to matrix x.
    Args:
        x: 2D NumPy array.
    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x
    return new_x

def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***

    # Part (a): Train and test on true labels
    train_x, train_y = util.load_dataset(train_path,label_col='t')
    test_x, test_y = util.load_dataset(test_path,label_col='t')
    train_x_inter = add_intercept(train_x)
    test_x_inter = add_intercept(test_x)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(train_x_inter, train_y)
    pred_y_prob = classifier.predict(test_x_inter)
    test_pred_y = (pred_y_prob > 0.5).astype(int)
    util.plot(test_x, test_y, classifier.theta,'Q2_1_Part_a.png')

    # Make sure to save predicted probabilities to output_path_true using np.savetxt()

    # Part (b): Train on y-labels and test on true labels
    train_x, train_y = util.load_dataset(train_path,label_col='y')
    test_x, test_y = util.load_dataset(test_path,label_col='t')
    train_x_inter = add_intercept(train_x)
    test_x_inter = add_intercept(test_x)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(train_x_inter, train_y)
    pred_y_prob = classifier.predict(test_x_inter)
    test_pred_y = (pred_y_prob > 0.5).astype(int)
    util.plot(test_x, test_y, classifier.theta,'Q2_2_Part_b.png')

    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    #PART (c)
    train_x, train_y = util.load_dataset(train_path)
    test_x, test_y = util.load_dataset(test_path)
    valid_x,valid_y = util.load_dataset(valid_path)
    valid_x_inter = add_intercept(valid_x)
    train_x_inter = add_intercept(train_x)
    test_x_inter = add_intercept(test_x)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(train_x_inter, train_y)

    #calculate correction
    correction=np.mean(classifier.predict(valid_x_inter))
    pred_y_prob = classifier.predict(test_x_inter)
    test_pred_y = (pred_y_prob > 0.5).astype(int)
    util.plot(test_x, test_y, classifier.theta,'Q2_3_Part_c.png',correction)

    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
