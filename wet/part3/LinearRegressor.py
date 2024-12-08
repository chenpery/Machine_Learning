from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Custom linear regression model
    """
    def __init__(self, lr: float = 1e-5, huber_delta: float = 1.0):
        """
        Initialize an instance of this class.
        ** Do not edit this method **

        :param lr: the SGD learning rate (step size)
        """
        self.lr = lr
        self.batch_size = 32
        self.w = None
        self.b = 0.0
        self.huber_delta = huber_delta

    # Initialize a random weight vector
    def init_solution(self, n_features: int):
        """
        Randomize an initial solution (weight vector)
        ** Do not edit this method **

        :param n_features:
        """
        self.w = np.zeros(n_features)
        self.b = 0.0

    @staticmethod
    def loss(w, b: float, X, y, huber_delta: float):
        """
        Compute the MSE objective loss.

        :param w: weight vector for linear regression; array of shape (n_features,)
        :param b: bias scalar for linear regression
        :param X: samples for loss computation; array of shape (n_samples, n_features)
        :param y: targets for loss computation; array of shape (n_samples,)
        :return: the linear regression objective loss (float scalar)
        """

        # TODO: complete the loss calculation
        residuals = X.dot(w) + b - y

        indicator_smaller_than_delta = (np.abs(residuals) <= huber_delta)

        # sign * is np.multiply so we calc the squared loss for every row and than we choose just relevant rows with residuals smaller than delta
        squared_loss = np.sum(0.5 * (residuals ** 2) * indicator_smaller_than_delta)

        indicator_bigger_than_delta = (1 - indicator_smaller_than_delta)
        # sign * is np.multiply so we calc the linear loss for every row and than we choose just relevant rows with residuals bigger than delta
        linear_loss= np.sum((huber_delta * (np.abs(residuals) - 0.5 * huber_delta)) * (indicator_bigger_than_delta))

        num_rows = X.shape[0]

        loss = (squared_loss + linear_loss)/num_rows
        return loss

    @staticmethod
    def gradient(w, b: float, X, y, huber_delta: float):
        """
        Compute the (analytical) linear regression objective gradient.

        :param w: weight vector for linear regression; array of shape (n_features,)
        :param b: bias scalar for linear regression
        :param X: samples for loss computation; array of shape (n_samples, n_features)
        :param y: targets for loss computation; array of shape (n_samples,)
        :return: a tuple with (the gradient of the weights, the gradient of the bias)
        """
        # TODO: calculate the analytical gradient w.r.t w and b
        residuals = X.dot(w) + b - y
        sign_residuals = np.sign(residuals)

        indicator_smaller_than_delta = (np.abs(residuals) <= huber_delta)
        squared_loss_gradient_w = X.T @ (residuals * indicator_smaller_than_delta)

        indicator_bigger_than_delta = (1 - indicator_smaller_than_delta)
        linear_loss_gradient_w=  huber_delta*(X.T @ (sign_residuals * indicator_bigger_than_delta))

        squared_loss_gradient_b= np.dot(residuals.T, indicator_smaller_than_delta)
        linear_loss_gradient_b = huber_delta*np.dot(sign_residuals.T, indicator_bigger_than_delta)

        num_rows = X.shape[0]
        g_w = (squared_loss_gradient_w + linear_loss_gradient_w)/num_rows
        g_b = (squared_loss_gradient_b + linear_loss_gradient_b)/num_rows

        return g_w, g_b

    def fit_with_logs(self, X, y, max_iter: int = 1000, keep_losses: bool = True,
                      X_val  =None, y_val = None):
        """
        Fit the model according to the given training data.

        :param X: training samples; array of shape (n_samples, n_features)
        :param y: training targets; array of shape (n_samples,)
        :param max_iter: number of SGD iterations
        :param keep_losses: should compute the train & val losses during training?
        :param X_val: validation samples to compute the loss for (for logs only)
        :param y_val: validation labels to compute the loss for (for logs only)
        :return: training and validation losses during training
        """
        # Initialize learned parameters
        self.init_solution(X.shape[1])

        train_losses = []
        val_losses = []

        if keep_losses:
            train_losses.append(self.loss(self.w, self.b, X, y, self.huber_delta))
            val_losses.append(self.loss(self.w, self.b, X_val, y_val, self.huber_delta))

        # Iterate over batches (SGD)
        for itr in range(0, max_iter):
            start_idx = (itr * self.batch_size) % X.shape[0]
            end_idx = min(X.shape[0], start_idx + self.batch_size)
            batch_X = X[start_idx: end_idx]
            batch_y = y[start_idx: end_idx]

            # TODO: Compute the gradient for the current *batch*
            g_w, g_b = self.gradient(self.w, self.b, batch_X, batch_y, self.huber_delta)

            # Perform a gradient step
            # TODO: update the learned parameters correctly
            self.w -= self.lr*g_w
            self.b -= self.lr*g_b

            if keep_losses:
                train_losses.append(self.loss(self.w, self.b,  X, y, self.huber_delta))
                val_losses.append(self.loss(self.w, self.b,  X_val, y_val, self.huber_delta))

        return train_losses, val_losses

    def fit(self, X, y, max_iter: int = 1000):
        """
        Fit the model according to the given training data.
        ** Do not edit this method **

        :param X: training samples; array of shape (n_samples, n_features)
        :param y: training targets; array of shape (n_samples,)
        :param max_iter: number of SGD iterations
        """
        self.fit_with_logs(X, y, max_iter=max_iter, keep_losses=False)

        return self

    def predict(self, X):
        """
        Regress labels on samples in X.

        :param X: samples for prediction; array of shape (n_samples, n_features)
        :return: Predicted continuous labels for samples in X; array of shape (n_samples,)
        """

        # TODO: Compute
        y_pred = X.dot(self.w) + self.b

        return y_pred