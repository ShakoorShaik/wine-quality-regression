import numpy as np
from sklearn.neural_network import MLPRegressor

class ANNRegression:
    def __init__(
        self,
        num_layers,
        num_units,
        l2_coeff=0.0,
        learning_rate=1e-2,
        max_epochs=2000,
        activation="relu",
        random_state=42,
        batch_size=128,
        use_weighted_mse=False,
    ):
        """
        Initialize the class of a neural network.

        args:
            - num_layers (int): Number of hidden layers.
            - num_units (int or sequence[int]): Either a single hidden width
              used for every hidden layer, or an explicit list of hidden-layer
              widths such as [32, 16].
            - l2_coeff (float): The coefficient of the L2 regularization term.
            - learning_rate (float): The optimization step size.
            - max_epochs (int): Number of training iterations.
            - activation (str): Hidden layer activation function.
                One of {"relu", "sigmoid", "tanh"}.
            - random_state (int): Random seed for reproducibility.
        """
        assert l2_coeff >= 0, f"l2_coeff must be non-negative. Got: {l2_coeff}"
        assert learning_rate > 0, f"learning_rate must be positive. Got: {learning_rate}"
        assert max_epochs >= 1, f"max_epochs must be at least 1. Got: {max_epochs}"
        assert activation in {"relu", "sigmoid", "tanh"}, (
            f"Unsupported activation: {activation}"
        )

        hidden_layer_sizes = self.normalize_hidden_layer_sizes(num_layers, num_units)

        self.num_layers = len(hidden_layer_sizes)
        self.num_units = num_units
        self.hidden_layer_sizes = hidden_layer_sizes
        self.l2_coeff = l2_coeff
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.activation = activation
        self.random_state = random_state
        self.batch_size = batch_size

        self.model = None
        self.parameters = None
        self.loss_history = []

    def normalize_hidden_layer_sizes(self, num_layers, num_units):
        if np.isscalar(num_units):
            assert num_layers >= 1, f"num_layers must be at least 1. Got: {num_layers}"
            assert int(num_units) >= 1, f"num_units must be at least 1. Got: {num_units}"
            return tuple([int(num_units)] * int(num_layers))

        hidden_layer_sizes = tuple(int(width) for width in num_units)

        assert len(hidden_layer_sizes) >= 1, (
            f"At least one hidden layer is required. Got: {hidden_layer_sizes}"
        )
        assert all(width >= 1 for width in hidden_layer_sizes), (
            f"All hidden-layer widths must be positive. Got: {hidden_layer_sizes}"
        )
        assert num_layers == len(hidden_layer_sizes), (
            f"num_layers must match len(num_units). "
            f"(num_layers: {num_layers}, len(num_units): {len(hidden_layer_sizes)})"
        )

        return hidden_layer_sizes

    def to_numpy(self, array, as_column_vector=False):
        if hasattr(array, "to_numpy"):
            array = array.to_numpy()
        else:
            array = np.asarray(array)

        array = array.astype(np.float64)

        if as_column_vector:
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            assert array.ndim == 2 and array.shape[1] == 1, (
                f"Expected shape (N, 1). Got: {array.shape}"
            )
        else:
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            assert array.ndim == 2, f"X must be 2D. Got shape: {array.shape}"

        return array

    def sklearn_activation(self):
        if self.activation == "sigmoid":
            return "logistic"
        return self.activation

    def build_model(self, l2_coeff):
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.sklearn_activation(),
            solver="adam",
            alpha=l2_coeff,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_epochs,
            shuffle=True,
            random_state=self.random_state,
            tol=1e-4,
            n_iter_no_change=40,
            early_stopping=True,
            validation_fraction=0.1,
        )

    def predict(self, X):
        assert self.model is not None, "The model must be fitted before prediction."

        X = self.to_numpy(X)
        prediction = self.model.predict(X).reshape(-1, 1)
        return prediction

    def fit(self, train_X, train_Y):
        self.fit_gradient_descent(train_X, train_Y, l2_coeff=0.0)

    def fit_with_l2_regularization(self, train_X, train_Y):
        self.fit_gradient_descent(train_X, train_Y, l2_coeff=self.l2_coeff)

    def fit_gradient_descent(self, train_X, train_Y, l2_coeff):
        train_X = self.to_numpy(train_X)
        train_Y = self.to_numpy(train_Y, as_column_vector=True).ravel()

        assert train_X.shape[0] == train_Y.shape[0], (
            f"Number of inputs and outputs are different. "
            f"(train_X: {train_X.shape[0]}, train_Y: {train_Y.shape[0]})"
        )

        self.model = self.build_model(l2_coeff)
        self.model.fit(train_X, train_Y)

        self.loss_history = list(self.model.loss_curve_)
        self.parameters = {
            "coefs": self.model.coefs_,
            "intercepts": self.model.intercepts_,
        }

    def compute_mse(self, X, observed_Y):
        observed_Y = self.to_numpy(observed_Y, as_column_vector=True)
        pred_Y = self.predict(X)

        assert pred_Y.shape == observed_Y.shape, (
            f"Prediction and observed output shapes do not match. "
            f"(pred_Y: {pred_Y.shape}, observed_Y: {observed_Y.shape})"
        )

        mse = float(np.mean((pred_Y - observed_Y) ** 2))
        return mse

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    Y = (2.0 * X[:, [0]] - 1.5 * X[:, [1]] + 0.5 * X[:, [2]] + 3.0)

    model = ANNRegression(
        num_layers=1,
        num_units=16,
        learning_rate=1e-2,
        max_epochs=3000,
        activation="relu",
        random_state=0,
    )

    model.fit(X, Y)
    pred_Y = model.predict(X)
    print("Training MSE:", model.compute_mse(X, Y))
    print("Prediction shape correct:", pred_Y.shape == Y.shape)
