import pickle
import numpy as np


def linear(x, derivation=False):
    """ activation function linear """
    return 1 if derivation else x

def relu(x, derivation=False):
    """ activation function relu """
    return 1.0 * (x > 0) if derivation else np.maximum(x, 0)


def step_function(x):
    """ activation step function """
    return 1 if x >= 0 else 0

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class neural_network():
    def __init__(self, input_shape, hidden_neurons, output_shape, learning_rate):
        self.l1_weights = np.random.normal(scale=0.1, size=(input_shape, hidden_neurons))
        self.l1_biases = np.zeros(hidden_neurons)

        self.l2_weights = np.random.normal(scale=0.1, size=(hidden_neurons, output_shape))
        self.l2_biases = np.zeros(output_shape)

        self.learning_rate = learning_rate

    def fit(self, x, y, epochs=1):
        """
        method implements backpropagation
        """
        for _ in range(epochs):
            # Forward propagation
            # First layer
            u1 = np.dot(x, self.l1_weights) + self.l1_biases
            l1o = relu(u1)

            # Second layer
            u2 = np.dot(l1o, self.l2_weights) + self.l2_biases
            l2o = linear(u2)

            # Backward Propagation
            # Second layer
            d_l2o = l2o - y
            d_u2 = linear(u2, derivation=True)

            g_l2 = np.dot(l1o.T, d_u2 * d_l2o)
            d_l2b = d_l2o * d_u2
            # First layer
            d_l1o = np.dot(d_l2o, self.l2_weights.T)
            d_u1 = relu(u1, derivation=True)

            g_l1 = np.dot(x.T, d_u1 * d_l1o)
            d_l1b = d_l1o * d_u1

            # Update weights and biases
            self.l1_weights -= self.learning_rate * g_l1
            self.l1_biases -= self.learning_rate * d_l1b.sum(axis=0)

            self.l2_weights -= self.learning_rate * g_l2
            self.l2_biases -= self.learning_rate * d_l2b.sum(axis=0)

        # Return actual loss
        return np.mean(np.subtract(y, l2o) ** 2)

    def predict(self, x):
        """
        method predicts q-values for state x
        """
        # First layer
        u1 = np.dot(x, self.l1_weights) + self.l1_biases
        l1o = relu(u1)

        # Second layer
        u2 = np.dot(l1o, self.l2_weights) + self.l2_biases
        l2o = linear(u2)

        return softmax(l2o)
        # return l2o

    def save_model(self, name):
        """
        method saves model
        """
        with open("{}.pkl".format(name), "wb") as model:
            pickle.dump(self, model, pickle.HIGHEST_PROTOCOL)

    def load_model(self, name):
        """
        method loads model
        """
        with open("{}".format(name), "rb") as model:
            tmp_model = pickle.load(model)

        self.l1_weights = tmp_model.l1_weights
        self.l1_biases = tmp_model.l1_biases

        self.l2_weights = tmp_model.l2_weights
        self.l2_biases = tmp_model.l2_biases

        self.learning_rate = tmp_model.learning_rate

    def load_state_dict(self, tmp_model):
        """
        method loads state dict from model
        """
        self.l1_weights = tmp_model.l1_weights
        self.l1_biases = tmp_model.l1_biases

        self.l2_weights = tmp_model.l2_weights
        self.l2_biases = tmp_model.l2_biases

        self.learning_rate = tmp_model.learning_rate
