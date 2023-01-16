import numpy as np


class FNN:
    def __init__(self, neurons_count, activations, params: np.ndarray):
        self.neurons_count = neurons_count
        self.activations = activations
        self.layers_count = len(self.neurons_count)

        self.rebuild(params=params)

    def rebuild(self, params: np.ndarray) -> None:
        split = [0]
        for i in range(1, self.layers_count):
            split.append(self.neurons_count[i] *
                         self.neurons_count[i-1] + self.neurons_count[i] + split[i - 1])

        self.layers = []
        for i in range(1, self.layers_count):
            l = split[i - 1]
            m = split[i] - self.neurons_count[i]
            r = split[i]
            # l to m for layer weights
            # m to r for biases
            self.layers.append(LayerDense(
                n_inputs=self.neurons_count[i], n_neurons=self.neurons_count[i-1],
                weights=params[l:m], biases=params[m:r]))

    def predict(self, input: np.ndarray) -> np.ndarray:
        out = input
        for i in range(1, self.layers_count):
            out = self.layers[i - 1].forward(input=out)
            if (self.activations[i] == "relu"):
                out = np.array(list(map(relu, out)))
            elif (self.activations[i] == "sigmoid"):
                out = np.array(list(map(sigmoid, out)))
        return softmax(out)


class LayerDense:
    def __init__(self, n_inputs, n_neurons, weights: np.ndarray, biases: np.ndarray) -> None:
        self.weights = weights.reshape((n_neurons, n_inputs))
        self.biases = biases

    def forward(self, input) -> np.ndarray:
        return np.dot(input, self.weights) + self.biases


def relu(Z: float) -> float:
    return max(0, Z)


def sigmoid(Z: float) -> float:
    return 1/(1+np.exp(-Z))


def softmax(Z: np.ndarray) -> np.ndarray:
    e_x = np.exp(Z - np.max(Z))
    return e_x / e_x.sum()
