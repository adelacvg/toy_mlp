import numpy as np
from random import random

from numpy.core.numeric import outer

class MLP:
    """A Multilayer Perceptron class"""

    def __init__(self,num_inputs,hidden_layers,num_outputs):

        self.num_inputs = num_inputs
        self.hidden_layer = hidden_layers
        self.num_outputs = num_outputs
        
        layers = [num_inputs] + hidden_layers + [num_outputs]

        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def forward_propagate(self, inputs):

        activations = inputs

        self.activations[0] = activations

        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)

            activations = self._sigmoid(net_inputs)

            self.activations[i+1] = activations
        
        return activations

    def back_propagate(self, error):

        for i in reversed(range(len(self.derivatives))):

            activations = self.activations[i+1]

            delta = error * self._sigmoid_derivative(activations)

            delta_re = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i]

            current_activations = current_activations.reshape(current_activations.shape[0], -1)

            self.derivatives[i] = np.dot(current_activations, delta_re)

            error = np.dot(delta, self.weights[i].T)

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0

            for j, input in enumerate(inputs):
                target = targets[j]

                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)
            
            print("Error: {} at epoch {}".format(sum_error / len(items), i+1))

        print("Training complete!")
        print("=====")

    
    def gradient_descent(self, learningRate=1):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives*learningRate

    def _sigmoid(self, x):

        y = 1.0 / (1 + np.exp(-x))
        return y

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _mse(self, target, output):
        return np.average((target-output) ** 2)

if __name__ == "__main__":

    items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    mlp = MLP(2, [5], 1)

    mlp.train(items, targets, 50, 0.1)

    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    output = mlp.forward_propagate(input)

    print("Our network believes that {} + {} is equal to {}.".format(input[0], input[1], output[0]))
