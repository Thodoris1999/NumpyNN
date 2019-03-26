import numpy as np
import matplotlib.pyplot as plt

epoch_losses = []
losses_avg = []


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse_loss(output, labels):
    return np.multiply(output - labels, output - labels)


def d_mse_loss(output, labels):
    return 2 * (output - labels)


class NeuralNet:
    def __init__(self, layer_sizes, learning_rate, activation=sigmoid, d_activation=d_sigmoid, loss=mse_loss,
                 d_loss=d_mse_loss):
        self.activation = activation
        self.d_activation = d_activation
        self.loss = loss
        self.d_loss = d_loss
        self.learning_rate = learning_rate
        self.weights = [2 * np.random.random((layer_sizes[i], layer_sizes[i - 1])) - 1 for i in
                        range(1, len(layer_sizes))]
        self.biases = [2 * np.random.random(layer_sizes[i]) - 1 for i in range(1, len(layer_sizes))]

    def feedforward(self, x):
        z_all = []
        a_all = [x]
        cur_a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, cur_a) + b
            z_all.append(z)
            a = self.activation(z)
            a_all.append(a)
            cur_a = a

        return z_all, a_all

    def backprop(self, x, z_all, a_all, y):
        d_biases = np.zeros_like(self.biases)
        d_weights = np.zeros_like(self.weights)

        d_aL = self.d_loss(a_all[-1], y)
        d_out_zL = self.d_activation(z_all[-1])
        d_zL = d_aL * d_out_zL

        d_biases[-1] = d_zL
        d_weights[-1] = np.dot(d_zL.reshape((-1, 1)), a_all[-2].reshape((1, -1)))

        d_zl = d_zL

        for l in range(len(self.weights) - 2, -1, -1):
            d_al = np.dot(d_zl, self.weights[l + 1])
            d_al_zl = self.d_activation(z_all[l])
            d_zl = d_al * d_al_zl

            d_weights[l] = np.dot(d_zl.reshape((-1, 1)), a_all[l].reshape((1, -1)))
            d_biases[l] = d_zl

        return d_weights, d_biases

    def step(self, x, y):
        z_all, a_all = self.feedforward(x)
        d_weights, d_biases = self.backprop(x, z_all, a_all, y)

        loss = self.loss(a_all[-1], y)
        epoch_losses.append(loss)

        self.weights -= self.learning_rate * d_weights
        self.biases -= self.learning_rate * d_biases

    def train(self, x, y, epochs):
        for epoch in range(epochs):
            epoch_losses.clear()
            for i in range(len(x)):
                self.step(x[i], y[i])
            losses_avg.append(sum(epoch_losses) / 4)

    def predict(self, x):
        _, a_all = self.feedforward(x)
        print("input={}, predicted={}".format(x, a_all[-1]))


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
net = NeuralNet(layer_sizes=(2, 5, 1), learning_rate=0.1)
net.train(x, y, 6000)

net.predict([0, 0])
net.predict([0, 1])
net.predict([1, 0])
net.predict([1, 1])

plt.plot(losses_avg)
plt.show()
