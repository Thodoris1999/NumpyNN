import numpy as np
import matplotlib.pyplot as plt

epoch_losses = []
losses_avg = []


class NeuralNet:
    def __init__(self, hidden_size, learning_rate):
        self.learning_rate = learning_rate
        self.weights1 = 2 * np.random.random((hidden_size, 2)) - 1
        self.biases1 = 2 * np.random.random(hidden_size) - 1
        self.weights2 = 2 * np.random.random((1, hidden_size)) - 1
        self.biases2 = 2 * np.random.random(1) - 1

    def feedforward(self, x):
        self.a1 = sigmoid(np.dot(self.weights1, x) + self.biases1)
        self.output = sigmoid(np.dot(self.weights2, self.a1) + self.biases2)

    def backprop(self, x, y):
        d_out = d_mse_loss(self.output, y)
        d_out_zout = d_sigmoid(np.dot(self.weights2, self.a1) + self.biases2)
        d_z2 = d_out * d_out_zout

        self.d_biases2 = d_z2
        self.d_weights2 = np.dot(d_z2.reshape((-1, 1)), self.a1.reshape((1, -1)))

        d_a1 = np.dot(d_z2, self.weights2)
        d_a1_z1 = d_sigmoid(np.dot(self.weights1, x) + self.biases1)
        d_z1 = d_a1 * d_a1_z1

        self.d_weights1 = np.dot(d_z1.reshape((-1, 1)), x.reshape((1, -1)))
        self.d_biases1 = d_z1

    def step(self, x, y):
        self.feedforward(x)
        self.backprop(x, y)

        loss = mse_loss(self.output, y)
        epoch_losses.append(loss)

        self.weights2 -= self.learning_rate * self.d_weights2
        self.biases2 -= self.learning_rate * self.d_biases2
        self.weights1 -= self.learning_rate * self.d_weights1
        self.biases1 -= self.learning_rate * self.d_biases1

    def train(self, x, y, epochs):
        for epoch in range(epochs):
            epoch_losses.clear()
            for i in range(len(x)):
                self.step(x[i], y[i])
            losses_avg.append(sum(epoch_losses) / 4)

    def predict(self, x):
        self.feedforward(x)
        print("input={}, predicted={}".format(x, self.output))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse_loss(output, labels):
    return np.multiply(output - labels, output - labels)


def d_mse_loss(output, labels):
    return 2 * (output - labels)


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
net = NeuralNet(5, 0.1)
net.train(x, y, 6000)

net.predict([0, 0])
net.predict([0, 1])
net.predict([1, 0])
net.predict([1, 1])

plt.plot(losses_avg)
plt.show()
