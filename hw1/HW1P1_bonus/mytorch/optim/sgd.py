import numpy as np


class SGD:

    def __init__(self, model, lr=0.1, momentum=0):

        self.l = list(filter(lambda x: hasattr(x, "W"), model.layers))
        self.L = len(model.layers)
        self.lr = lr
        self.mu = momentum
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f")
                    for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f")
                    for i in range(self.L)]

    def step(self):

        for i in range(self.L):

            if self.mu == 0:
                # Update weights and biases without momentum
                self.l[i].W = self.l[i].W - self.lr * self.l[i].dLdW
                self.l[i].b -= self.lr * self.l[i].dLdb

            else:
                # Update velocities with momentum
                self.v_W[i] = self.mu * self.v_W[i] - self.lr * self.l[i].dLdW
                self.v_b[i] = self.mu * self.v_b[i] - self.lr * self.l[i].dLdb
                # Update weights and biases with momentum
                self.l[i].W += self.v_W[i]
                self.l[i].b += self.v_b[i]