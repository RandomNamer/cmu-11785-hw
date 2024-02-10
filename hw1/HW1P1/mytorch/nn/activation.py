import numpy as np
import scipy
import math

from mytorch.nn.module import Module

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid(Module):
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        dAdZ = self.A * (1 - self.A)
        dLdZ = dLdA * dAdZ
        return dLdZ


class Tanh(Module):
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):
        dAdZ = 1 - self.A ** 2
        dLdZ = dLdA * dAdZ
        return dLdZ


class ReLU(Module):
    def forward(self, Z):
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):
        dAdZ = (self.A > 0).astype(float)
        dLdZ = dLdA * dAdZ
        return dLdZ


class GELU(Module):
    def forward(self, Z):
        self.Z = Z
        self.A = 0.5 * Z * (1 + scipy.special.erf(Z / math.sqrt(2)))
        return self.A

    def backward(self, dLdA):
        dLdZ = dLdA*(0.5*(1+scipy.special.erf(self.Z/math.sqrt(2)))+(self.Z/math.sqrt(2*math.pi))*np.exp(-self.Z*self.Z/2))
        return dLdZ


class Softmax(Module):
    def forward(self, Z):
        Z_exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.A = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)
        return self.A

    def backward(self, dLdA):
        N, C = dLdA.shape
        dLdZ = np.zeros((N, C, C))
        for i in range(N):
            for j in range(C):
                for k in range(C):
                    if j == k:
                        dLdZ[i, j, k] = self.A[i, j] * (1 - self.A[i, j])
                    else:
                        dLdZ[i, j, k] = -self.A[i, j] * self.A[i, k]
        return np.einsum('ijk,ik->ij', dLdZ, dLdA)