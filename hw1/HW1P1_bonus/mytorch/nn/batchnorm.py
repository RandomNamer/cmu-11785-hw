import numpy as np

class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        self.Z = Z
        self.N = Z.shape[0]

        if eval == False:
            self.M = np.mean(Z, axis=0)
            self.V = np.var(Z, axis=0)

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V

            self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)
        else:
            self.NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)

        self.BZ = self.BW * self.NZ + self.Bb

        return self.BZ

    def backward(self, dLdBZ):
        N, D = self.Z.shape

        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0)
        self.dLdBb = np.sum(dLdBZ, axis=0)

        dLdNZ = dLdBZ * self.BW
        dLdV = np.sum(dLdNZ * (self.Z - self.M), axis=0) * -0.5 * (self.V + self.eps)**-1.5
        dLdM = np.sum(dLdNZ, axis=0) * -1 / np.sqrt(self.V + self.eps) + dLdV * np.mean(-2*(self.Z - self.M), axis=0)

        dLdZ = dLdNZ / np.sqrt(self.V + self.eps) + dLdV * 2 * (self.Z - self.M) / N + dLdM / N

        return dLdZ