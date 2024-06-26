import numpy as np

# from mytorch.nn.module import Module
# from mytorch.nn import Softmax

class Softmax:
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


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = np.shape(A)[0]  # TODO
        self.C = np.shape(A)[1]  # TODO
        se = (self.A - self.Y) ** 2
        sse = np.sum(se)
        mse = sse / (self.N * self.C)

        return mse

    def backward(self):

        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = A.shape[0]
        C = A.shape[1]
        self.N = N

        Ones_C = np.ones(C, dtype='f')
        Ones_N = np.ones(N, dtype='f')

        self.softmax = Softmax().forward(A)
        crossentropy = -Y * np.log(self.softmax)
        sum_crossentropy = np.sum(np.dot(Ones_N.T, crossentropy))
        L = sum_crossentropy / N

        return L

    def backward(self):
        dLdA = (self.softmax - self.Y) / self.N
        return dLdA