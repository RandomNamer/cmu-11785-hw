import numpy as np

# from mytorch.nn.module import Module

class Linear():
    def __init__(self, in_features, out_features, random_init=False, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        if random_init:
            self.W = np.random.randn(out_features, in_features)
            self.b = np.random.randn(out_features, 1)
        else:
            self.W = np.zeros((out_features, in_features))
            self.b = np.zeros((out_features, 1))
            
        
        self.debug = debug
        

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A
        self.N = A.shape[0]
        self.Ones = np.ones((self.N,1))
        Z = np.dot(A, self.W.T) + np.dot(self.Ones, self.b.T)

        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: The gradient of the loss with respect to the output
        :return: The gradient of the loss with respect to the input
        """
        
        dLdA = np.dot(dLdZ, self.W)
        self.dLdW = np.dot(dLdZ.T, self.A) 
        self.dLdb = np.dot(dLdZ.T, self.Ones)

        if self.debug:
            self.dLdA = dLdA

        return dLdA