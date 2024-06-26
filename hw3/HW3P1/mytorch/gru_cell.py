import numpy as np
from nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        
        self.z = self.z_act.forward(np.dot(self.Wzh, h_prev_t) + self.bzh + np.dot(self.Wzx, x) + self.bzx)
        
        self.r = self.r_act.forward(np.dot(self.Wrh, h_prev_t) + self.brh + np.dot(self.Wrx, x) + self.brx)
        
        self.n = self.h_act.forward( self.r * (np.dot(self.Wnh, h_prev_t) + self.bnh) + np.dot(self.Wnx, x) + self.bnx)
         
        self.h = (1 - self.z) * self.n + self.z * h_prev_t
        
        return self.h

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (hidden_dim)
            derivative of the loss wrt the input hidden h.

        """

        # SOME TIPS:
        # 1) Make sure the shapes of the calculated dWs and dbs match the initalized shapes of the respective Ws and bs
        # 2) When in doubt about shapes, please refer to the table in the writeup.
        # 3) Know that the autograder grades the gradients in a certain order, and the local autograder will tell you which gradient you are currently failing.
        
        dLdh = delta
        
        dLdz = dLdh*(self.hidden-self.n)
        dLdn = dLdh*(1-self.z)
        
        dntanh = self.h_act.backward(dLdn, state=self.n)
        print(dntanh, self.x)
        
        self.dWnx = np.outer(np.expand_dims(dntanh, axis=1), np.expand_dims(self.x, axis=0))
        self.dbnx = dntanh
        dr = dntanh * (self.Wnh @ self.hidden + self.bnh)
        self.dWnh = np.expand_dims(dntanh, axis=1) * np.expand_dims(self.r, axis=1) @ np.expand_dims(self.hidden, axis=1).T 
        self.dbnh = dntanh * self.r
        
        dzsig = self.z_act.backward(dLdz)
        self.dWzx = np.expand_dims(dzsig, axis=1) @ np.expand_dims(self.x, axis=1).T
        self.dbzx = dzsig
        self.dWzh = np.expand_dims(dzsig, axis=1) * np.expand_dims(self.hidden, axis=1).T
        self.dbzh = dzsig
        
        drsig = self.r_act.backward(dr)
        self.dWrx = np.expand_dims(drsig, axis=1) @ np.expand_dims(self.x, axis=1).T
        self.dbrx = drsig
        self.dWrh = np.expand_dims(drsig, axis=1) * np.expand_dims(self.hidden, axis=1).T 
        self.dbrh = drsig
        
        dx = np.squeeze(np.expand_dims(dntanh, axis=1).T @ self.Wnx + np.expand_dims(dzsig, axis=1).T @ self.Wzx + np.expand_dims(drsig, axis=1).T @ self.Wrx)
        dh_prev_t = dLdh * self.z + dntanh * self.r @ self.Wnh + dzsig @ self.Wzh + drsig @ self.Wrh
        
        return dx, dh_prev_t