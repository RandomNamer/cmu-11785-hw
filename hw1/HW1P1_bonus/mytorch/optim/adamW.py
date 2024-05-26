# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class AdamW():
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.l = model.layers[::2] # every second layer is activation function
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0
        self.weight_decay=weight_decay

        self.m_W = [np.zeros(l.W.shape, dtype="f") for l in self.l]
        self.v_W = [np.zeros(l.W.shape, dtype="f") for l in self.l]

        self.m_b = [np.zeros(l.b.shape, dtype="f") for l in self.l]
        self.v_b = [np.zeros(l.b.shape, dtype="f") for l in self.l]

    def step(self):

        self.t += 1
        for layer_id, layer in enumerate(self.l):
            mw = self.m_W[layer_id]
            vw = self.v_W[layer_id]
            mb = self.m_b[layer_id]
            vb = self.v_b[layer_id]
            
            mw = self.beta1 * mw + (1 - self.beta1) * layer.dLdW
            mb = self.beta1 * mb + (1 - self.beta1) * layer.dLdb

            vw = self.beta2 * vw + (1 - self.beta2) * layer.dLdW * layer.dLdW
            vb = self.beta2 * vb + (1 - self.beta2) * layer.dLdb * layer.dLdb
            
            self.m_W[layer_id] = mw
            self.v_W[layer_id] = vw
            self.m_b[layer_id] = mb
            self.v_b[layer_id] = vb
            

            mw_hat = mw / (1 - self.beta1 ** self.t)
            mb_hat = mb / (1 - self.beta1 ** self.t)

            vw_hat = vw / (1 - self.beta2 ** self.t)
            vb_hat = vb / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * mw_hat / np.sqrt(vw_hat + self.eps) + self.weight_decay * layer.W * self.lr
            layer.b -= self.lr * mb_hat / np.sqrt((vb_hat + self.eps)) + self.weight_decay * layer.b * self.lr
