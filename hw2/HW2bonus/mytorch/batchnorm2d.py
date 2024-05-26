# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class BatchNorm2d:

    def __init__(self, num_features, alpha=0.9):
        # num features: number of channels
        self.alpha = alpha
        self.eps = 1e-8

        self.Z = None
        self.NZ = None
        self.BZ = None

        self.BW = np.ones((1, num_features, 1, 1))
        self.Bb = np.zeros((1, num_features, 1, 1))
        self.dLdBW = np.zeros((1, num_features, 1, 1))
        self.dLdBb = np.zeros((1, num_features, 1, 1))

        self.M = np.zeros((1, num_features, 1, 1))
        self.V = np.ones((1, num_features, 1, 1))

        # inference parameters
        self.running_M = np.zeros((1, num_features, 1, 1))
        self.running_V = np.ones((1, num_features, 1, 1))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """

        self.Z = Z
        if not eval:
            # Compute mean and variance for each channel
            self.M = np.mean(Z, axis=(0, 2, 3), keepdims=True)
            self.V = np.var(Z, axis=(0, 2, 3), keepdims=True)
            
            # Update running mean and variance
            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V

            # Normalize
            self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)
        else:
            # Use running mean and variance for normalization during inference
            self.NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
        
        # Apply scale and shift (gamma and beta)
        self.BZ = self.BW * self.NZ + self.Bb
        return self.BZ

    def backward(self, dLdBZ):
        N, C, H, W = self.Z.shape
        # print(self.NZ.shape)
        NZ = (self.Z - self.M) / np.sqrt(self.V + self.eps)
        self.dLdBW = np.sum(dLdBZ * NZ, axis=(0, 2, 3), keepdims=True)
        
        self.dLdBb = np.sum(dLdBZ, axis=(0, 2, 3), keepdims=True)

        # Gradient w.r.t. normalized data
        dLdNZ = dLdBZ * self.BW
        
        # Intermediate gradients for variance
        dLdV = -0.5 * np.sum(dLdNZ * (self.Z - self.M), axis=(0, 2, 3), keepdims=True) * (self.V + self.eps)**(-1.5)
        
        # Intermediate gradients for mean
        dLdM = -np.sum(dLdNZ, axis=(0, 2, 3), keepdims=True) / np.sqrt(self.V + self.eps) + dLdV * np.mean(-2 * (self.Z - self.M), axis=(0, 2, 3), keepdims=True)

        # Gradient w.r.t. input
        dLdZ = dLdNZ / np.sqrt(self.V + self.eps) + dLdV * 2 * (self.Z - self.M) / (N * H * W) + dLdM / (N * H * W)
        
        return dLdZ