# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class Dropout(object):
    def __init__(self, p=0.5):
        self.p = 1 - p
        self.mask = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):

        if train:
            # TODO: Generate mask and apply to x

            self.mask = np.random.binomial(1, self.p, x.shape)
            # print("mask", self.mask)
            return x * self.mask / self.p
            
        else:
            # TODO: Return x as is

            return x
		
    def backward(self, delta):
        # TODO: Multiply mask with delta and return

        return delta * self.mask