import sys

sys.path.append("mytorch")

from Conv2d import *
from activation import *
from batchnorm2d import *

import numpy as np
import os

def zero_init(d1,d2,d3,d4):
    return np.zeros((d1,d2,d3,d4))

def bias_one_init(d1):
    return np.ones(d1)

def bias_minus_one_init(d1):
    return -1*np.ones(d1)

class ConvBlock(object):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.layers = [
            Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias_init_fn=None),
            BatchNorm2d(out_channels),
        ]

    def forward(self, A):
        x = A
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        x = grad
        for layer in reversed(self.layers):
            x = layer.backward(x)
        return x

class IdentityBlock(object):
    def forward(self, A):
        return A
	
    def backward(self, grad):
        return grad


class ResBlock(object):
    def __init__(self, in_channels, out_channels, filter_size, stride=3, padding=1):
        self.convolution_layers = [
			ConvBlock(in_channels, out_channels, filter_size, stride, padding),
            ReLU(),
            ConvBlock(out_channels, out_channels, 1, 1, 0),
        ]  # TODO Initialize all layers in this list.
        self.final_activation = ReLU()  

        if (
            stride != 1
            or in_channels != out_channels
            or filter_size != 1
            or padding != 0
        ):
            self.residual_connection = ConvBlock(in_channels, out_channels, filter_size, stride, padding)
        else:
            self.residual_connection = IdentityBlock()

    def forward(self, A):
        Z = A
        """
		Implement the forward for convolution layer.

		"""
        for layer in self.convolution_layers:
            Z = layer.forward(Z)

        """
		Add the residual connection to the output of the convolution layers

		"""
        residual_out = self.residual_connection.forward(A)

        """
		Pass the the sum of the residual layer and convolution layer to the final activation function
		"""
        return self.final_activation.forward(Z + residual_out)

    def backward(self, grad):
        
        """
        Implement the backward of the final activation
        """
        grad = self.final_activation.backward(grad)

        conv_grad = grad

        """
		Implement the backward of residual layer to get "residual_grad"
		"""
        residual_grad = self.residual_connection.backward(grad)

        # print("residual_grad", residual_grad.shape)

        """
		Implement the backward of the convolution layer to get "convlayers_grad"
		"""
        for i, layer in enumerate(reversed(self.convolution_layers)):
            # print(f"conv_grad{i}", conv_grad.shape)
            conv_grad = layer.backward(conv_grad)
        """
		Add convlayers_grad and residual_grad to get the final gradient 
		"""
        return conv_grad + residual_grad

