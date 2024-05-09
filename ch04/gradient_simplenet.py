# coding: utf-8
import sys, os
# Append the parent directory to sys.path so that Python can find modules in the parent directory
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root_path = os.path.join(current_script_path, os.pardir)
sys.path.append(os.path.abspath(project_root_path))
# sys.path.append('/home/gyehwancho/nvme/ML_practice/dl_from_scratch/WegraLee-deep-learning-from-scratch')
import numpy as np
# Import softmax and cross_entropy_error functions from common functions module
from common.functions import softmax, cross_entropy_error
# Import numerical_gradient function from common gradient module
from common.gradient import numerical_gradient

# Define a simple neural network class called simpleNet
class simpleNet:
    def __init__(self):
        # Initialize weights W with a 2x3 matrix with random values from a normal distribution
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        # Compute the dot product of input x and weights W to get the output
        return np.dot(x, self.W)

    def loss(self, x, t):
        # Compute the predicted output for input x
        z = self.predict(x)
        # Apply the softmax function to the output to get the probabilities
        y = softmax(z)
        # Calculate the cross-entropy loss between the predicted probabilities and the true labels t
        loss = cross_entropy_error(y, t)
        return loss

# Example input vector
x = np.array([0.6, 0.9])
# Example target vector (one-hot encoded)
t = np.array([0, 0, 1])

# Create an instance of simpleNet
net = simpleNet()

# Define a lambda function that calculates the loss of the network for the weights W
f = lambda w: net.loss(x, t)
# Compute the gradient of the loss with respect to the weights W using numerical gradient estimation
dW = numerical_gradient(f, net.W)

# Print the gradient of the weights
print(dW)


# There is only one gradient descent step conducted here, and the weights are indeed represented by 2 x 3 numbwers