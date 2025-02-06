import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Models.MDN import vector_to_lower_triangular

class GaussianNN_MV(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        This model predicts a multivariate Gaussian by outputting the mean vector and
        the parameters for the lower-triangular matrix (Cholesky factor) for the covariance.
        """
        super(GaussianNN_MV, self).__init__()
        self.output_dim = output_dim

        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, output_dim)
        # Output the parameters for the lower-triangular matrix.
        self.L_params = nn.Linear(hidden_dim, output_dim * (output_dim + 1) // 2) # variance

    def forward(self, x):
        h = self.hidden(x)
        # Mean: shape (batch_size, output_dim)
        mu = self.mu(h)
        # L_params: shape (batch_size, output_dim*(output_dim+1)//2)
        L_params = self.L_params(h)
        # Convert to lower-triangular matrix: shape (batch_size, output_dim, output_dim)
        L = vector_to_lower_triangular(L_params, self.output_dim)
        return mu, L