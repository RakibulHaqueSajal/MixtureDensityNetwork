import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def vector_to_lower_triangular(vec, D):
    """
    Converts a tensor of shape (..., D*(D+1)//2) into a lower-triangular matrix of shape (..., D, D).
    The diagonal elements are exponentiated (or can use softplus) to guarantee positivity.

    Args:
        vec: Tensor of shape (..., D*(D+1)//2)
        D: int, target dimension.

    Returns:
        L: Tensor of shape (..., D, D) which is lower triangular with positive diagonal.
    """
    new_shape = vec.shape[:-1] + (D, D)
    L = torch.zeros(new_shape, device=vec.device, dtype=vec.dtype)

    tril_indices = torch.tril_indices(row=D, col=D, offset=0)
    L[..., tril_indices[0], tril_indices[1]] = vec
    # Exponentiate the diagonal entries to ensure they are positive.
    diag_idx = torch.arange(D, device=vec.device)
    L[..., diag_idx, diag_idx] = F.softplus(L[..., diag_idx, diag_idx])
    return L


class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gaussians):
        """
        Args:
            input_dim: Dimension of the input.
            hidden_dim: Number of hidden units.
            output_dim: Dimension of the target output (multivariate).
            num_gaussians: Number of mixture components.
        """
        super(MDN, self).__init__()
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians

        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        # Mixing coefficients for each component.
        self.pi = nn.Linear(hidden_dim, num_gaussians)
        # Means: output a mean vector for each mixture component.
        self.mu = nn.Linear(hidden_dim, num_gaussians * output_dim)
        # Covariance: for multivariate output, we parameterize the lower-triangular matrix.
        self.L_params = nn.Linear(hidden_dim, num_gaussians * (output_dim * (output_dim + 1) // 2))

    def forward(self, x):
        h = self.hidden(x)
        # Calculate mixing coefficients and apply softmax.
        pi = F.softmax(self.pi(h), dim=1)  # Shape: (batch_size, num_gaussians)

        # Calculate means and reshape to (batch_size, num_gaussians, output_dim)
        mu = self.mu(h).reshape(-1, self.num_gaussians, self.output_dim)
        # Comment: h has shape (batch_size, hidden_dim) and self.mu(h) outputs a tensor of shape
        # (batch_size, num_gaussians * output_dim). After reshaping, mu becomes (batch_size, num_gaussians, output_dim).

        # Calculate covariance parameters and reshape.
        L_params = self.L_params(h)
        # Reshape to (batch_size, num_gaussians, output_dim*(output_dim+1)//2)
        L_params = L_params.reshape(-1, self.num_gaussians, self.output_dim * (self.output_dim + 1) // 2)
        # Comment: Each mixture component now has a parameter vector of length output_dim*(output_dim+1)//2.

        # Convert each parameter vector to a lower-triangular matrix.
        L = vector_to_lower_triangular(L_params, self.output_dim)
        # L shape: (batch_size, num_gaussians, output_dim, output_dim)

        return pi, mu, L