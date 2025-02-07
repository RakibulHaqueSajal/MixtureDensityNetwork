import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



def mdn_loss(pi, mu, L, y, epoch=0):
    """
    Computes the negative log-likelihood loss for a mixture of multivariate Gaussians.

    Args:
        pi: Tensor of shape (batch, num_gaussians)
        mu: Tensor of shape (batch, num_gaussians, output_dim)
        L: Tensor of shape (batch, num_gaussians, output_dim, output_dim), lower-triangular factors.
        y: Tensor of shape (batch, output_dim)
    Returns:
        Negative log-likelihood loss (scalar).
    """
    batch, num_components, D = mu.shape
    y_exp = y.unsqueeze(1).expand_as(mu)  # Shape: (batch, num_gaussians, output_dim)

    # Flatten batch and component dimensions:
    # mu_flat shape: (batch*num_gaussians, output_dim)
    mu_flat = mu.reshape(-1, D)
    # L_flat shape: (batch*num_gaussians, output_dim, output_dim)
    L_flat = L.reshape(-1, D, D)
    # y_flat shape: (batch*num_gaussians, output_dim)
    y_flat = y_exp.reshape(-1, D)

    # Create a multivariate normal distribution for each flattened component.
    mvn = torch.distributions.MultivariateNormal(loc=mu_flat, scale_tril=L_flat)
    log_prob_flat = mvn.log_prob(y_flat)  # Shape: (batch*num_gaussians,)
    # Reshape back to (batch, num_gaussians)
    log_prob = log_prob_flat.reshape(batch, num_components)

    weighted_log_prob = log_prob + torch.log(pi + 1e-8)
    log_sum = torch.logsumexp(weighted_log_prob, dim=1)
    return -torch.mean(log_sum)

def mdn_mse_loss(pi, mu, y):
    """
    Computes a mean squared error (MSE) loss for an MDN.
    
    Instead of using the negative log-likelihood of the mixture of Gaussians,
    we use the weighted average of the means (the expected value) as the prediction,
    and then compute the MSE between this prediction and the target y.
    
    Args:
        pi: Tensor of shape (batch, num_gaussians) -- mixing coefficients.
        mu: Tensor of shape (batch, num_gaussians, output_dim) -- means of each Gaussian.
        y: Tensor of shape (batch, output_dim) -- true target values.
    
    Returns:
        MSE loss (scalar).
    """
    # Compute the expected value of the mixture distribution.
    # pi.unsqueeze(-1) changes the shape to (batch, num_gaussians, 1) so that it broadcasts with mu.
    y_pred = torch.sum(pi.unsqueeze(-1) * mu, dim=1)  # Resulting shape: (batch, output_dim)
    
    # Compute the MSE loss between the predicted value and the true target.
    loss = torch.mean((y_pred - y) ** 2)
    return loss

def gaussian_mv_loss(mu, L, y):
    """
    Negative log-likelihood loss for a multivariate Gaussian.

    Args:
        mu: Tensor of shape (batch, output_dim)
        L: Tensor of shape (batch, output_dim, output_dim), lower-triangular matrix.
        y: Tensor of shape (batch, output_dim)
    Returns:
        Negative log-likelihood loss (scalar).
    """
    mvn = torch.distributions.MultivariateNormal(loc=mu, scale_tril=L)
    log_prob = mvn.log_prob(y)

    # print determinant of variance

    return -torch.mean(log_prob)