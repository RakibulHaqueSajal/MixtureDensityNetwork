import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Models.MDN import MDN 
from Models.GNN import GaussianNN_MV
from Models.DNN import DeterministicNN



def train_model(model, loss_fn, optimizer, dataloader, n_epochs=100):
    """
    Generic training loop.
    Depending on the model type, the appropriate forward and loss computations are used.
    """
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.unsqueeze(-1)
            batch_y = batch_y.unsqueeze(-1)
            optimizer.zero_grad()
            if isinstance(model, MDN):
                pi, mu, L = model(batch_x)
                loss = loss_fn(pi, mu, L, batch_y)
            elif isinstance(model, GaussianNN_MV):
                mu, L = model(batch_x)
                loss = loss_fn(mu, L, batch_y)
            elif isinstance(model, DeterministicNN):
                pred = model(batch_x)
                loss = F.mse_loss(pred, batch_y)
            else:
                raise ValueError("Unknown model type")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")
    return model

def evaluate_model(model, loss_fn, dataloader):
    """
    Evaluate the model on the data from dataloader.
    Returns average loss, RMSE, predictions, variances, and targets.
    """
    model.eval()
    total_loss = 0.0
    predictions = []
    variances = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.unsqueeze(-1)
            batch_y = batch_y.unsqueeze(-1)
            
            if isinstance(model, MDN):
                pi, mu, L = model(batch_x)
                loss = loss_fn(pi, mu, L, batch_y)
                pred = torch.sum(pi.unsqueeze(-1) * mu, dim=1)  # Weighted average of the means
                var = torch.sum(pi.unsqueeze(-1) * torch.diagonal(L, dim1=-2, dim2=-1), dim=1)  # Weighted variance # Average variance per sample
                print(var.shape)
            elif isinstance(model, GaussianNN_MV):
                mu, L = model(batch_x)
                loss = loss_fn(mu, L, batch_y)
                pred = mu
                var = torch.mean(torch.diagonal(L, dim1=-2, dim2=-1), dim=-1)  # Average variance per sample
            elif isinstance(model, DeterministicNN):
                pred = model(batch_x)
                loss = F.mse_loss(pred, batch_y)
                var = torch.zeros_like(pred)  # No variance for deterministic model
            
            total_loss += loss.item() * batch_x.size(0)
            predictions.append(pred.cpu().numpy())
            variances.append(var.cpu().numpy())
            targets.append(batch_y.cpu().numpy())
    
    total_loss /= len(dataloader.dataset)
    predictions = np.concatenate(predictions, axis=0)
    variances = np.concatenate(variances, axis=0)
    targets = np.concatenate(targets, axis=0)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    return total_loss, rmse, predictions, variances, targets