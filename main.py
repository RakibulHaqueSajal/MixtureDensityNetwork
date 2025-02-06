import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from Models.GNN import GaussianNN_MV
from data.data_generation import generate_2d_synthetic_data, generate_two_spirals
from Models.MDN import MDN
from exp.exp_main import train_model,evaluate_model
from utils.loss import gaussian_mv_loss, mdn_loss,mdn_mse_loss

# set all seeds to 0
torch.manual_seed(0)
np.random.seed(0)

#Generating the dataset
n_samples=1000
data, labels = generate_two_spirals()
labels=np.expand_dims(labels,axis=1)
print(data.shape)
print(labels.shape)
# Visualize the generated data
plt.figure(figsize=(8, 6))
scatter = plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Synthetic 2D Data from a Mixture of Gaussians')
plt.colorbar(scatter, label='Component Label')
plt.show()

#Splitting the dataset
x_train, y_test = train_test_split(data, test_size=0.2, random_state=42)
# Convert data to PyTorch tensors.
x_train_tensor = torch.tensor(x_train[:, 0], dtype=torch.float32)
x_train_label = torch.tensor(x_train[:, 1], dtype=torch.float32)
y_test_tensor = torch.tensor(y_test[:, 0], dtype=torch.float32)
y_test_label = torch.tensor(y_test[:, 1], dtype=torch.float32)

# Create DataLoaders.
train_dataset = TensorDataset(x_train_tensor, x_train_label)
test_dataset = TensorDataset(y_test_tensor, y_test_label)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


#Hyperparameters.
input_dim = 1
hidden_dim = 4
output_dim = 1   # Three-dimensional target.
n_epochs = 80
lr = 0.0001

# A. MDN Model

num_gaussians = 1
print("Training MDN (multivariate) ...")
mdn_model = MDN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_gaussians=num_gaussians)
# gnn_model = GaussianNN_MV(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
mdn_optimizer = optim.Adam(mdn_model.parameters(), lr=lr)
# gnn_optimizer = optim.Adam(gnn_model.parameters(), lr=lr)
mdn_model = train_model(mdn_model, mdn_loss, mdn_optimizer, train_loader, n_epochs=n_epochs)
# gnn_model = train_model(gnn_model, gaussian_mv_loss, gnn_optimizer, train_loader, n_epochs=n_epochs)
# gnn_train_loss, gnn_train_rmse, _, _ = evaluate_model(gnn_model, gaussian_mv_loss, train_loader)
# gnn_test_loss, gnn_test_rmse, gnn_preds, targets = evaluate_model(gnn_model, gaussian_mv_loss, test_loader)
# print(f"GNN Train Loss: {gnn_train_loss:.4f}, Train RMSE: {gnn_train_rmse:.4f}")
# print(f"GNN Test Loss: {gnn_test_loss:.4f}, Test RMSE: {gnn_test_rmse:.4f}")
mdn_train_loss, mdn_train_rmse, _, _ = evaluate_model(mdn_model, mdn_loss, train_loader)
mdn_test_loss, mdn_test_rmse, mdn_preds, targets = evaluate_model(mdn_model, mdn_loss, test_loader)
print(f"MDN Train Loss: {mdn_train_loss:.4f}, Train RMSE: {mdn_train_rmse:.4f}")
print(f"MDN Test Loss: {mdn_test_loss:.4f}, Test RMSE: {mdn_test_rmse:.4f}")

# mdn_model.eval()
# with torch.no_grad():
#     x_test_tensor = torch.tensor(x_test, dtype=torch.float32)  # If you don't already have it
#     pi_pred, mu_pred, L_pred = mdn_model(x_test_tensor)
# pred_cluster = torch.argmax(pi_pred, dim=1)  # shape: (n_test_samples,)
# pred_cluster = pred_cluster.cpu().numpy()

# import matplotlib.pyplot as plt

# # Convert test data to NumPy (if not already)
# x_test_np = x_test  # shape (n_test_samples, 2)
# y_test_np = y_test.squeeze()  # true labels, shape (n_test_samples,)

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # Left: Ground-Truth Labels
# sc1 = axes[0].scatter(
#     x_test_np[:, 0], x_test_np[:, 1],
#     c=y_test_np, cmap='viridis', alpha=0.6
# )
# axes[0].set_title("True Gaussian Component")
# axes[0].set_xlabel("Dim 1")
# axes[0].set_ylabel("Dim 2")
# fig.colorbar(sc1, ax=axes[0], label="True Label")

# # Right: Predicted Label via Argmax(pi)
# sc2 = axes[1].scatter(
#     x_test_np[:, 0], x_test_np[:, 1],
#     c=pred_cluster, cmap='viridis', alpha=0.6
# )
# axes[1].set_title("Predicted Mixture Component")
# axes[1].set_xlabel("Dim 1")
# axes[1].set_ylabel("Dim 2")
# fig.colorbar(sc2, ax=axes[1], label="Predicted Label")

# plt.tight_layout()
# plt.show()
