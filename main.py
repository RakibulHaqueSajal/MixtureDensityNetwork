# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from Models.GNN import GaussianNN_MV
from data.data_generation import generate_2d_synthetic_data, generate_two_spirals,generate_sine_curve
from Models.MDN import MDN
from exp.exp_main import train_model, evaluate_model
from utils.loss import gaussian_mv_loss, mdn_loss, mdn_mse_loss


torch.manual_seed(21)
np.random.seed(21)

# Data Generation
def generate_data():
    n_samples = 1000
    data = generate_sine_curve(n_points=n_samples)
    return data

# Data Preprocessing
def preprocess_data(data):
    x_train, y_test = train_test_split(data, test_size=0.2, random_state=42)
    x_train_tensor = torch.tensor(x_train[:, 0], dtype=torch.float32)
    x_train_label = torch.tensor(x_train[:, 1], dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test[:, 0], dtype=torch.float32)
    y_test_label = torch.tensor(y_test[:, 1], dtype=torch.float32)
    return x_train_tensor, x_train_label, y_test_tensor, y_test_label

# Create DataLoaders
def create_data_loaders(x_train_tensor, x_train_label, y_test_tensor, y_test_label):
    train_dataset = TensorDataset(x_train_tensor, x_train_label)
    test_dataset = TensorDataset(y_test_tensor, y_test_label)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader



  
def visualize_results(y_test_tensor, y_test_label, predictions, variances):
    y_test_tensor = y_test_tensor.cpu().detach().numpy()
    y_test_label = y_test_label.cpu().detach().numpy()
    predictions = predictions.squeeze()
    std_dev = np.sqrt(variances.squeeze())  # Convert variance to standard deviation

    # Sort for smooth curve visualization
    sorted_indices = np.argsort(y_test_tensor)
    y_test_tensor = y_test_tensor[sorted_indices]
    y_test_label = y_test_label[sorted_indices]
    predictions = predictions[sorted_indices]
    std_dev = std_dev[sorted_indices]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_tensor, y_test_label, label="True Labels", color="blue", alpha=0.6)
    plt.plot(y_test_tensor, predictions, label="Predicted Mean", color="red")
    plt.fill_between(
        y_test_tensor,
        predictions - 1 * std_dev,
        predictions + 1 * std_dev,
        color="red",
        alpha=0.3,
        label="Confidence Interval (±σ)"
    )
    plt.xlabel("y_test_tensor (X-axis)")
    plt.ylabel("Values (Y-axis)")
    plt.title("True Labels vs Gaussian Predictions with Variance")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Generate data
    data = generate_data()
    
    # Preprocess data
    x_train_tensor, x_train_label, y_test_tensor, y_test_label = preprocess_data(data)
    
    # Create DataLoaders
    train_loader, test_loader = create_data_loaders(x_train_tensor, x_train_label, y_test_tensor, y_test_label)
    
    # Define model
    input_dim = 1
    hidden_dim = 128
    output_dim = 1
    num_gaussians = 3
    mdn_model = MDN(input_dim, hidden_dim, output_dim, num_gaussians)
    #mdn_model = GaussianNN_MV(input_dim, hidden_dim, output_dim)
    
    # Train model
    mdn_optimizer = optim.Adam(mdn_model.parameters(), lr=0.0001)

    criterion=mdn_loss
    mdn_model = train_model(mdn_model,criterion, mdn_optimizer, train_loader, n_epochs=800)
    
   # Evaluate model
    mdn_train_loss, mdn_train_rmse, _, _, _ = evaluate_model(mdn_model, criterion, train_loader)
    mdn_test_loss, mdn_test_rmse, mdn_preds, variances, targets = evaluate_model(mdn_model, criterion, test_loader)

      # Print results
    print(f"Train Loss: {mdn_train_loss:.4f}, Train RMSE: {mdn_train_rmse:.4f}")
    print(f"Test Loss: {mdn_test_loss:.4f}, Test RMSE: {mdn_test_rmse:.4f}")

     # Visualize predictions with variance
    visualize_results(y_test_tensor, y_test_label, mdn_preds, variances)