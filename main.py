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
from data.data_generation import generate_2d_synthetic_data, generate_two_spirals
from Models.MDN import MDN
from exp.exp_main import train_model, evaluate_model
from utils.loss import gaussian_mv_loss, mdn_loss, mdn_mse_loss


torch.manual_seed(21)
np.random.seed(21)

# Data Generation
def generate_data():
    n_samples = 1000
    data, labels = generate_two_spirals()
    labels = np.expand_dims(labels, axis=1)
    return data, labels

# Data Preprocessing
def preprocess_data(data, labels):
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

# Model Definition
def define_model(input_dim, hidden_dim, output_dim, num_gaussians):
    mdn_model = MDN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_gaussians=num_gaussians)
    return mdn_model


# Main Function
def main():
    # Generate data
    data, labels = generate_data()
    
    # Preprocess data
    x_train_tensor, x_train_label, y_test_tensor, y_test_label = preprocess_data(data, labels)
    
    # Create DataLoaders
    train_loader, test_loader = create_data_loaders(x_train_tensor, x_train_label, y_test_tensor, y_test_label)
    
    # Define model
    input_dim = 1
    hidden_dim = 4
    output_dim = 1
    num_gaussians = 1
    mdn_model = define_model(input_dim, hidden_dim, output_dim, num_gaussians)
    
    # Train model
    mdn_optimizer = optim.Adam(mdn_model.parameters(), lr=0.0001)
    mdn_model = train_model(mdn_model, mdn_loss, mdn_optimizer, train_loader, n_epochs=80)
    
    # Evaluate model
    mdn_train_loss, mdn_train_rmse, _, _ = evaluate_model(mdn_model, mdn_loss, train_loader)
    mdn_test_loss, mdn_test_rmse, mdn_preds, targets = evaluate_model(mdn_model, mdn_loss, test_loader)
    
    # Print results
    print(f"MDN Train Loss: {mdn_train_loss:.4f}, Train RMSE: {mdn_train_rmse:.4f}")
    print(f"MDN Test Loss: {mdn_test_loss:.4f}, Test RMSE: {mdn_test_rmse:.4f}")

    visualize_results(y_test_tensor, y_test_label, mdn_preds)
    
def visualize_results(y_test_tensor, y_test_label, mdn_preds):
   # Ensure tensors are on CPU and converted to numpy for plotting
    y_test_tensor = y_test_tensor.cpu().detach().numpy()
    y_test_label = y_test_label.cpu().detach().numpy()
    mdn_preds = mdn_preds.squeeze()# Convert from (1000,1) to (1000,)

    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_tensor, y_test_label, label="True Labels", color="blue", alpha=0.6)
    plt.scatter(y_test_tensor, mdn_preds, label="MDN Predictions", color="red", alpha=0.6)
    plt.xlabel("y_test_tensor (X-axis)")
    plt.ylabel("Values (Y-axis)")
    plt.title("True Labels vs MDN Predictions")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()