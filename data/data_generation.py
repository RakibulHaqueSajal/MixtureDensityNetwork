
import numpy as np 


def generate_2d_synthetic_data(n_samples=1000, seed=42):
    """
    Generates a synthetic 2D dataset from a mixture of three Gaussians.
    
    Args:
        n_samples (int): Total number of samples to generate.
        seed (int): Random seed for reproducibility.
    
    Returns:
        data (np.ndarray): Generated 2D data points of shape (n_samples, 2).
        labels (np.ndarray): Component labels for each data point (optional).
    """
    np.random.seed(seed)
    
    # Define the parameters for three Gaussian components
    means = np.array([
        [0, 0],
        [5, 5],
        [-5, 5]
    ])
    
    covariances = np.array([
        [[1.0, 0.8], [0.8, 1.0]],  # Component 1 covariance
        [[1.0, -0.6], [-0.6, 1.0]],  # Component 2 covariance
        [[1.0, 0.0], [0.0, 1.0]]    # Component 3 covariance
    ])
    
    # Mixing weights (they should sum to 1)
    weights = np.array([0.4, 0.4, 0.2])
    
    # Choose which component each sample comes from
    component_choices = np.random.choice(len(weights), size=n_samples, p=weights)
    
    # Allocate array for data points and (optionally) labels
    data = np.zeros((n_samples, 2))
    labels = np.zeros(n_samples, dtype=int)
    
    # Generate data for each sample based on its chosen component
    for i, comp in enumerate(component_choices):
        data[i, :] = np.random.multivariate_normal(mean=means[comp], cov=covariances[comp])
        labels[i] = comp  # Save the label if you want to analyze the mixture components
    
    return data, labels


def generate_two_spirals(n_points=1000, noise=0.5, random_seed=42):
    """
    Generates a 2D two-spirals dataset.
    
    Args:
        n_points (int): Total number of data points (will be split evenly between the two spirals).
        noise (float): Standard deviation of Gaussian noise added to the data.
        random_seed (int): Seed for reproducibility.
    
    Returns:
        data (np.ndarray): Generated data points of shape (n_points, 2).
        labels (np.ndarray): Labels for each data point indicating spiral membership (0 or 1).
    """
    np.random.seed(random_seed)
    n_points_per_spiral = n_points // 2

    # Generate an array of angles (theta) for the points
    theta = np.sqrt(np.random.rand(n_points_per_spiral)) * 4 * np.pi  # scale angles

    # First spiral: points from (r*cos(theta), r*sin(theta))
    r = theta
    x1 = r * np.cos(theta) + np.random.randn(n_points_per_spiral) * noise
    y1 = r * np.sin(theta) + np.random.randn(n_points_per_spiral) * noise

    # Second spiral: mirror image with a phase shift of pi
    x2 = r * np.cos(theta + np.pi) + np.random.randn(n_points_per_spiral) * noise
    y2 = r * np.sin(theta + np.pi) + np.random.randn(n_points_per_spiral) * noise

    # Combine the two spirals into one dataset
    data = np.vstack((np.column_stack((x1, y1)),
                      np.column_stack((x2, y2))))
    labels = np.hstack((np.zeros(n_points_per_spiral, dtype=int),
                        np.ones(n_points_per_spiral, dtype=int)))
    
    return data, labels

