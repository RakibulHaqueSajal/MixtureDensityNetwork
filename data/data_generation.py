
import numpy as np 

def generate_one_spiral(n_points=1000, noise=0.5, random_seed=42):
    """
    Generates a 1D two-spirals dataset.
    
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

    # Combine the two spirals into one dataset
    data = np.vstack((np.column_stack((x1, y1))))
   
    
    return data


# Generate Sine Curve Function
def generate_sine_curve(n_points=1000, noise=0.1, random_seed=42):
    """
    Generates a 2D sine curve dataset.

    Args:
        n_points (int): Number of data points.
        noise (float): Standard deviation of Gaussian noise added to the data.
        random_seed (int): Seed for reproducibility.

    Returns:
        data (np.ndarray): Generated data points of shape (n_points, 2).
        labels (np.ndarray): Corresponding sine values with noise.
    """
    np.random.seed(random_seed)
    x = np.linspace(0, 4 * np.pi, n_points)  # Generate x values
    y = np.sin(x) + np.random.normal(0, noise, n_points)  # Generate noisy sine values

    data = np.column_stack((x, y))
    

    return data