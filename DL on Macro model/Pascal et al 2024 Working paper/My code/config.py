"""
config.py - Configuration management for RBC model solution

This module provides configuration classes for the various component of RBC model solution framework.

"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable, Union, List, Tuple, Optional

#_________________________________________________________________________________________________________________________________#
#                           RBC Parameters                                                                                        #
#_________________________________________________________________________________________________________________________________#

@dataclass
class RbcParams:
    """
    RBC model parameters
    """

    # Core Parameters
    # These parameters are initialize automatically via @dataclass
    
    alpha: float = 0.36            # Capital share in production function
    eta: float = 0.34              # Labor weight in utility
    rho: float = 0.918             # Productivity shock persistence
    beta: float = 0.96             # Discount factor
    delta: float = 0.1             # Depreciation rate
    
    # Shock parameters
    sigma_e: float = 0.014         # Standard deviation of TFP innovation

    # After initialization, the following parameters are also computed
    def __post_init__(self):
        """
        Derived parameters calculated after initialization
        """
        self.one_minus_alpha = 1.0 - self.alpha
        
        # Parameter for the full depreciation case solution
        self.gamma = ((1.0 - self.alpha) * self.eta) / ((1.0 - self.alpha * self.beta) * (1 - self.eta))
        
        # Labor supply in full depreciation case
        self.n_constant = self.gamma / (1 + self.gamma)
        
        # Mean of innovation (usually zero)
        self.mu_e = 0.0

    def production_fn(self, a, k_prev, n):
        """Production function of the RBC model.
        
        Args:
            a: Productivity level
            k_prev: Capital from previous period
            n: Labor input
            
        Returns:
            Output level
        """
        return a * (k_prev ** self.alpha) * (n ** self.one_minus_alpha)


    def marginal_product_capital(self, a, k_prev, n):
        """Marginal product of capital.
        
        Args:
            a: Productivity level
            k_prev: Capital from previous period
            n: Labor input
            
        Returns:
            Marginal product of capital
        """
        return self.alpha * a * ((n / k_prev) ** self.one_minus_alpha)

     def utility_fn(self, c):
        """Utility function (logarithmic).
        
        Args:
            c: Consumption level
            
        Returns:
            Utility value
        """
        return np.log(c)
    
    def marginal_utility(self, c):
        """Marginal utility of consumption.
        
        Args:
            c: Consumption level
            
        Returns:
            Marginal utility
        """
        return self.eta / c
   

#_________________________________________________________________________________________________________________________________#
#                           Neural Network Parameters                                                                             #
#_________________________________________________________________________________________________________________________________#

@dataclass
class NetworkParams:
    """Neural network architecture and training parameters."""
    
    # Architecture
    input_dim: int = 2             # Input dimension (k_{t-1}, a_t)
    hidden_dims: List[int] = None  # Hidden layer dimensions
    output_dim: int = 1            # Output dimension (consumption share)
    activation: str = "sigmoid"    # Activation function
    dropout_rate: float = 0.0      # Dropout rate
    
    # Training parameters
    learning_rate: float = 1e-3    # Learning rate
    batch_size: int = 100          # Batch size
    num_epochs: int = 5000         # Number of training epochs
    optimizer: str = "Adam"        # Optimizer (Adam, SGD)
    
    def __post_init__(self):
        """Set default values and validate parameters."""
        # Default hidden dimensions if not provided
        if self.hidden_dims is None:
            self.hidden_dims = [16]
            
        # Validate activation function
        valid_activations = ["relu", "tanh", "sigmoid"]
        if self.activation.lower() not in valid_activations:
            raise ValueError(f"Activation must be one of: {valid_activations}")
            
        # Validate optimizer
        valid_optimizers = ["adam", "sgd", "swa"]
        if self.optimizer.lower() not in valid_optimizers:
            raise ValueError(f"Optimizer must be one of: {valid_optimizers}")



#_________________________________________________________________________________________________________________________________#
#                          Numerical Parameters                                                                                   #
#_________________________________________________________________________________________________________________________________#

@dataclass
class NumericalParams:
    """Parameters for numerical methods."""
    
    # Integration parameters
    num_mc_samples: int = 1000     # Number of Monte Carlo samples
    gauss_hermite_order: int = 5   # Order for Gauss-Hermite quadrature
    use_sobol: bool = False        # Whether to use Sobol sequences
    
    # Accuracy evaluation
    accuracy_samples: int = 1000   # Number of samples for accuracy evaluation
    euler_tolerance: float = 1e-6  # Tolerance for Euler equation residuals




#_________________________________________________________________________________________________________________________________#
#                         Complete Configuration                                                                                  #
#_________________________________________________________________________________________________________________________________#

class ModelConfig:
    """Complete configuration for the RBC model solution."""
    
    def __init__(
        self,
        economic_params: EconomicParams = None,
        network_params: NetworkParams = None,
        numerical_params: NumericalParams = None,
        device: str = "cpu",
        random_seed: int = 123,
        output_dir: str = "./output"
    ):
        """Initialize the configuration.
        
        Args:
            economic_params: Economic model parameters
            network_params: Neural network parameters
            numerical_params: Numerical method parameters
            device: Computation device (cpu or cuda)
            random_seed: Random seed for reproducibility
            output_dir: Directory for saving outputs
        """
        # Use defaults if not provided
        self.economic = economic_params or EconomicParams()
        self.network = network_params or NetworkParams()
        self.numerical = numerical_params or NumericalParams()
        
        self.device = device
        self.random_seed = random_seed
        self.output_dir = output_dir
    
    def set_random_seed(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            
    def save(self, filepath):
        """Save configuration to a file."""
        import json
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert to dictionary
        config_dict = {
            "economic": self.economic.__dict__,
            "network": self.network.__dict__,
            "numerical": self.numerical.__dict__,
            "device": self.device,
            "random_seed": self.random_seed,
            "output_dir": self.output_dir
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
            
    @classmethod
    def load(cls, filepath):
        """Load configuration from a file."""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            
        # Create parameter objects
        economic = EconomicParams(**config_dict["economic"])
        network = NetworkParams(**config_dict["network"])
        numerical = NumericalParams(**config_dict["numerical"])
        
        # Create and return configuration
        return cls(
            economic_params=economic,
            network_params=network,
            numerical_params=numerical,
            device=config_dict["device"],
            random_seed=config_dict["random_seed"],
            output_dir=config_dict["output_dir"]
        )
    

