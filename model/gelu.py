from torch import nn
import torch
import math

class Gelu(nn.Module):
    
    """ 
    Gaussian Error Linear Units

    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass of the GELU activation function.       

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        return (
            0.5 * x * 
            (
                1 + torch.tanh(
                    math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
                    )
            )
        )
