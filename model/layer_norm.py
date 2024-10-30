from torch import nn
import torch

class LayerNorm(nn.Module):
    """
    Layer Normalization
    
    """
    
    def __init__(self,
                 features: int,
                 eps: int = 1e-6) -> None:
        """
        Initializes the Layer normalization module    

        Args:
            features (int): Number of features in
            the input tensor.
            
            eps (int): Epsilon value to avoid 
            division by zero.
        
        """
        
        super().__init__()
        
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Foward pass of Layer normalization     

        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: After normalization
        """
        
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return  ((x - mean) / (std + self.eps)) * self.a + self.b