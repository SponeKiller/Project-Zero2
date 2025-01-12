from torch import nn
import torch


class Classification(nn.Module):
    
    """
    Classification
    """
    
    def __init__(self, 
                 class_type: str,
                 d_model: int, 
                 num_classes: int) -> None:
        
        """
        Inicialize Classification
        
        Args:
        class_type: str - "token" or "gap"
        d_model: int - dimension of the model
        num_classes: int - number of classes
        
        """
        
        super().__init__()
        
        self.class_type = class_type
        self.d_model = d_model
       
            
        self.classifier = nn.Linear(d_model, num_classes)
        

    def forward(self, x: torch.Tensor) -> list[str]:
        
        """
        Forward pass of Classification
        
        Args:
        x: torch.Tensor - input tensor
        
        Returns:
            list[str] - prediction
        
        
        """
        if self.class_type == "token":
            x = x[:, 0]
                
        elif self.class_type == "gap":
            x = torch.mean(x, 
                           dim=list(range(1, len(x.shape) - 1)), 
                           keepdim=False)
        
        x = self.classifier(x)
        
        
        if (self.training == False):
            # Return prediction if not training mode
            x = torch.argmax(x, dim=-1)
        
        return x
        
        
        

