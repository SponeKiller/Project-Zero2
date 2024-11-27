from torch import nn
import torch
from typing import Optional


class Classification(nn.Module):
    
    """
    Classification
    """
    
    def __init__(self, 
                 class_type: str,
                 labels: list[str], 
                 d_model: Optional[int] = None, 
                 num_classes: Optional[int] = None) -> None:
        
        """
        Inicialize Classification
        
        Args:
        class_type: str - "token" or "gap"
        labels: list[str] - list of labels
        d_model: Optional[int] - dimension of the model
        num_classes: Optiona[int] - number of classes
        
        """
        
        super().__init__()
        
        self.class_type = class_type
        self.d_model = d_model
        self.labels = labels
        
       
            
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
            
        predictions = torch.argmax(self.classifier(x), dim=-1)
        
        return [self.labels[prediction] for prediction in predictions]
        
        
        

