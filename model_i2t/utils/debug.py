from typing import Optional

import torch
from pathlib import Path
from datetime import datetime


from model_i2t.main import VisionTransformer

class Debug:
    """
    Debugging utilities
    """
    def __init__(
        self,
        num_epochs: int,
        num_train_batches: int,
        num_eval_batches: int,
        file_path: str,
        enable: bool = True
    ):
        """
        Inicialize Debugging utilities
        
        Args:
            num_epochs (int): The number of epochs
            num_train_batches (int): The number of training batches
            num_eval_batches (int): The number of evaluation batches
            file_path (str): The file path to write debug information
            enable (bool): Enable debugging
        
        """
        self.enable = enable
        self.num_epochs = num_epochs
        self.num_train_batches = num_train_batches
        self.num_eval_batches = num_eval_batches
        self.loss = [[] for _ in range(num_epochs)]
        self.total_loss = 0
        self.accuracy = [[] for _ in range(num_epochs)]
        self.total_accuracy = 0
        
        
        # Create file to write debug information 
        path = Path(file_path)
        
        file_name = path.stem
        file_suffix = path.suffix
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.file_path = f"{file_name}_{current_time}{file_suffix}"
        
        if path.is_dir():
            self.file_path = path.parent / self.file_path
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        
        
    def render_loss(self, 
                    loss: torch.Tensor,
                    curr_batch: int, 
                    curr_epoch: int,
                    avg_per_batches: int = 1, 
                    avg_per_epochs: int = 1,
                    type: str = "train",):
        """
        Render the loss
        
        Args:
            loss (torch.Tensor): The loss
            curr_batch (int): The current batch
            curr_epoch (int): The current epoch
            avg_per_batches (int): The average per batches
            avg_per_epochs (int): The average per epochs
            type (str): The type of loss (train, eval)
        
        """
        
        if self.enable is False:
            return
        
        if type not in ["train", "validation"]:
            type = "train"
            
        if type == "train":
            num_batches = self.num_train_batches
        else:
            num_batches = self.num_eval_batches
  
        print(f"{type.upper()} LOSS: {loss}")
 
        self.loss[curr_epoch].append(loss)
        
        if curr_batch > avg_per_batches:
            print(
                f"AVG {type.upper()} LOSS PER BATCH {avg_per_batches}: "
                f"{sum(self.loss[curr_epoch][curr_batch - avg_per_batches:curr_batch]) / avg_per_batches}"
            )
              
        if curr_epoch >= avg_per_epochs and curr_batch == 0:
            print(
                f"AVG {type.upper()} LOSS PER EPOCH {avg_per_epochs}: "
                f"{sum(self.loss[curr_epoch - avg_per_epochs:curr_epoch]) / avg_per_epochs}"
            )
            
        if num_batches == curr_batch:
            print(f"FINAL AVG {type.upper()} BATCH LOSS : "
                  f"{sum(self.loss[curr_epoch]) / num_batches}"
            )
        
        if self.num_epochs == curr_epoch:
            print(f"FINAL AVG {type.upper()} EPOCH LOSS: "
                  f"{sum(self.loss) / self.num_epochs}"
            )
                
    def render_accuracy(self, 
                        prediction: torch.Tensor, 
                        labels: torch.Tensor,
                        curr_batch: int,
                        curr_epoch: int,
                        avg_per_batches: int = 1, 
                        avg_per_epochs: int = 1,
                        type: str = "train",
                        show_prediction: bool = True,):
        """
        Render the accuracy
        
        Args:
            prediction (torch.Tensor): The prediction
            labels (torch.Tensor): The labels
            curr_batch (int): The current batch
            curr_epoch (int): The current epoch
            avg_per_batches (int): The average per batches
            avg_per_epochs (int): The average per epochs
            type (str): The type of accuracy (train, eval)
            show_prediction (bool): Show the prediction

        """

        if self.enable is False:
            return
        
        if type not in ["train", "eval"]:
            type = "train"
        
        if type == "train":
            num_batches = self.num_train_batches
        else:
            num_batches = self.num_eval_batches
            
        
        if show_prediction:
            print(f"{type.upper()} PREDICTION: {prediction}")
            print(f"{type.upper()} LABELS: {labels}")
        
        correct = (prediction == labels).sum().item()
        
        print(f"{type.upper()} ACCURACY: "
              f"{(correct / len(prediction)) * 100} %"
        )
        
        self.accuracy[curr_epoch].append(correct / len(prediction))
        
        if curr_batch > avg_per_batches:
            print(
                f"AVG {type.upper()} ACCURACY PER BATCH {avg_per_batches}: "
                f"{(sum(self.accuracy[curr_epoch][curr_batch - avg_per_batches:curr_batch]) / avg_per_batches) * 100} %"
            )
        
        if curr_epoch >= avg_per_epochs and curr_batch == 0:
            print(
                f"AVG {type.upper()} ACCURACY PER EPOCH {avg_per_epochs}: "
                f"{(sum(self.accuracy[curr_epoch - avg_per_epochs:curr_epoch]) / avg_per_epochs) * 100} %"
            )
            
        if num_batches == curr_batch:
            print(
                f"FINAL AVG {type.upper()} BATCH ACCURACY : "
                f"{(sum(self.accuracy[curr_epoch]) / num_batches) * 100} %"
            )
        
        if self.num_epochs == curr_epoch:
            print(
                f"FINAL AVG {type.upper()} EPOCH ACCURACY: "
                f"{(sum(self.accuracy) / self.num_epochs) * 100} %"
            )


    def write_to_file(self, message: str, heading: Optional[str] = None):
        """
        Write a message to the debug file
        
        Args:
            message (str): The message to write
            heading (Optional[str]): The heading of the message
        """
        if self.enable is False:
            return
        
        heading_len = len(heading) if heading is not None else 0
        
        with open(self.file_path, "a") as file:
            if heading is not None:
                file.write("+" + "-" * (heading_len + 2) + "+")
                file.write("\n")
                file.write("| " + heading + " |")
                file.write("\n")
                file.write("+" + "-" * (heading_len + 2) + "+")
                file.write("\n\n")
            
            file.write(message)
            file.write("\n")