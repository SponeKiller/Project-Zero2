from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainArgs:
    """
    Train model configurations parameters 

    Args:
        run_parallel (str): Run parallel training option (FSDP, DDP, PP (Pipeline paralelism))
        batch_size (int): Amount of sequences put to one traing cycle
        num_epochs (int): Number of training cycle
        lr (int): Learning rate of model
        train_dta (str): path to training data
    Returns:
        [int], [str] : model config parameters

    """
    batch_train_size: int = 50
    batch_eval_size: int = 5
    num_epochs: int = 20
    num_classes: int = 10
    
    #Optimizer
    lr: int = 10**-4
    beta1 = 0
    beta2 = 0
    epsilon = 1**-9
    weight_decay = 0
    amsgrad = False
    
    train_ds_size = 0.9
    train_data: str = "/train/train_data/"
    
    preload: bool = True
    model_path: str = "/model/"
    model_name: str = "VisionTransformer"
    
    label_smoothing: float = 0.1
    augment: bool = False  # Augment dataset
    



