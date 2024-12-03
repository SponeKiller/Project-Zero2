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
    run_parallel: str = None
    batch_size: int = 8
    num_epochs: int = 20
    num_classes: int = 10
    lr: int = 10**-4
    train_steps: int = 1
    eval_steps: int = 10
    train_data: str = "/train/train_data/"
    lambda_1: int = 0.01
    lambda_2: int = 0.01
    beta1 = 0
    beta2 = 0
    epsilon = 1**-9
    weight_decay = 0
    amsgrad = False
    train_ds_size = 0.9
    



