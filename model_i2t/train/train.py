import json
from typing import List, Tuple
from pathlib import Path
import pickle
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
import numpy as np

from model_i2t.main import VisionTransformer
from model_i2t.train.config import TrainArgs
from model_i2t.train.datasets import Train_Dataset
from model_i2t.utils.debug import Debug



class Train():
    
    """
    Train model
    
    """

    def __init__(self, model:VisionTransformer, config:TrainArgs):
        
        """
        Inicialize model training
        
        Args:
            model (VisionTransformer): Model to train
            config (TrainArgs): Configuration for training model
        """
        
        self.model: VisionTransformer = model
        self.config: TrainArgs = config
        self.device: torch.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self.base_dir = Path(__file__).resolve().parent
        self.model_filename = self.base_dir / self._get_weights_file_path()
        self.optimizer: torch.optim.AdamW = torch.optim.AdamW(
            self.model.parameters(),
            betas=(self.config.beta1, self.config.beta2),
            lr=self.config.lr,
            eps=self.config.epsilon,
            weight_decay=self.config.weight_decay,
            amsgrad=self.config.amsgrad
        )
        
        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        ).to(self.device)
        self.model.to(self.device, dtype=self.config.dtype)
        
        self.debug = Debug(self.config.num_epochs, 
                           self.base_dir / self.config.debug_path,            self.config.debug)
        

    def pretrain_training(self):
        
        train_ds, validation_ds = self._set_dataset(
            Train_Dataset)

        initial_epoch = 0
        best_val_accuracy = 0

        for epoch in range(initial_epoch, self.config.num_epochs):
            
            torch.cuda.empty_cache()
            
            self._load_model_state()
            
            self._run_training(train_ds, epoch)

            val_accuracy = self._run_validation(validation_ds, epoch)

            if (
                val_accuracy > best_val_accuracy or 
                self.model_filename.is_file() == False
            ):
                # Save new best model           
                best_val_accuracy = val_accuracy
                self._save_model_state()
                
                
    def finetune_training(self):
    
        """
        Finetune training
        """       
        
        print("Currently not implemented")
        return
    
    def reinforce_training(self):
        """
        Reinforce learinig 
        """
        print("Currently not implemented")
        return

        
        
    def _set_dataset(self,
                     dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
        
        """
        Set dataset
        
        Args:
            dataset (Dataset): Dataset to use for training
        """

        #loading data
        ds_raw = self._load_dataset()
             
        if self.config.augment:
            self.augment_dataset()
        
        # Size of the training/validation dataset
        
        num_examples: np.ndarray = ds_raw['data'].shape[0]
        
        train_ds_size = int(
            self.config.train_ds_size * 
            num_examples
        )
        
        val_ds_size = int(num_examples - train_ds_size)

        
        dataset = dataset(ds_raw, dtype=self.config.dtype)
        
        train_ds, val_ds = random_split(dataset, [train_ds_size, val_ds_size])


        train_dataloader = DataLoader(train_ds,
                                      batch_size=self.config.batch_train_size, shuffle=True)
        
        
        val_dataloader = DataLoader(val_ds,
                                    batch_size=self.config.batch_eval_size, shuffle=True)
            
        
        print("Dataset has been successfully loaded")

        return train_dataloader, val_dataloader
    

    def _load_dataset(self) -> List[str]:

        """
        Load dataset from csv file

        Args:
            path (str): Path to the directory containing csv files.
            
        Returns:
            List[str] - data from csv file.

        Raises:
            AssertionError: if in provided path wont find any jsonl files.
        """
        
        file_path = self.base_dir / self.config.dataset_path
        
        assert file_path.is_file(), (
            f"File not found in directory {str(file_path)}"
        )
        

        ## Loading based on file extension
        
        match file_path.suffix:
            
            case ".jsonl":
                return self._load_json(file_path)
            case ".json":
                return self._load_json(file_path)
            case ".pkl":
                return self._load_pkl(file_path)
            case _:
                raise NotImplementedError(
                    f"Dataset loader does not support"
                    f"{file_path.suffix} file extension"
                )
        
    def augment_dataset(self):
        
        """
        Augment dataset
        
        Note:
        
        User can choose to augment dataset or not   
        If user choose to augment dataset, Augmentation class will be called
        
        """
        print("Currently not implemented")
        return
        
        augment = Augmentation(ds_raw)
        ds_raw = augment.augment_ds      

    def _run_training(self, 
                      train_ds: torch.Tensor, 
                      epoch: int) -> None:
        """
        Run training
        
        Args:
            batch (torch.Tensor): Batch of data
            
        """  
        
        self.model.train()
            
        batch_iterator: torch.Tensor = (
            tqdm(train_ds, 
                 desc=f"Processing epoch {epoch:02d}")
        )
       
        for batch in batch_iterator:
            # New line after batch iterator
            print("")
            
            # Model prediction 
            output = self.model.forward(batch['decoder_input'].to(self.device))
            
            # Compute the loss using a simple cross entrophy
            
            output: torch.Tensor = output.to(self.config.dtype)
            
            labels: torch.Tensor = batch['labels'].to(self.device) 
            
            loss: torch.Tensor = self.loss_fn(output, labels)
            
            prediction = torch.argmax(output, dim=-1)
            
            self.debug.render_loss(loss, epoch, 5)
            self.debug.render_accuracy(prediction, labels, epoch, 5)
   
            # Backpropagate the loss
            loss.backward()
            
            # Update the weights
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
  
    def _run_validation(self, 
                        validation_ds: torch.Tensor,
                        epoch: int) -> None:
        
        self.model.eval()
        
        total = 0
        correct = 0
        predicted_img = []
        labels = []
        
        batch_iterator: torch.Tensor = (
            tqdm(validation_ds, 
                 desc=f"Validation epoch {epoch:02d}")
        )

        with torch.no_grad():
            for batch in batch_iterator:
                
                # New line after batch iterator
                print("")

                input = batch["decoder_input"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                predicted_img = self.model.forward(input)
                
                self.debug.render_accuracy(predicted_img, labels, epoch, 5, "validation")
                
                correct += (predicted_img == labels).sum().item()  
                total += labels.size(0)
                
        
        accuracy = (correct / total) * 100
        
        return accuracy
        
        
    def _get_weights_file_path(self) -> str:
        """
        Get the path to the model weights file
        
        Args:
            epoch (Optional[str]): Epoch number
        
        Returns:
            str: Path to the model weights file
        """
        weights_path = f"{self.config.model_name}.pt"
        return os.path.join(self.config.model_path, weights_path)
    
    
    def _save_model_state(self) -> None:
        """
        Save model state
        """
        
        
        
        torch.save({ 
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.model_filename)
        
    


    def _load_model_state(self) -> None:
        """
        Load model state
        """
    
        if self.config.preload == False:
            return
            
            
        
        if self.model_filename.is_file():
            print(f'Preloading model {self.model_filename.name}')
            state = torch.load(self.model_filename)
            
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state['model_state_dict'], 
                strict=self.config.strict_load
            )
            
            if missing_keys == False and unexpected_keys == False:
                # Can not use if exists new parameters in model
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
        else:
            print('No model to preload')       

    def _load_json(self, file_path: Path) -> List[str]:
        """
        Load json file
        
        Args:
            file_path (Path): Path to the json file
        
        Returns:
            List[str]: Data from json file
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    
    def _load_pkl(self, file_path: Path) -> List[str]:
        """
        Load pkl file
        
        Args:
            file_path (Path): Path to the pkl file
        
        Returns:
            List[str]: Data from pkl file
        """
        
        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="bytes")
            # Convert bytes to string (value is float)
            data = {key.decode("utf-8"): value for key, value in data.items()}
            
        data["data"] = data["data"].reshape(-1, *self.config.image_shape)
        
        print(f"Data shape: {data['data'].shape}")
        return data