import json
from typing import List, Tuple
from pathlib import Path
import pickle


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from pathlib import Path

from model_i2t.main import VisionTransformer
from model_i2t.train.config import TrainArgs
from model_i2t.train.datasets import Train_Dataset




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

            if val_accuracy > best_val_accuracy:
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
            self.augment_dataset(ds_raw)
        
        # Size of the training/validation dataset
        train_ds_size = int(self.config.train_ds_size * len(ds_raw))
        val_ds_size = int(len(ds_raw) - train_ds_size)

        dataset = dataset(ds_raw)
        
         
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
        
        file_path = Path(self.config.dataset_path)

        assert file_path.is_file(), (
            f"File not found in directory {str(file_path)}"
        )
        

        ## Loading based on file extension
        
        match file_path.suffix:
            
            case ".jsonl":
                return Train._load_json(file_path)
            case ".json":
                return Train._load_json(file_path)
            case ".pkl":
                return Train._load_pkl(file_path)
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
            output = self.model.forward(batch['decoder_input'].to(self.device))

            # Compare the output with the label
            label: torch.Tensor = batch['labels'].to(self.device) 

            # Compute the loss using a simple cross entrophy
            loss: torch.Tensor = self.loss_fn(
                output.view(-1, self.config.num_classes),
                label.view(-1)
            )

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
        target_img = []
        
        batch_iterator: torch.Tensor = (
            tqdm(validation_ds, 
                 desc=f"Validation epoch {epoch:02d}")
        )

        with torch.no_grad():
            for batch in batch_iterator:

                output_img = self.model.forward(
                    batch["decoder_input"].to(self.device)
                )

                predicted_img.append(torch.argmax(output_img, dim=-1))
                target_img.append(batch["labels"])
                
                
                if (predicted_img[-1] == target_img[-1]):
                    correct += 1
                
                total += 1
        
        accuracy = (correct / total) * 100
        
        print(f"{f'TARGET: ':>12}{target_img}")
        print(f"{f'PREDICTED: ':>12}{output_img}")
        print(f"ACCURACY: {accuracy} %")
        
        return accuracy
        
        
    def _get_weights_file_path(self) -> str:
        """
        Get the path to the model weights file
        
        Args:
            epoch (Optional[str]): Epoch number
        
        Returns:
            str: Path to the model weights file
        """
        model_filename = f"{self.config.model_name}.pt"
        return str(Path('.') / self.config.model_path / model_filename)
    
    
    def _save_model_state(self) -> None:
        """
        Save model state
        """
        
        model_filename = self._get_weights_file_path()
        
        torch.save({ 
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_filename)
        
    


    def _load_model_state(self) -> None:
        """
        Load model state
        """
    
        if self.config.preload:
            model_filename = Path(self._get_weights_file_path())
        else:
            return
            
        
        if model_filename.is_file():
            print(f'Preloading model {model_filename.name}')
            state = torch.load(model_filename)
            self.model.load_state_dict(state['model_state_dict'])
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
         else:
            print('No model to preload')       

    @staticmethod
    def _load_json(file_path: Path) -> List[str]:
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
    
    @staticmethod
    def _load_pkl(file_path: Path) -> List[str]:
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
        return data