# ===================
# DataModule to create dataloaders for each split
# Author: @liamhebert
# ===================


import pytorch_lightning as pl
from torch.utils.data import DataLoader
from recsys_dataset import RecSysMasterDataset
import torch.utils.data as data
import math 

class RecsysDataset(pl.LightningDataModule):
    def __init__(self, num_workers, 
                       train_batch_size, 
                       test_batch_size, 
                       root_folder, 
                       data_folder, 
                       split_percent):
        self.save_hyperparameters()
        self.train_dataset, self.test_dataset = generate_split(percent=self.split_percent, root_folder = self.root_folder, data_folder = self.data_folder)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.test_batch_size, 
                          shuffle=True, 
                          drop_last=True, 
                          num_workers=self.num_workers, 
                          worker_init_fn=worker_init_fn)
    
    def valid_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.test_batch_size, 
                          shuffle=True, 
                          drop_last=True, 
                          num_workers=self.num_workers, 
                          worker_init_fn=worker_init_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.test_batch_size, 
                          shuffle=True, 
                          drop_last=True, 
                          num_workers=self.num_workers, 
                          worker_init_fn=worker_init_fn) 

def generate_split(percent, root_folder, data_folder):
    """
    Generate train-test splits 
    """
    train = RecSysMasterDataset(root_folder, data_folder)
    train.datasets = train.datasets[:int(len(train.datasets) * percent)]
    test = RecSysMasterDataset(root_folder, data_folder)
    test.datasets = test.datasets[int(len(test.datasets) * percent):]
    return train, test


def worker_init_fn(worker_id):
    """
    Since we organize the dataset into chunks, we assign chunks to dataloader workers.
    """
    worker_info = data.get_worker_info()
    dataset = worker_info.dataset
    num_workers = worker_info.num_workers
    worker_id = worker_info.id
    size = len(dataset.datasets)
    per_worker = int(math.ceil(size / num_workers))
    start = worker_id * per_worker
    end = min(start + per_worker, size)

    dataset.datasets = dataset.datasets[start:end]
