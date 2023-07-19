# ===================
# DataModule to create dataloaders for each split
# Author: @liamhebert
# ===================


import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dm_dataloader import RecSysMasterDataset
import torch.utils.data as data
import math


class RecsysDataset(pl.LightningDataModule):
    # passing the nessesary arguments from MicroParade main program
    def __init__(self, num_workers, train_batch_size, test_batch_size, valid_batch_size, root_folder, data_folder,
                 split_percent_train, split_percent_test):
        super().__init__()
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_batch_size = valid_batch_size
        self.root_folder = root_folder
        self.data_folder = data_folder
        self.split_percent_train = split_percent_train
        self.split_percent_test = split_percent_test
        self.save_hyperparameters()
        self.train_dataset, self.test_dataset, self.valid_dataset = generate_split(
            percent_train=self.split_percent_train,
            percent_test=self.split_percent_test,
            root_folder=self.root_folder,
            data_folder=self.data_folder)
    # Override the dataloaders of pytorch lightening data module.
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          # shuffle=True,
                          drop_last=True,
                          num_workers=self.num_workers)
                          # worker_init_fn=worker_init_fn)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.valid_batch_size,
                          # shuffle=True,
                          drop_last=True,
                          num_workers=self.num_workers)
                          # worker_init_fn=worker_init_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.test_batch_size,
                          # shuffle=True,
                          drop_last=True,
                          num_workers=self.num_workers)
                          # worker_init_fn=worker_init_fn)

# helper functions
def generate_split(percent_train, percent_test, root_folder, data_folder):
    """
    Generate train-test splits
    default:
        train: [:80%]
        test: [80%:90%]
        valid: [90%:]
    """
    train = RecSysMasterDataset(root_folder, data_folder)
    # This length is the same for train, test and valid, which equals to the num of parquets.
    data_length = len(train.datasets)
    train.datasets = train.datasets[:int(data_length * percent_train)]
    test = RecSysMasterDataset(root_folder, data_folder)
    test.datasets = test.datasets[int(data_length * percent_train): \
                                  int(data_length * percent_train) + int(data_length * percent_test)]
    valid = RecSysMasterDataset(root_folder, data_folder)
    valid.datasets = valid.datasets[int(data_length * percent_train) + int(data_length * percent_test):]

    return train, test, valid


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
