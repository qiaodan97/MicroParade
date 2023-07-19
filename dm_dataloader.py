# ===================
# Contains the dataset chunking and serving code for RecSys dataset
# Supporting language model for both bert and mini-LM,
# but language indicator needs to be changed when switching language model (line 74~95)
# Author: @liamhebert, @qiaodan97
# ===================


import gc
import math
import torch
from torch.utils.data import IterableDataset, Dataset
import pandas as pd
import numpy as np
import transformers


class RecSysMasterDataset(IterableDataset):
    """
    Master class for the dataset.
    Since the RecSys dataset is so large, we instead load parts of the dataset into memory
    Those chunks are the RecSysDataInstance classes, which are controlled by this class
    """

    def __init__(self, root_folder, data_folder):
        self.root_folder = root_folder
        self.data_folder = data_folder
        super(RecSysMasterDataset, self).__init__()
        # Read csv containing location of each data chunk and its size
        self.datasets = pd.read_csv(self.root_folder + self.data_folder + 'recsys2021-size.csv', names=['file', 'size'],
                                    index_col='file', header=0).squeeze()

        # used for tpu usage, can be ignored
        self.world_size = 1
        self.rank = 0

    def add_tpu_data(self, world_size, rank):
        """
        Used when using multiple gpus/tpus, allows sampler to distribute data
        """
        self.world_size = world_size
        self.rank = rank

    def __len__(self):
        return self.datasets.sum()

    def __iter__(self):
        return self.dataset_builder()

    def dataset_builder(self):
        """
        Iteratively grab the next chunk in the index and start processing it
        """
        curr = None
        sampler = None
        for dataset in self.datasets.index:
            del curr, sampler
            gc.collect()
            # load data into memory
            # curr = RecSysDatasetInstance(self.root_folder + self.data_folder + dataset)
            curr = RecSysDatasetInstance(self.root_folder + self.data_folder + dataset)
            # distributed sampler for sampling across multiple gpus/tpus
            sampler = torch.utils.data.distributed.DistributedSampler(
                curr,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True
            )

            for x in sampler:
                yield curr[x]

        # return RecSysDatasetInstance(next(self.datasets))

# TODO: change to be recognized from the arguments in main program
# Lazy approach but this is used to allow for easy access between different language models
# target_tokenizer - language model we want to use
# target_tokenizer_bert = transformers.AutoTokenizer.from_pretrained(
#     '/home/qiaodan/projects/def-emilios/qiaodan/MicroParade/code/bert-base-multilingual-cased/')
# tokenizer - language model we used to encode the dataset initially
# tokenizer_bert = transformers.AutoTokenizer.from_pretrained(
#     '/home/qiaodan/projects/def-emilios/qiaodan/MicroParade/code/bert-base-multilingual-cased/')

# target_tokenizer_minilm = transformers.AutoTokenizer.from_pretrained(
#     '/home/qiaodan/projects/def-emilios/qiaodan/MicroParade/code/Multilingual-MiniLM-L12-H384/')
# tokenizer - language model we used to encode the dataset initially
# tokenizer_minilm = transformers.AutoTokenizer.from_pretrained(
#     '/home/qiaodan/projects/def-emilios/qiaodan/MicroParade/code/Multilingual-MiniLM-L12-H384/')

# Used to control the language model.
# target_tokenizer = target_tokenizer_minilm
# tokenizer = tokenizer_bert
lang_model = "minilm"

class RecSysDatasetInstance(Dataset):
    """
    Processes and serves a chunk of the data held in memory
    """

    def __init__(self, parquet):
        # Utility functions to decode parquet constraints
        def parse_array(x):
            return np.array(x.split('\x01'))

        def decode_tokens(x):
            x = np.array([np.array([int(y) for y in z.split('\t')]) for z in x], dtype='object')
            return x

        self.data = pd.read_parquet(parquet)
        # label of previous interactions
        cols = ['reply_timestamp_history',
                'retweet_timestamp_history',
                'retweet_with_comment_timestamp_history',
                'like_timestamp_history',
                'mask_history']

        self.data[cols] = self.data[cols].applymap(lambda x: x == 'True')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.data.iloc[item]
        # first 12 features are from text preprocessor
        nv_features = x[12:]
        # gc.collect()

        # create torch tensors and return, model input
        # for minilm

        if lang_model == "minilm":
            return torch.Tensor(nv_features), \
                   torch.Tensor([int(x['like_timestamp_data'])])

def clean_tweet(x):
    """
    Cleans a tweet by removing unnecessary material
    """
    # things we want to do:
    #   - remove links
    #   - remove @users
    #   - remove RT :
    #   - remove [CLS] (we add it anyway)
    tokens = x.split(' ')
    result = []
    i = 1
    while i < len(tokens) - 1:  # ignore first and last token
        target = tokens[i]
        if target == 'https':  # replace links with [LINK]
            # url is a
            i += 8
            result += ['[LINK]']
        elif target == 'RT':
            # format is RT @ <user> :
            i += 2
            while i < len(tokens) - 1 and tokens[i] != ':':  # grab the username
                i += 1
            i += 1  # skip the ':'
        elif target == '@':
            i += 2
            while i < len(tokens) - 2 and tokens[i] == '_':
                i += 2
        elif target == '¶':
            i += 1
        elif target == '[UNK]':
            i += 1
        elif target == '[CLS]':
            i += 5
        else:
            result += [tokens[i]]
            i += 1
    return ' '.join(result)