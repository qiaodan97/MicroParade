# ===================
# Contains the dataset chunking and serving code for RecSys dataset
# Author: @liamhebert
# ===================


import argparse
import gc
import math
import torch
from torch.utils.data import IterableDataset, Dataset
import pandas as pd
import numpy as np
import transformers
from sklearn.preprocessing import MinMaxScaler

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
                                    index_col='file', header=0, squeeze=True)

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

        # return RecSysDatasetInstance(next(self.datasets), self.ref, self.args)

# Lazy approach but this is used to allow for easy access between different language models
# target_tokenizer - language model we want to use 
target_tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
# tokenizer - language model we used to encode the dataset initially
tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

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
        # decode some data compresssion 
        self.data[['text_tokens_history',
                   'reply_timestamp_history',
                   'retweet_timestamp_history',
                   'interaction_time_history',
                   'mask_history',
                   'retweet_with_comment_timestamp_history',
                   'like_timestamp_history']] = self.data[['text_tokens_history',
                                                           'reply_timestamp_history',
                                                           'retweet_timestamp_history',
                                                           'interaction_time_history',
                                                           'mask_history',
                                                           'retweet_with_comment_timestamp_history',
                                                           'like_timestamp_history']].applymap(parse_array)
        # label of previous interactions
        cols = ['reply_timestamp_history',
                'retweet_timestamp_history',
                'retweet_with_comment_timestamp_history',
                'like_timestamp_history',
                'mask_history']

        self.data[cols] = self.data[cols].applymap(lambda x: x == 'True')

        self.data['interaction_time_history'] = self.data['interaction_time_history'].apply(lambda x: x.astype(int))
        self.data['text_tokens_history'] = self.data['text_tokens_history'].apply(decode_tokens)
        self.data['text_tokens_data'] = self.data['text_tokens_data'].apply(
            lambda x: np.array([int(y) for y in x.split('\t')]))
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.data.iloc[item]
        # first 12 features are from nv_tabular
        nv_features = x[12:]
        x = x[:12]

        # history information
        history_engagement = np.stack((x['like_timestamp_history'],
                                       x['reply_timestamp_history'],
                                       x['retweet_timestamp_history'],
                                       x['retweet_with_comment_timestamp_history'])).T

        # history_mask = parse_array(x['mask_history']) == 'True'

        # convert array of ints to decoded strings and clean them
        # decode tweets using tokenizer in order to convert to new format
        history = [clean_tweet(tweet) for tweet in tokenizer.batch_decode(x['text_tokens_history'])]
        text_tokens = clean_tweet(tokenizer.decode(x['text_tokens_data']))

        # tokenize history and target according to target tokenizer
        # [CLS] <engagement> : <history_tweet> [SEP] <target_tweet>
        tokenized = target_tokenizer(
            history, # TODO: I think I removed the history engagement piece here by accident, should be replaced
            [text_tokens for _ in range(len(history))],
            return_tensors='pt', padding='max_length', truncation=True)
        del history_engagement, history
        # [CLS] <engagement>
        target = target_tokenizer(text_tokens, return_tensors='pt', padding='max_length', truncation=True)
        del text_tokens
        gc.collect()
        # create torch tensors and return, model input
        return torch.cat((target['input_ids'], tokenized['input_ids']), 0), \
               torch.cat((target['attention_mask'], tokenized['attention_mask']), 0), \
               torch.cat((target['token_type_ids'], tokenized['token_type_ids']), 0), \
               torch.Tensor(x['interaction_time_history']), \
               torch.Tensor(x['mask_history']).bool(), \
               torch.Tensor(nv_features), \
               torch.Tensor([x['reply_timestamp_data'],
                             x['retweet_timestamp_data'],
                             x['retweet_with_comment_timestamp_data'],
                             x['like_timestamp_data']])


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
                # username += [tokens[i]]
                i += 1
            # result += [''.join(username)]
            i += 1  # skip the ':'
        elif target == '@':
            i += 2
            while i < len(tokens) - 2 and tokens[i] == '_':
                # username += [tokens[i], tokens[i + 1]]  # add the username
                i += 2
        elif target == '¶':
            i += 1
        elif target == '[UNK]':
            i += 1
        elif target == '[CLS]':
            i += 5

        # elif target == '[UNK]':
        #     i += 1  # skip [unk]
        #     flag = False
        #     while tokens[i] == '[UNK]':
        #         i += 1
        #         flag = True
        #     if not flag:
        #         result[i - 1]
        else:
            result += [tokens[i]]
            i += 1
    return ' '.join(result)


def generate_split(args, percent):
    """
    Generate train-test splits 
    """
    train = RecSysMasterDataset(args)
    train.datasets = train.datasets[:int(len(train.datasets) * percent)]
    test = RecSysMasterDataset(args)
    test.datasets = test.datasets[int(len(test.datasets) * percent):]
    return train, test


def worker_init_fn(worker_id):
    """
    Since we organize the dataset into chunks, we assign chunks to dataloader workers.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    num_workers = worker_info.num_workers
    worker_id = worker_info.id
    size = len(dataset.datasets)
    per_worker = int(math.ceil(size / num_workers))
    start = worker_id * per_worker
    end = min(start + per_worker, size)

    dataset.datasets = dataset.datasets[start:end]

