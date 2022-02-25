import argparse
import time

import numpy as np
import torch
from distributed import Client
from torch.utils.data.dataset import Dataset
import transformers
import dask.dataframe as dd
import gc


class RecSys2020Dataset(Dataset):
    def __init__(self, args):
        # Parameters for the model
        super(RecSys2020Dataset, self).__init__()
        self.representative_size = args.representative_size
        self.target_tokenizer = args.target_tokenizer
        self.tokenizer = None
        self.root = args.root_folder
        self.file = args.data_file

        print("loading the data...")
        # Here, we select the subset of columns we want in our dataset
        # All the columns in the dataset
        columns = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                   "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id",
                   "engaged_with_user_follower_count",
                   "engaged_with_user_following_count", "engaged_with_user_is_verified",
                   "engaged_with_user_account_creation",
                   "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count",
                   "engaging_user_is_verified",
                   "engaging_user_account_creation", "engagee_follows_engager",
                   "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]
        # The columns we want
        columns_we_want = ["text_tokens", "engaging_user_id", "reply_timestamp",
                           "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]  #
        # casting columns to types
        dtypes = {
            'text_tokens': 'object',  # String
            'reply_timestamp': 'float',  # unix timestamp
            'retweet_timestamp': 'float',  # unix timestamp
            'retweet_with_comment_timestamp': 'float',  # unix timestamp
            'like_timestamp': 'float',  # unix timestamp
            'engaging_user_id': 'object'  # String
        }

        # Load the csv using dask, partitioning the dataset into chunks
        # This does not load the entire dataset into memory, rather partition the data that can be selectively loaded into memory
        self.data = dd.read_csv(self.root + self.file, delimiter="\x01", encoding='utf-8',
                                header=0,
                                names=columns,
                                usecols=columns_we_want,
                                dtype=dtypes).set_index('engaging_user_id')  # faster indexing

        self.ref = None

    def convert_tokens(self):
        # First step: We create interaction_time by taking the smallest time out of reply, retweet, retweet_w_comment
        # and like.
        #
        # this is done in parallel and by loading paritions in and out of memory
        self.data['interaction_time'] = self.data[
            ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]].min(axis=1)

        # We convert all our labels into boolean values, saves memory and can be used in inference
        # engaged_with is created to label tweets that should be part of history or not
        self.data[['engaged_with', 'reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp',
                   'like_timestamp']] = \
            self.data[['interaction_time', 'reply_timestamp', 'retweet_timestamp',
                       'retweet_with_comment_timestamp', 'like_timestamp']].notnull()

        # Since sort_values cannot be done in parallel, keeping the dask dataframe in dask form is no different to
        # converting it to a pandas dataframe. Keeping in dask form however forces all partitions to move to a single
        # worker, causing it to out of memory

        # currently looking at ways to potentially avoid doing this
        self.data = self.data.compute()  # convert to pandas

        # Sort values by interaction time. Grouping and then sorting takes much longer and surprisingly more memory
        self.data = self.data.sort_values(by='interaction_time')  # na's are put last
        self.data['interaction_time'] = self.data['interaction_time'].fillna(-1)
        self.data['interaction_time'] = ((np.floor(self.data['interaction_time'] / 86400) + 4) % 7).astype('uint8')

        # we create a dataframe to hold the history of each user, filter data by ones engaged with
        self.ref = self.data.loc[self.data['engaged_with']]
        print('grouping')
        del self.ref['engaged_with']

        # Group this dataframe by user (the index). We don't care if the users are sorted in the dataframe
        self.ref = self.ref.groupby(self.ref.index, sort=False, group_keys=False)  # engaging_user_id
        print('grouped')

        # Squeeze each group into lists. The list is sorted according to our earlier sort.
        # This constitutes a users timeline
        self.ref = self.ref.aggregate(
            {
                'text_tokens': list,
                'interaction_time': list,
                'reply_timestamp': list,
                'retweet_timestamp': list,
                'retweet_with_comment_timestamp': list,
                'like_timestamp': list
            }
        )
        print('ref done')

        print('data creation started')

        def index(x):
            reps = x.loc[x]  # get tweets that were engaged with
            # (engaged_with is a series of booleans, can be used to mask)

            # This creates a list in the form of [0, 1, 2, 3, ... x, -1, -1, -1, ..., y]
            # where x is the amount of engaged with tweets and y is the total length of tweets
            return np.concatenate((np.arange(len(reps)), np.full(len(x) - len(reps), -1)))

        print('applying')

        # here, we label each tweet in a group by where they appear on the timeline. This is used to sample the history
        # correctly. The index function above generates the history_index column for each group

        # functions by taking the engaged_with series, grouping it by its index (users) and then transforming it to
        # create history_index with the use of the index function
        self.data['history_index'] = self.data['engaged_with'].groupby(self.data.index, sort=False).transform(index)
        del self.data['engaged_with']

        print('data done')
        print("WE ARE DONE")

    def __len__(self):
        # necessary dataset function
        return len(self.data)

    def add_tokenizers(self, target='bert-base-multilingual-cased'):
        # since tokenizers contribute to saved dataset size, we hold off adding them until dataset is loaded
        # step takes no time, but have a massive impact on size
        self.target_tokenizer = transformers.AutoTokenizer.from_pretrained(target)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    def remove_tokenizers(self):
        # remove tokenizers, useful before saving
        self.target_tokenizer = None
        self.tokenizer = None

    def __getitem__(self, idx):
        x = self.data.iloc[idx]  # get example
        # important to note, dask does not support iloc for index based gathering, only label based

        def decode_tokens(x):
            x = np.array(x.split('\t')).astype(int)
            return x

        if x.name in self.ref:  # does the tweet have history?
            history_info = self.ref.loc[x.name]  # retrieve the users history
            # sample the users history according to get_historical_info
            history, history_times, history_engagement, history_mask = self.get_historical_info(history_info,
                                                                                                x['history_index'],
                                                                                                self.representative_size)

            # convert tokens from a string to a list of ints
            history = np.array([decode_tokens(hist) for hist in history])  # convert tweets to arrays of ints, rather then string
        else:
            # if the tweet has no history, use default values that will be fully masked
            history = np.full((self.representative_size, 1), np.expand_dims(np.array(101), 0))
            history_times = np.full(self.representative_size, 0)
            history_engagement = np.full((self.representative_size, 4), np.full(4, 0))
            history_mask = np.concatenate(([False], np.full(self.representative_size, True)))

        # convert array of ints to decoded strings and clean them
        history = np.array([clean_tweet(tweet) for tweet in self.tokenizer.batch_decode(history)])
        text_tokens = clean_tweet(self.tokenizer.decode(decode_tokens(x['text_tokens'])))

        # prefix each history tweet by its engagement type
        def prefix(engagements, text) -> str:
            prepend = ""
            if engagements[0]:  # reply
                prepend += 'reply, '
            if engagements[1]:  # retweet
                prepend += 'retweet, '
            if engagements[2]:  # retweet with comment
                prepend += 'retweet with comment, '
            if engagements[3]:  # like
                prepend += 'like, '

            prepend = prepend[:-2] + ':' + text  # remove ', '
            return prepend

        # tokenize history and target according to target tokenizer
        # [CLS] <engagement> : <history_tweet> [SEP] <target_tweet>
        tokenized = self.target_tokenizer([prefix(engage, text) for engage, text in zip(history_engagement, history)],
                                          [text_tokens for _ in range(len(history))],
                                          return_tensors='pt', padding='max_length', truncation=True)
        del history_engagement, history
        # [CLS] <target_tweet>
        target = self.target_tokenizer(text_tokens, return_tensors='pt', padding='max_length', truncation=True)
        del text_tokens
        # create torch tensors and return, model input
        return torch.cat((target['input_ids'], tokenized['input_ids']), 0), \
               torch.cat((target['attention_mask'], tokenized['attention_mask']), 0), \
               torch.cat((target['token_type_ids'], tokenized['token_type_ids']), 0), \
               torch.Tensor(history_times), \
               torch.Tensor(history_mask).bool(), \
               torch.Tensor([x['reply_timestamp'],
                             x['retweet_timestamp'],
                             x['retweet_with_comment_timestamp'],
                             x['like_timestamp']])

    def get_historical_info(self, x, index, representative_size):
        text_tokens = x['text_tokens']  # get the users tweet history
        times = x['interaction_time']  # all the times they interacted with the tweet
        engagements = np.stack((x['like_timestamp'],
                                x['reply_timestamp'],
                                x['retweet_timestamp'],
                                x['retweet_with_comment_timestamp'])).T  # combine their engagements together

        # using gathered indexes from get_interaction, quickly sample the users history using numpy masking
        def quick_sample(stuff, indexes, missing, num_missing):
            return np.concatenate((stuff[indexes], np.full((num_missing, len(missing)), missing)))

        if index != -1:
            indexes, mask, num_missing = self.get_iteration(len(text_tokens), representative_size, index)
        else:
            indexes, mask, num_missing = self.get_iteration(len(text_tokens), representative_size, len(text_tokens) - 1)

        history = quick_sample(text_tokens, indexes, np.array(101), num_missing)
        history_times = quick_sample(times, indexes, np.array(-1), num_missing)
        engage = quick_sample(engagements, indexes, np.fill(4, 0), num_missing)

        mask = np.concatenate(([False], mask))  # to ensure that the target tweet is always engaged with
        return history, history_times, engage, mask

    # Creates the mask for sampling history
    def get_iteration(self, length, rep_size, i):
        num_missing = 0
        stuff = np.arange(length)
        size_left = rep_size // 2
        size_right = rep_size - size_left

        iteration = []
        mask = []
        if i < size_left:
            iteration += stuff[0:i] + stuff[i + 1: rep_size + 1]
            mask += [False for _ in range(len(iteration))]
            for k in range(rep_size - len(iteration)):
                if rep_size + 1 + k < len(stuff):
                    mask += [False]
                    iteration += [stuff[rep_size + 1 + k]]
                else:
                    num_missing += 1
                    # mask += [True]
                    # iteration += [missing]
        elif i < len(stuff) - size_right:
            iteration += stuff[i - size_left: i] + stuff[i + 1: i - size_left + rep_size + 1]
            mask += [True for _ in range(len(iteration))]
            for k in range(rep_size - len(iteration)):
                if i - size_left - k >= 0 and i - size_left + rep_size + 1 + k < len(stuff):
                    mask += [False]
                    if k % 2 == 0:
                        iteration += [stuff[i - size_left - k]]
                    else:
                        iteration += [stuff[i - size_left + rep_size + 1 + k]]
                elif i - size_left - k >= 0:
                    mask += [False]
                    iteration += [stuff[i - size_left - k]]
                elif i - size_left + rep_size + 1 + k < len(stuff):
                    mask += [False]
                    iteration += [stuff[i - size_left + rep_size + 1 + k]]
                else:
                    num_missing += 1
                    # mask += [True]
                    # iteration += [missing]
        else:
            iteration += stuff[len(stuff) - size_right - size_left - 1: i] + stuff[i + 1: len(stuff)]
            mask += [False for _ in range(len(iteration))]
            curr = len(iteration)
            for k in range(rep_size - curr):
                if rep_size - curr - 3 - k >= 0:
                    mask += [False]

                    iteration += [stuff[rep_size - curr - 3 - k]]
                else:
                    num_missing += 1
                    # mask += [True]
                    # iteration += [missing]
        return np.array(iteration), np.array(mask), num_missing


def clean_tweet(x):
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
            username = []
            while i < len(tokens) - 1 and tokens[i] != ':':  # grab the username
                username += [tokens[i]]
                i += 1
            result += [''.join(username)]
            i += 1  # skip the ':'
        elif target == '@':
            i += 1
            username = [tokens[i]]
            i += 1
            while i < len(tokens) - 2 and tokens[i] == '_':
                username += [tokens[i], tokens[i + 1]]  # add the username
                i += 2
            result += [''.join(username)]

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


if __name__ == '__main__':
    # print(dask.__version__)
    # print(pd.__version__)
    parser = argparse.ArgumentParser()

    parser.add_argument("--dask_num_workers", type=int, default=2)
    parser.add_argument("--dask_threads_per_worker", type=int, default=6)
    parser.add_argument("--dask_blocksize", default='10MB')
    parser.add_argument("--root_folder", default='E:\\liamx\\Documents\\recsys2020\\')
    parser.add_argument("--data_file", default='val.tsv')
    parser.add_argument("--result_file", default='final_recsys_dataset.pt')
    parser.add_argument("--representative_size", type=int, default=4)
    parser.add_argument("--target_tokenizer", default='bert-base-multilingual-cased')
    args = parser.parse_args()
    print(args)
    print(transformers.__version__)
    # #
    client = Client(n_workers=args.dask_num_workers, threads_per_worker=args.dask_threads_per_worker)
    print(client.dashboard_link)
    print(client)
    data = RecSys2020Dataset(args)
    curr = time.time()
    data.convert_tokens()
    print("TIME", time.time() - curr)
    # print('loading')
    # data = torch.load(args.result_file)
    # print('loaded')

    print('adding tokenizers')
    data.add_tokenizers()

    print('added tokenizers')
    # client.close()
    try:
        for i in range(1000, 1100):
            print(data[i])

        print(data)
    except Exception as e:
        print('saving')
        gc.collect()
        data.remove_tokenizers()
        torch.save(data, args.result_file)
        print('saved')
        raise e

    print('saving')
    data.remove_tokenizers()
    gc.collect()
    torch.save(data, args.result_file)
    print('saved')
