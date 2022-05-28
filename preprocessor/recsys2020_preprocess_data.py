import argparse
import time

import dask.dataframe as dd
import numpy as np
from dask.distributed import Client
import pandas as pd


def main(args):
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
                       "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]
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
    print(args.root_folder + args.data_file)
    data = dd.read_csv(args.root_folder + args.data_file, delimiter="\x01", encoding='utf-8',
                       header=0,
                       names=columns,
                       dtype=dtypes,
                       usecols=columns_we_want).set_index('engaging_user_id')

    data.index = data.index.astype(str)

    # drop = [col for col in data.columns if col not in columns_we_want]
    # for col in drop:
    #     data = data.drop(col, axis=1)

    data['interaction_time'] = data[
        ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]].min(axis=1)

    # We convert all our labels into boolean values, saves memory and can be used in inference
    # engaged_with is created to label tweets that should be part of history or not
    data[['engaged_with', 'reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp',
          'like_timestamp']] = \
        data[['interaction_time', 'reply_timestamp', 'retweet_timestamp',
              'retweet_with_comment_timestamp', 'like_timestamp']].notnull()

    dtypes = {
        'text_tokens': 'object',  # String
        'reply_timestamp': bool,  # unix timestamp
        'retweet_timestamp': bool,  # unix timestamp
        'retweet_with_comment_timestamp': bool,  # unix timestamp
        'like_timestamp': bool,  # unix timestamp
        # 'engaging_user_id': 'object',  # String
        'interaction_time': 'uint8',  # unix timestamp
        'engaged_with': bool,
        'history_index': 'uint8'
    }

    def apply(x):
        x = x.sort_values(by='interaction_time')
        x['history_index'] = np.arange(len(x))

        return x

    data['interaction_time'] = data['interaction_time'].fillna(-1)
    data['interaction_time'] = (np.floor(data['interaction_time'] / 3600) % 24).astype('uint8')

    data = data.groupby('engaging_user_id', group_keys=False, sort=False).apply(apply, meta=dtypes)

    num_partitions = int(data.npartitions / 4)
    data = data.persist()
    ref = data.loc[data['engaged_with']]

    del ref['engaged_with']

    # Group this dataframe by user (the index). We don't care if the users are sorted in the dataframe
    ref = ref.groupby(ref.index, sort=False)  # engaging_user_id

    ref = ref.aggregate(list)

    # extra_data = dd.read_parquet('E:\\liamx\\Documents\\recsys2021-val\\final.parquet')
    #
    # extra_data['interaction_time'] = extra_data[
    #     ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]].min(axis=1)
    # extra_data['interaction_time'] = extra_data['interaction_time'].fillna(-1)
    # extra_data['interaction_time'] = (np.floor(extra_data['interaction_time'] / 3600) % 24).astype('uint8')
    # extra_data = extra_data.groupby(extra_data.index, sort=False).aggregate(list)
    # print(extra_data.index)

    ref = ref.drop('history_index', axis=1)
    # cols = ref.columns
    # ref = ref.merge(extra_data, left_index=True, right_index=True, how='outer', suffixes=('', '_extra'))

    # print(cols)
    # for col in cols:
    #     ref[col] = ref[col] + ref[col + '_extra']
    #     ref = ref.drop(col + '_extra', axis=1)
    #     ref[col] = ref[col].apply(lambda x: x[:args.representative_size], meta=list)
    ref.index = ref.index.rename(0)
    data.index = data.index.rename(0)

    # TODO: Drop all rows that do not have a history (impossible to predict and can effect training)

    data = data.merge(ref, left_index=True, right_index=True, how='inner', suffixes=('_data', '_history'))

    data.index = data.index.rename('engaging_user_id')

    meta = {
        # 'engaging_user_id': str,
        1: str,
        2: str,
        3: str,
        4: str,
        5: str,
        6: str
    }

    # data[['interaction_time_history', 'mask_history']] \
    #     = data[
    #     ['interaction_time_history', 'history_index_data']
    # ].apply(get_historical_info_times, representative_size=args.representative_size, axis=1, meta=meta,
    #         result_type='expand')
    # rows = ['text_tokens_history', 'reply_timestamp_history', 'retweet_timestamp_history',
    #         'retweet_with_comment_timestamp_history', 'like_timestamp_history']
    # missing = ['101', False, False, False, False]
    # for row, missing in zip(rows, missing):
    #     data[row] = data[[row, 'history_index_data']].apply(get_historical_info,
    #                                                         representative_size=args.representative_size,
    #                                                         missing=missing, axis=1, meta=str)
    data[['text_tokens_history',
          'reply_timestamp_history',
          'retweet_timestamp_history',
          'retweet_with_comment_timestamp_history',
          'like_timestamp_history',
          'interaction_time_history',
          'mask_history']] = data[['history_index',
                                   'text_tokens_history',
                                   'reply_timestamp_history',
                                   'retweet_timestamp_history',
                                   'retweet_with_comment_timestamp_history',
                                   'like_timestamp_history',
                                   'interaction_time_history']] \
        .apply(get_historical_info, representative_size=args.representative_size, axis=1, meta=meta,
               result_type='expand')
    # dtypes = {
    #     'text_tokens': 'object',  # String
    #     'reply_timestamp': bool,  # unix timestamp
    #     'retweet_timestamp': bool,  # unix timestamp
    #     'retweet_with_comment_timestamp': bool,  # unix timestamp
    #     'like_timestamp': bool,  # unix timestamp
    #     'interaction_time': 'uint8',  # unix timestamp
    #     'engaged_with': bool,
    #     'history_index': 'uint8'
    # }
    del data['history_index'], data['interaction_time_data']

    data = data.persist()
    data.index = data.index.astype(str)
    data.to_parquet(args.root_folder + args.destination_folder)
    data = data.map_partitions(len).compute()

    data.index = [f"part.{idx}.parquet" for idx in data.index]
    data.to_csv(args.root_folder + 'recsys2021-size.csv')


def get_historical_info(x, representative_size):
    text_tokens = x['text_tokens_history']  # get the users tweet history
    if (type(text_tokens) == list and len(text_tokens) < 2) or (type(text_tokens) != list and pd.isna(text_tokens)):
        # no history
        text_tokens = np.full(representative_size, '101')
        times = np.full(representative_size, 24)
        like = np.full(representative_size, np.array([False]))
        reply = np.full(representative_size, np.array([False]))
        retweet = np.full(representative_size, np.array([False]))
        retweet_with_comment = np.full(representative_size, np.array([False]))
        mask = np.full(representative_size, [True])
    else:
        index = x['history_index_data']
        text_tokens = np.array(text_tokens)
        times = np.array(x['interaction_time_history'])  # all the times they interacted with the tweet
        like = np.array(x['like_timestamp_history'])
        reply = np.array(x['reply_timestamp_history'])
        retweet = np.array(x['retweet_timestamp_history'])
        retweet_with_comment = np.array(x['retweet_with_comment_timestamp_history'])

        # using gathered indexes from get_interaction, quickly sample the users history using numpy masking
        def quick_sample(stuff, indexes, missing, num_missing):

            # Exception: ValueError('all the input arrays must have same number of dimensions,
            # but the array at index 0 has 1 dimension(s) and the array at index 1 has 2 dimension(s)')
            # if missing is list:
            #     return np.concatenate((stuff[indexes], np.full((num_missing, len(missing)), missing)))
            return np.concatenate((stuff[indexes], np.full(num_missing, missing)))

        if index < len(text_tokens):
            indexes, mask, num_missing = get_iteration(len(text_tokens), representative_size, index)
        else:
            indexes, mask, num_missing = get_iteration(len(text_tokens), representative_size,
                                                       len(text_tokens) - 1)

        like = quick_sample(like, indexes, np.array(False), num_missing)
        reply = quick_sample(reply, indexes, np.array(False), num_missing)
        retweet = quick_sample(retweet, indexes, np.array(False), num_missing)
        retweet_with_comment = quick_sample(retweet_with_comment, indexes, np.array(False), num_missing)
        times = quick_sample(times, indexes, np.array(24), num_missing)
        text_tokens = quick_sample(text_tokens, indexes, '101', num_missing)

    mask = np.concatenate(([False], mask))  # to ensure that the target tweet is always engaged with
    times = np.concatenate(([24], times))  # so we don't add time to the example

    like = "\x01".join([str(x) for x in like])
    reply = "\x01".join([str(x) for x in reply])
    retweet = "\x01".join([str(x) for x in retweet])
    retweet_with_comment = "\x01".join([str(x) for x in retweet_with_comment])
    times = "\x01".join([str(x) for x in times])
    text_tokens = "\x01".join(text_tokens)
    mask = "\x01".join([str(x) for x in mask])

    return text_tokens, reply, retweet, retweet_with_comment, like, times, mask


# Creates the mask for sampling history
def get_iteration(length, rep_size, i):
    num_missing = 0
    stuff = np.arange(length).tolist()  # this can be done nicer
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
                mask += [True]
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
                mask += [True]
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
                mask += [True]
                # iteration += [missing]
    return np.array(iteration), np.array(mask), num_missing


if __name__ == '__main__':
    # print(dask.__version__)
    # print(pd.__version__)
    parser = argparse.ArgumentParser()

    parser.add_argument("--dask_num_workers", type=int, default=2)
    parser.add_argument("--dask_threads_per_worker", type=int, default=2)
    # parser.add_argument("--root_folder", default='/lustre03/project/6001735/qiaodan/MicroParade1/datasets/recsys2021/')
    parser.add_argument("--root_folder", default='../datasets/recsys2021/')
    parser.add_argument("--destination_folder", default='result')
    # parser.add_argument("--root_folder", default='gs://micro-parade-data/recsys2021/recsys2021/')
    # parser.add_argument("--data_file", default='part-00000.csv')
    parser.add_argument("--data_file", default='val.tsv')
    parser.add_argument("--representative_size", type=int, default=0)
    parser.add_argument("--target_tokenizer", default='bert-base-multilingual-cased1')
    args = parser.parse_args()
    print(args)
    # #
    client = Client(n_workers=args.dask_num_workers,
                    threads_per_worker=args.dask_threads_per_worker,
                    dashboard_address='0.0.0.0:8780')
    print(client.dashboard_link)
    print(client)
    start = time.time()
    main(args)
    print('LENGTH', time.time() - start)
