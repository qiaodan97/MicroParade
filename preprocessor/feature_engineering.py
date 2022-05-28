# External Dependencies
import gc
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import time

#
import argparse
# import cupy as cp  # CuPy is an implementation of NumPy-compatible multi-dimensional array on GPU
import cudf  # cuDF is an implementation of Pandas-like Dataframe on GPU
import numpy as np
# NVTabular is the core library, we will use here for feature engineering/preprocessing on GPU
import nvtabular as nvt
from nvtabular.ops import Categorify, DifferenceLag, FillMissing, JoinGroupby, Rename

# import dask  # dask is an open-source library to nateively scale Python on multiple workers/nodes
# import dask_cudf  # dask_cudf uses dask to scale cuDF dataframes on multiple workers/nodes
# More dask / dask_cluster related libraries to scale NVTabular


def splitmedia(col):
    if col.shape[0] == 0:
        return col
    else:
        return (
                col.str.split("\t", expand=True)[0].fillna("")
                + "_"
                + col.str.split("\t", expand=True)[1].fillna("")
        )


def count_token(col, token):
    not_null = col.isnull() == 0
    return ((col.str.count(token) + 1) * not_null).fillna(0)


def prep_dataset(INPUT_DATA_DIR, OUTPUT_DIR):
    features = [
        # Tweet Features
        "text_tokens",
        "hashtags",
        "tweet_id",
        "media",
        "links",
        "domains",
        "tweet_type",
        "language",
        "timestamp",
        # Engaged With User Features
        "a_user_id",
        "a_follower_count",
        "a_following_count",
        "a_is_verified",
        "a_account_creation",
        "b_user_id",
        # Engaging User Features
        "b_follower_count",
        "b_following_count",
        "b_is_verified",
        "b_account_creation",
        "b_follows_a",
        # Engagement Features
        "reply",  # Target Reply
        "retweet",  # Target Retweet
        "retweet_comment",  # Target Retweet with comment
        "like",  # Target Like
    ]

    count_features = (
            nvt.ColumnGroup(["hashtags", "domains", "links"])
            >> (lambda col: count_token(col, "\t"))
            >> Rename(postfix="_count_t")
    )

    split_media = nvt.ColumnSelector(["media"]) >> (lambda col: splitmedia(col))

    multihot_filled = ["hashtags", "domains", "links"] >> FillMissing()
    cat_columns = ["language", "tweet_type", "tweet_id", "a_user_id", "b_user_id"]
    cat_features = split_media + multihot_filled + cat_columns >> Categorify(out_path=OUTPUT_DIR)

    label_name = ["reply", "retweet", "retweet_comment", "like"]
    label_name_feature = label_name >> nvt.ops.FillMissing()

    weekday = (
            nvt.ColumnSelector(["timestamp"])
            >> (lambda col: cudf.to_datetime(col, unit="s").dt.weekday)
            >> nvt.ops.Rename(postfix="_wd")
    )

    output = count_features + cat_features + label_name_feature + weekday

    remaining_columns = [x for x in features if x not in (output.columns + ["text_tokens"])]

    proc = nvt.Workflow(output + remaining_columns)

    trains_itrs = nvt.Dataset(
        INPUT_DATA_DIR + "val.tsv",
        header=None,
        names=features,
        engine="csv",
        sep="\x01",
        part_size="1GB",
    )

    proc.fit(trains_itrs)

    dict_dtypes = {}
    for col in label_name + [
        "media",
        "language",
        "tweet_type",
        "tweet_id",
        "a_user_id",
        "b_user_id",
        "hashtags",
        "domains",
        "links",
        "timestamp",
        "a_follower_count",
        "a_following_count",
        "a_account_creation",
        "b_follower_count",
        "b_following_count",
        "b_account_creation",
    ]:
        dict_dtypes[col] = np.uint32

    proc.transform(trains_itrs).to_parquet(
        output_path=OUTPUT_DIR + "preprocess/", dtypes=dict_dtypes, out_files_per_proc=10
    )


def feature_engineering(OUTPUT_DIR, dataset_to_be_processed):
    count_encode = (
            ["media", "tweet_type", "language", "a_user_id", "b_user_id"]
            >> Rename(postfix="_c")
            >> JoinGroupby(cont_cols=["reply"], stats=["count"], out_path="./")
    )

    datetime = nvt.ColumnSelector(["timestamp"]) >> (
        lambda col: cudf.to_datetime(col.astype("int32"), unit="s")
    )

    hour = datetime >> (lambda col: col.dt.hour) >> Rename(postfix="_hour")
    minute = datetime >> (lambda col: col.dt.minute) >> Rename(postfix="_minute")
    seconds = datetime >> (lambda col: col.dt.second) >> Rename(postfix="_second")

    diff_lag = (
            nvt.ColumnSelector(["b_follower_count", "b_following_count", "language"])
            >> (lambda col: col.astype("float32"))
            >> DifferenceLag(partition_cols=["b_user_id"], shift=[1, -1])
            >> FillMissing(fill_val=0)
    )

    LABEL_COLUMNS = ["reply", "retweet", "retweet_comment", "like"]
    labels = nvt.ColumnSelector(LABEL_COLUMNS) >> (lambda col: (col > 0).astype("int8"))

    target_encode = [
                        "media",
                        "tweet_type",
                        "language",
                        "a_user_id",
                        "b_user_id",
                        ["domains", "language", "b_follows_a", "tweet_type", "media", "a_is_verified"],
                    ] >> nvt.ops.TargetEncoding(
        labels,
        kfold=5,
        p_smooth=20,
        out_dtype="float32",
    )

    output = count_encode + hour + minute + seconds + diff_lag + labels + target_encode
    # (output).graph

    df_tmp = cudf.read_parquet(OUTPUT_DIR + "/preprocess/part.0.parquet")
    all_input_columns = df_tmp.columns
    del df_tmp
    gc.collect()

    remaining_columns = [x for x in all_input_columns if x not in (output.columns + ["text_tokens"])]

    proc = nvt.Workflow(output + remaining_columns)

    # train_dataset = nvt.Dataset(
    #     glob.glob(OUTPUT_DIR + "nv_train/*.parquet"), engine="parquet", part_size="2GB"
    # )
    # valid_dataset = nvt.Dataset(
    #     glob.glob(OUTPUT_DIR + "nv_valid/*.parquet"), engine="parquet", part_size="2GB"
    # )
    dataset_to_be_processed = nvt.Dataset(
        glob.glob(OUTPUT_DIR + "preprocess/*.parquet"), engine="parquet", part_size="2GB"
    )

    time_fe_start = time.time()
    proc.fit(dataset_to_be_processed)
    time_fe = time.time() - time_fe_start

    dict_dtypes = {}
    for col in ["a_is_verified", "b_is_verified", "b_follows_a"]:
        dict_dtypes[col] = np.int8

    time_fe_start = time.time()
    # proc.transform(train_dataset).to_parquet(
    #     output_path=OUTPUT_DIR + "nv_train_fe/", dtypes=dict_dtypes
    # )
    # proc.transform(valid_dataset).to_parquet(
    #     output_path=OUTPUT_DIR + "nv_valid_fe/", dtypes=dict_dtypes
    # )
    proc.transform(dataset_to_be_processed).to_parquet(
        output_path=OUTPUT_DIR + "nv_dataset_fe/", dtypes=dict_dtypes
    )
    time_fe += time.time() - time_fe_start


if __name__ == '__main__':
    # time_total_start = time.time()

    # INPUT_DATA_DIR = os.environ.get("~/../../mnt/d/summer2022/recsys2021-val/", "/dataset/")
    # OUTPUT_DIR = os.environ.get("~/../../mnt/d/summer2022/recsys2021-val/result", "./")

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", default="/lustre03/project/6001735/qiaodan/MicroParade1/datasets/")
    parser.add_argument("--output_dir", default="/lustre03/project/6001735/qiaodan/MicroParade1/datasets/")
    args = parser.parse_args()

    INPUT_DATA_DIR = os.environ.get(args.input_dir, "/dataset/")
    OUTPUT_DIR = os.environ.get(args.output_dir, "./")

    file_name = 'val.tsv'

    prep_dataset(INPUT_DATA_DIR, OUTPUT_DIR)

    feature_engineering(OUTPUT_DIR)
