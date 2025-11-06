# %%
"""
https://nijianmo.github.io/amazon/index.html
https://grouplens.org/datasets/movielens/
https://cseweb.ucsd.edu/~jmcauley/datasets.html#twitch
"""
# %%
import pandas as pd
import numpy as np
from cornac.data import Dataset
# %%
k_fold = 10
train_ratio = 0.1
seed = 42
# %%
# data_file = 'Original Dataset/AMAZON/AMAZON_FASHION.csv'
# names = ['item_id', 'user_id', 'rating', 'timestamp']
# df = pd.read_csv(data_file, sep=',', header=None, names=names, engine='python')
# df['rating'] = 1
# # 移除交互次数少于5的用户和物品
# interaction_count = 5
# while True:
#     prev_len = len(df)
#     user_counts = df['user_id'].value_counts()
#     activate_users = user_counts[user_counts >= interaction_count].index
#     df = df[df['user_id'].isin(activate_users)]
#     item_counts = df['item_id'].value_counts()
#     activate_items = item_counts[item_counts >= interaction_count].index
#     df = df[df['item_id'].isin(activate_items)]
#     if len(df) == prev_len:
#         break
# # 开始建立数据集
# df = df.sort_values(by='timestamp').reset_index(drop=True)
# data_tuples = list(df[['user_id', 'item_id', 'rating', 'timestamp']].itertuples(index=False, name=None))
# global_dataset = Dataset.build(data=data_tuples, fmt='UIRT', seed=seed)
# uid_map = global_dataset.uid_map
# iid_map = global_dataset.iid_map
# num_users = global_dataset.num_users
# num_items = global_dataset.num_items
# global_dataset.save('Dataset/AMAZON/global_dataset.pkl')
# # 划分数据集
# split_index = int(len(df) * train_ratio)
# train_df = df.iloc[:split_index].copy()
# test_df = df.iloc[split_index:].copy()
# folds = np.array_split(test_df, k_fold)
# folds = [fold.reset_index(drop=True) for fold in folds]
# # 构建cornac.dataset
# train_tuples = list(train_df[['user_id', 'item_id', 'rating', 'timestamp']].itertuples(index=False, name=None))
# train_data = Dataset.build(train_tuples, fmt='UIRT', global_uid_map=uid_map, global_iid_map=iid_map, seed=seed)
# test_data = []
# for fold in folds:
#     fold_tuples = list(fold[['user_id', 'item_id', 'rating', 'timestamp']].itertuples(index=False, name=None))
#     test_fold_data = Dataset.build(
#         fold_tuples, fmt='UIRT',
#         global_uid_map = uid_map,
#         global_iid_map = iid_map,
#         seed = seed)
#     test_data.append(test_fold_data)
# # 保存数据集
# train_data.save('Dataset/AMAZON/train_data.pkl')
# for i, test_fold in enumerate(test_data):
#     test_fold.save(f'Dataset/AMAZON/test_data_fold_{i}.pkl')
# %%
data_file = 'Original Dataset/Twitch/100k_a.csv'
columns = ['user_id', 'stream_id', 'item_id', 'timestamp', 'time_stop']
df = pd.read_csv(data_file, header=None, names=columns)
df['rating'] = df['time_stop'] - df['timestamp']
df = df[df['rating'] >= 4].copy()
df['rating'] = 1
# 移除交互次数少于20的用户和物品
interaction_count = 20
while True:
    prev_len = len(df)
    user_counts = df['user_id'].value_counts()
    activate_users = user_counts[user_counts >= interaction_count].index
    df = df[df['user_id'].isin(activate_users)]
    item_counts = df['item_id'].value_counts()
    activate_items = item_counts[item_counts >= interaction_count].index
    df = df[df['item_id'].isin(activate_items)]
    if len(df) == prev_len:
        break
# 开始建立数据集
df = df.sort_values(by='timestamp').reset_index(drop=True)
data_tuples = list(df[['user_id', 'item_id', 'rating', 'timestamp']].itertuples(index=False, name=None))
global_dataset = Dataset.build(data=data_tuples, fmt='UIRT', seed=seed)
uid_map = global_dataset.uid_map
iid_map = global_dataset.iid_map
num_users = global_dataset.num_users
num_items = global_dataset.num_items
global_dataset.save('Dataset/Twitch/global_dataset.pkl')
# 划分数据集
split_index = int(len(df) * train_ratio)
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()
folds = np.array_split(test_df, k_fold)
folds = [fold.reset_index(drop=True) for fold in folds]
# 构建cornac.dataset
train_tuples = list(train_df[['user_id', 'item_id', 'rating', 'timestamp']].itertuples(index=False, name=None))
train_data = Dataset.build(train_tuples, fmt='UIRT', global_uid_map=uid_map, global_iid_map=iid_map, seed=seed)
test_data = []
for fold in folds:
    fold_tuples = list(fold[['user_id', 'item_id', 'rating', 'timestamp']].itertuples(index=False, name=None))
    test_fold_data = Dataset.build(
        fold_tuples, fmt='UIRT',
        global_uid_map = uid_map,
        global_iid_map = iid_map,
        seed = seed)
    test_data.append(test_fold_data)
# 保存数据集
train_data.save('Dataset/Twitch/train_data.pkl')
for i, test_fold in enumerate(test_data):
    test_fold.save(f'Dataset/Twitch/test_data_fold_{i}.pkl')
# %%
# data_file = 'Original Dataset/ml-latest-small/ratings.csv'
# names = ['user_id', 'item_id', 'rating', 'timestamp']
# df = pd.read_csv(data_file)
# df.columns = names
# df['rating'] = 1
# # 移除交互次数少于20的用户和物品
# interaction_count = 20
# while True:
#     prev_len = len(df)
#     user_counts = df['user_id'].value_counts()
#     activate_users = user_counts[user_counts >= interaction_count].index
#     df = df[df['user_id'].isin(activate_users)]
#     item_counts = df['item_id'].value_counts()
#     activate_items = item_counts[item_counts >= interaction_count].index
#     df = df[df['item_id'].isin(activate_items)]
#     if len(df) == prev_len:
#         break
# # 开始建立数据集
# df = df.sort_values(by='timestamp').reset_index(drop=True)
# data_tuples = list(df[['user_id', 'item_id', 'rating', 'timestamp']].itertuples(index=False, name=None))
# global_dataset = Dataset.build(data=data_tuples, fmt='UIRT', seed=seed)
# uid_map = global_dataset.uid_map
# iid_map = global_dataset.iid_map
# num_users = global_dataset.num_users
# num_items = global_dataset.num_items
# global_dataset.save('Dataset/Ml-Latest/global_dataset.pkl')
# # 划分数据集
# split_index = int(len(df) * train_ratio)
# train_df = df.iloc[:split_index].copy()
# test_df = df.iloc[split_index:].copy()
# folds = np.array_split(test_df, k_fold)
# folds = [fold.reset_index(drop=True) for fold in folds]
# # 构建cornac.dataset
# train_tuples = list(train_df[['user_id', 'item_id', 'rating', 'timestamp']].itertuples(index=False, name=None))
# train_data = Dataset.build(train_tuples, fmt='UIRT', global_uid_map=uid_map, global_iid_map=iid_map, seed=seed)
# test_data = []
# for fold in folds:
#     fold_tuples = list(fold[['user_id', 'item_id', 'rating', 'timestamp']].itertuples(index=False, name=None))
#     test_fold_data = Dataset.build(
#         fold_tuples, fmt='UIRT',
#         global_uid_map = uid_map,
#         global_iid_map = iid_map,
#         seed = seed)
#     test_data.append(test_fold_data)
# # 保存数据集
# train_data.save('Dataset/Ml-Latest/train_data.pkl')
# for i, test_fold in enumerate(test_data):
#     test_fold.save(f'Dataset/Ml-Latest/test_data_fold_{i}.pkl')