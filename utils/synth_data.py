# %%
import pandas as pd
import numpy as np
from scipy.special import expit
import math
from river.datasets.synth import LEDDrift, ConceptDriftStream, RandomRBFDrift
# %%


def LED(train_size, vaild_size, test_size, seed, noise_percentage, position, width):
    position = train_size + vaild_size + position
    stream = LEDDrift(seed=seed, noise_percentage=noise_percentage,
                      irrelevant_features=True, n_drift_features=0)
    drift_stream = LEDDrift(seed=seed, noise_percentage=noise_percentage,
                            irrelevant_features=True, n_drift_features=5)
    dataset = ConceptDriftStream(
        stream=stream, drift_stream=drift_stream, seed=seed, position=position, width=width)
    X_list = []
    Y_list = []
    for x, y in dataset.take(train_size + vaild_size + test_size):
        X_list.append(x)
        Y_list.append(y)
    df = pd.DataFrame(X_list)
    df['target'] = Y_list
    return df
# %%
def RBF(train_size, vaild_size, test_size, seed, noise_percentage, position, width):
    position = train_size + vaild_size + position
    stream = RandomRBFDrift(seed_model=seed, seed_sample=seed, n_classes=50, n_features=5, change_speed=0.0)
    drift_stream = RandomRBFDrift(seed_model=seed, seed_sample=seed, n_classes=50, n_features=5, change_speed=100)
    dataset = ConceptDriftStream(
        stream=stream, drift_stream=drift_stream, seed=seed, position=position, width=width)
    X_list = []
    Y_list = []
    noise_labels = list(range(10))
    rng = np.random.default_rng(seed)
    for x, y in dataset.take(train_size + vaild_size + test_size):
        if noise_percentage > 0 and rng.random() < noise_percentage:
            original_label = y
            labels = [l for l in noise_labels if l != original_label]
            y = rng.choice(labels)
        X_list.append(x)
        Y_list.append(y)
    df = pd.DataFrame(X_list)
    df['target'] = Y_list
    return df
# %%
def RecData(train_size, vaild_size, test_size, seed, n_users, n_items, n_features):
    total_timesteps = train_size + vaild_size + test_size
    rng = np.random.default_rng(seed)
    # --- 1. 漂移事件配置 (核心改造部分) ---
    drift_events = {
        0: {
            'type': 'single_focus_shift',
            'params': {
                'n_drift_users': n_users,         # 影响所有用户
                'feature_to_focus_on': 0,         # 用户将专注于第 0 个特征
                'new_low_preference': 0.05,       # 其他特征的偏好值上限
                'new_high_preference': 0.95,      # 专注特征的偏好值下限
            }
        },
        5000 + train_size + vaild_size: {
            'type': 'single_focus_shift',
            'params': {
                'n_drift_users': n_users,         # 影响所有用户
                'feature_to_focus_on': 1,         # 用户将专注于第 0 个特征
                'new_low_preference': 0.05,       # 其他特征的偏好值上限
                'new_high_preference': 0.95,      # 专注特征的偏好值下限
            }
        }
    }
    # --- 2. 初始化 ---
    user_preferences = rng.random((n_users, n_features))
    item_preferences = rng.random((n_items, n_features))
    # 定义物品偏好的分化程度
    item_low_pref = 0.05
    item_high_pref = 0.95
    # 将所有物品的ID分成 n_features 个组
    item_indices = np.arange(n_items)
    # np.array_split 可以处理无法整除的情况
    item_chunks = np.array_split(item_indices, n_features)
    # 为每个物品组分配一个它所专注的特征
    for focus_feature_idx, item_indices_chunk in enumerate(item_chunks):
        # 1. 找出所有需要降低偏好的特征索引
        all_feature_indices = np.arange(n_features)
        features_to_decrease_idx = np.delete(all_feature_indices, focus_feature_idx)
        # 2. 执行两极分化
        # a) 将专注特征的偏好提高到接近 1
        item_preferences[item_indices_chunk, focus_feature_idx] = \
            item_high_pref + rng.random(len(item_indices_chunk)) * (1 - item_high_pref)
        # b) 将所有其他特征的偏好降低到接近 0
        item_preferences[np.ix_(item_indices_chunk, features_to_decrease_idx)] = \
            rng.random((len(item_indices_chunk), len(features_to_decrease_idx))) * item_low_pref
    data = []
    # --- 3. 数据生成循环 (核心改造部分) ---
    for t in range(total_timesteps):
        # 检查是否到达漂移时刻
        if t in drift_events:
            # 保存漂移前的状态用于验证
            event = drift_events[t]
            params = event['params']
            # 获取参数
            n_drift_users = params['n_drift_users']
            feature_to_focus_idx = params['feature_to_focus_on']
            new_low_pref = params['new_low_preference']
            new_high_pref = params['new_high_preference']
            # --- 核心修改逻辑开始 ---
            # 1. 找出所有需要降低偏好的特征索引
            all_feature_indices = np.arange(n_features)
            features_to_decrease_idx = np.delete(all_feature_indices, feature_to_focus_idx)
            # 2. 选择要经历兴趣漂移的用户
            drift_users_idx = rng.choice(n_users, n_drift_users, replace=False)
            # 3. 执行漂移：提升一个特征，降低所有其他特征
            # a) 将专注特征的偏好提高到接近 1
            user_preferences[drift_users_idx, feature_to_focus_idx] = \
                new_high_pref + rng.random(len(drift_users_idx)) * (1 - new_high_pref)
            # b) 将所有其他特征的偏好降低到接近 0
            # 使用高级索引高效地更新矩阵的子块
            user_preferences[np.ix_(drift_users_idx, features_to_decrease_idx)] = \
                rng.random((len(drift_users_idx), len(features_to_decrease_idx))) * new_low_pref
            # --- 核心修改逻辑结束 ---
        # --- 4. 单个交互生成 (与原来相同) ---
        interaction_bool = False
        while not interaction_bool:
            user_id = rng.choice(n_users)
            item_id = rng.choice(n_items)
            user_vec = user_preferences[user_id]
            item_vec = item_preferences[item_id]
            match_score = np.dot(user_vec, item_vec)
            interaction_prob = expit(match_score)
            interaction_bool = rng.random() < interaction_prob
            if interaction_bool:
                data.append({
                    'timestep': t,
                    'user_id': user_id,
                    'item_id': item_id,
                    'interaction': 1
                })
    df = pd.DataFrame(data)
    return df
# %%
if __name__ == '__main__':
    control = {
        'n_users': 50,
        'n_items': 50,
        'n_features': 3,
        'test_size': 10000,
        'train_size': 2500,
        'vaild_size': 2500,
        'seed': 42}
    # control = {
    #     'train_size': 500,
    #     'vaild_size': 500,
    #     'test_size': 5000, 
    #     'seed': 42, 
    #     'noise_percentage': 0.01, 
    #     'position': 2500,
    #     'width': 500}
    df = RecData(**control)
