# %%
from utils.framework import Framework
import pandas as pd
import numpy as np
from tqdm import tqdm
# %%
def get_seed(control):
    global NUM_RUNS
    global model
    global k_fold
    global refit_times
    global last_k
    total_HR, total_NDCG = [], []
    for i in tqdm(range(NUM_RUNS), desc='运行种子'):
        control['seed'] = i
        work = Framework(**control)
        HR_list, NDCG_list, refit_times, cpu_elapsed = work.start_realdata(model, k_fold, refit_times)
        HR = np.mean(HR_list[-last_k:])
        NDCG = np.mean(NDCG_list[-last_k:])
        total_HR.append(HR)
        total_NDCG.append(NDCG)
    return np.mean(total_HR), np.mean(total_NDCG)
# %%
def get_k(k_list, control):
    result = {}
    for k in tqdm(k_list, desc='运行k数'):
        control['k'] = k
        HR, NDCG = get_seed(control)
        result[f'HR@{k}'] = HR
        result[f'NDCG@{k}'] = NDCG
    df = pd.DataFrame([result])
    return df
# %%
def get_drifter(drifters, k_list, control):
    result = []
    for drifter in tqdm(drifters, desc='运行检测器'):
        control['drifter'] = drifter
        df = get_k(k_list, control)
        result.append(df)
    combined_df = pd.concat(result, axis=0, keys=drifters)
    return combined_df
# %%
model = 'BPR' # BPR, BiVAECF, HPF, MF, SVD
k_fold = 10
refit_times = 10
last_k = 10
control = {
    'path': 'twitch', #amazon, ml-latest, twitch
    'synth_control': None,
    'RecData_control': None,
    'k': 10,
    'train_size': 500,
    'vaild_size': 0,
    'test_size': 20000,
    'seed': 42,
    'model': 'MLP',
    'drifter': 'Topk-DDM'}
NUM_RUNS = 100
k_list = [10]
drifters = ['Topk-DDM', 'DDM', 'MWDDM-H', 'MWDDM-M','VFDDM-H', 'VFDDM-M', 'VFDDM-K', 'EDDM'] #'MWDDM-H', 'MWDDM-M','VFDDM-H', 'VFDDM-M', 'VFDDM-K', 'EDDM'
df = get_drifter(drifters, k_list, control)
df.to_excel('table/RealData.xlsx')