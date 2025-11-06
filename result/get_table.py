# %%
from utils.framework import Framework
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
# %%
def run_metrics(control):
    drift_position = control['synth_control']['position']
    drift_width = control['synth_control']['width']
    start_point = drift_position - drift_width / 2
    end_point = drift_position + drift_width / 2
    falses = []
    detecteds = []
    misseds = []
    delays = []
    for i in tqdm(range(NUM_RUNS), desc="运行种子"):
        control['seed'] = i
        work = Framework(**control)
        detections = work.start_synth()
        false = 0
        detected = 0
        missed = 0
        delay = np.nan
        if not detections:
            missed = 1
        else:
            first_detection = detections[0]
            if first_detection < start_point:
                false = 1
            elif start_point <= first_detection <= end_point:
                detected = 1
                delay = first_detection - start_point 
            else:
                missed = 1
        falses.append(false)
        detecteds.append(detected)
        misseds.append(missed)
        delays.append(delay)
    total_falses = np.sum(falses)
    total_detecteds = np.sum(detecteds)
    total_misseds = np.sum(misseds)
    Precision = total_detecteds / (total_detecteds + total_falses)
    Recall = total_detecteds / (total_detecteds + total_misseds)
    F1_Score = 2 * (Precision * Recall) / (Precision + Recall)
    mean_delay = np.nanmean(delays)
    return total_falses, total_detecteds, total_misseds, mean_delay, Precision, Recall, F1_Score
# %%
def run_k(k_list, control):
    results = []
    k_name = []
    for k in tqdm(k_list, desc='运行k数'):
        control['k'] = k
        false, detected, missed, delay, precision, recall, f1 = run_metrics(control)
        results.append([false, detected, missed, delay, precision, recall, f1])
        k_name.append('k=' + str(k))
    df = pd.DataFrame(results, columns=['false', 'detected', 'missed', 'delay', 'precision', 'recall', 'f1'], index=k_name)
    return df
# %%
def run_noise(noises, k_list, control):
    results = []
    noise_name = []
    for noise in tqdm(noises, desc='运行噪声'):
        control['synth_control']['noise_percentage'] = noise
        df = run_k(k_list, control)
        results.append(df)
        noise_name.append('noise=' + str(noise))
    combined_df = pd.concat(results, axis=0, keys=noise_name)
    return combined_df
# %%
def run_width(widths, noises, k_list, control):
    results = []
    width_name = []
    for width in tqdm(widths, desc='运行宽度'):
        control['synth_control']['width'] = width
        df = run_noise(noises, k_list, control)
        results.append(df)
        width_name.append('width=' + str(width))
    combined_df = pd.concat(results, axis=0, keys=width_name)
    return combined_df
# %%
def run_drifter(drifters, widths, noises, k_list, control):
    results = []
    for drifter in tqdm(drifters, desc='运行检测器'):
        control['drifter'] = drifter
        df = run_width(widths, noises, k_list, control)
        results.append(df)
    combined_df = pd.concat(results, axis=1, keys=drifters)
    return combined_df
# %%
def run_model(sheet_name, drifters, widths, noises, k_list, control):
    df = run_drifter(drifters, widths, noises, k_list, control)
    # with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    #     df.to_excel(writer, sheet_name=sheet_name, index=True)
    return df
# %%
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    control = {
        'path': 'RBF',
        'synth_control': None,
        'RecData_control': None,
        'k': 3,
        'train_size': 500,
        'test_size': 5000,
        'seed': 42,
        'model': 'MLP',
        'drifter': 'DDM'}
    control['synth_control'] = {
        'noise_percentage': 0,
        'position': 2500,
        'width': 500}
    NUM_RUNS = 100
    k_list = [10]
    noises = [0.01, 0.05, 0.1, 0.2]
    drifters = ['Topk-DDM', 'DDM', 'MWDDM-H', 'MWDDM-M', 'VFDDM-H', 'VFDDM-M', 'VFDDM-K', 'EDDM']
    widths = [1000, 2000]
    path = 'table/results.xlsx'
    sheet_name = 'result'
    results = run_model(sheet_name, drifters, widths, noises, k_list, control)