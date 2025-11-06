from utils.framework import Framework
import numpy as np
import warnings

#%% Configuration parameters
control = {
    'path': 'RBF',  # Select dataset type
    'synth_control': None,  # Parameters for controlling the RBF
    'RecData_control': None,  # Parameters for controlling the RecData
    'k': 10,  # Top-K recommended K value
    'train_size': 1000,  # Number of training samples
    'vaild_size': 0,  # Number of the validation samples
    'test_size': 20000,  # Number of testing samples
    'seed': 6,  # Random seed
    'model': 'MLP',  # Initialize the model
    'drifter': 'Topk-DDM'}  # Drift detector
control['synth_control'] = {
    'noise_percentage': 0.01,  # Noise percentage
    'position': 2500,  # Number of the drift points
    'width': 500}  # Drift range
control['RecData_control'] = {
    'n_users': 100,  # Number of users
    'n_items': 50,  # Number of items
    'n_features': 5}  # Latent factor dimension
warnings.filterwarnings("ignore")

#%% Running RBF experiments
control['path'] = 'RBF'
framework = Framework(**control)
detections = framework.start_synth()
print(detections)  # Location where the output drift detector issues an alarm

#%% Running RecData experiments
control['path'] = 'RecData'
framework = Framework(**control)
model = 'BPR'  # Recommendation Model
refit_times = 10  # Recommended number of model retraining times
HR_list, NDCG_list, refits, cpu_elapsed = framework.start_recdata(model, refit_times)
print(f"HR@10: {np.mean(HR_list)}")
print(f"NDCG@10: {np.mean(NDCG_list)}")
print(f"refits: {len(refits)}")
print(f"runtime: {cpu_elapsed}")