# %%
import pandas as pd
import numpy as np
import time
from utils.synth_data import LED, RBF, RecData
from utils.util import Topk_DDM
from river.drift import ADWIN, NoDrift
from river.drift.binary import DDM, EDDM, HDDM_A, HDDM_W
from utils.other_drifter import BDDM, MWDDM, VFDDM
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from utils.evaluate_metrics import hit, compute_hit_rate, compute_ndcg
from cornac.data import Dataset
from cornac.models import BPR, BiVAECF, IBPR, HPF, MF, SVD
# %%


class Framework:
    def __init__(self,
                 path: str | None = None,
                 synth_control: dict | None = None,
                 RecData_control: dict | None = None,
                 k: int = 3,
                 train_size: int = 500,
                 vaild_size: int = 0,
                 test_size: int = 10000,
                 seed: int = 42,
                 model: str = 'LGBM',
                 drifter: str = 'DB_DDM'):
        self.path = path
        self.synth_control = synth_control
        self.RecData_control = RecData_control
        self.k = k
        self.train_size = train_size
        self.vaild_size = train_size if vaild_size is None else vaild_size
        self.test_size = test_size
        self.sample_size = train_size + test_size
        self.seed = seed
        self.model_fit(model)
        self.drifter_fit(drifter)
        self.drifter_name = drifter

    def load_data(self):
        if self.path == 'LED':
            data = LED(self.train_size, self.vaild_size, self.test_size, seed=self.seed, **self.synth_control)
        elif self.path == 'RBF':
            data = RBF(self.train_size, self.vaild_size, self.test_size, seed=self.seed, **self.synth_control)
        elif self.path == 'RecData':
            data = RecData(self.train_size, self.vaild_size, self.test_size, self.seed, **self.RecData_control)
            return data
        elif self.path == 'amazon': 
            path = '../dataset/DataSet/AMAZON/'
            return path
        elif self.path == 'ml-latest':
            path = '../dataset/DataSet/Ml-Latest/'
            return path
        elif self.path == 'twitch':
            path = '../dataset/DataSet/Twitch/'
            return  path
        else:
            raise ValueError
        X = data.iloc[:, :-1].to_numpy()
        Y = data.iloc[:, -1].to_numpy()
        x_train = X[:self.train_size]
        x_vaild = X[self.train_size: self.train_size + self.vaild_size]
        x_test = X[self.train_size + self.vaild_size:]
        y_train = Y[:self.train_size]
        y_vaild = Y[self.train_size: self.train_size + self.vaild_size]
        y_test = Y[self.train_size + self.vaild_size:]
        return x_train, x_vaild, x_test, y_train, y_vaild, y_test

    def model_fit(self, model):
        if model == 'CAT':
            self.model = CatBoostClassifier(random_state=self.seed, logging_level='Silent', allow_writing_files=False)
        elif model == 'LGBM': 
            self.model = LGBMClassifier(random_state=self.seed, verbosity=-1)
        elif model == 'RF': 
            self.model = RandomForestClassifier(random_state=self.seed)
        elif model == 'NB': 
            self.model = GaussianNB()
        elif model == 'KNN': 
            self.model = KNeighborsClassifier()
        elif model == 'MLP': 
            self.model = MLPClassifier(random_state=self.seed)
        else: 
            raise ValueError("Supported models are: CAT, LGBM, RF, NB, KNN, MLP.")

    def drifter_fit(self, drifter):
        self.renew = False
        self.drifter_refit = True
        if drifter == 'Topk-DDM':
            self.drifter = Topk_DDM(k=self.k, min_batch_samples=500)
            self.renew = True
        elif drifter == 'DDM': 
            self.drifter = DDM()
        elif drifter == 'EDDM': 
            self.drifter = EDDM()
        elif drifter == 'ADWIN': 
            self.drifter = ADWIN()
        elif drifter == 'HDDM-A': 
            self.drifter = HDDM_A()
        elif drifter == 'HDDM-W': 
            self.drifter = HDDM_W()
        elif drifter == 'BDDM': 
            self.drifter = BDDM()
        elif drifter == 'MWDDM-H': 
            self.drifter = MWDDM(mode='H')
        elif drifter == 'MWDDM-M': 
            self.drifter = MWDDM(mode='M')
        elif drifter == 'VFDDM-H': 
            self.drifter = VFDDM(test_type='H')
        elif drifter == 'VFDDM-M': 
            self.drifter = VFDDM(test_type='M')
        elif drifter == 'VFDDM-K': 
            self.drifter = VFDDM(test_type='K')
        elif drifter == 'Static Baseline': 
            self.drifter = NoDrift()
            self.drifter_refit = False
        else:
            raise ValueError()
    
    def creat_recmodel(self, model, init_params=None):
        if model == 'BPR':
            self.model = BPR(k=50, seed=self.seed, max_iter=500, init_params=init_params)
            del_offset = True
        elif model == 'BiVAECF': 
            self.model = BiVAECF(k=50, likelihood='bern', n_epochs=500, seed=self.seed)
            del_offset = False
        elif model == 'IBPR': 
            self.model = IBPR(k=50, max_iter=500, init_params=init_params)
            del_offset = False
        elif model == 'HPF': 
            self.model = HPF(k=50, max_iter=500, seed=self.seed, init_params=init_params)
            del_offset = False
        elif model == 'MF': 
            self.model = MF(k=50, backend='pytorch', optimizer='adam', max_iter=100, seed=self.seed, init_params=init_params)
            del_offset = True
        elif model == 'SVD': 
            self.model = SVD(k=50, max_iter=500, seed=self.seed, init_params=init_params)
            del_offset = True
        else:
            raise ValueError()
        return del_offset

    def start_synth(self):
        x_train, x_vaild, x_test, y_train, y_vaild, y_test = self.load_data()
        num_class = np.max(np.concatenate((y_train, y_vaild, y_test))) + 1
        self.model.fit(x_train, y_train)
        if self.renew:
            ref_x = x_train
            ref_y = np.eye(num_class)[y_train]
            self.drifter.set_reference(x=ref_x, y=ref_y)
        top_k_list, y_true_list = [], []
        detections = []
        for i, (x, y) in enumerate(zip(x_test, y_test)):
            y_prob = self.model.predict_proba([x])[0]
            top_k = self.model.classes_[np.argsort(-y_prob)[:self.k]]
            top_k_list.append(top_k)
            y_true_list.append(y)
            hits = int(y in top_k)
            error = 1 - hits
            # Update drifter
            user = x
            top_item = np.eye(num_class)[top_k]
            rank = np.where(top_k == y)[0] + 1 if hits else None
            if self.renew:
                self.drifter.update(x=user, y=top_item, rank=rank)
            else:
                self.drifter.update(error)
            if self.drifter.drift_detected and self.drifter_refit:
                detections.append(i)
        return detections
        
    def start_recdata(self, model, refit_times):
        df = self.load_data()
        full_df = list(df[['user_id', 'item_id', 'interaction']].itertuples(index=False, name=None))
        global_dataset = Dataset.build(full_df, seed=self.seed)
        uid_map = global_dataset.uid_map
        iid_map = global_dataset.iid_map
        train_df = df.iloc[:self.train_size][['user_id', 'item_id', 'interaction']].copy()
        vaild_df = df.iloc[self.train_size: self.train_size + self.vaild_size][['user_id', 'item_id', 'interaction']].copy()
        test_df = df.iloc[self.train_size + self.vaild_size:][['user_id', 'item_id', 'interaction']].copy()
        rows = list(train_df.itertuples(index=False, name=None))
        train_set = Dataset.build(rows, global_uid_map=uid_map, global_iid_map=iid_map, seed=self.seed)
        del_offset = self.creat_recmodel(model)
        self.model.fit(train_set)
        if del_offset:
            user_embed_matrix = self.model.get_user_vectors()[:, :-1]
            item_embed_matrix = self.model.get_item_vectors()[:, :-1]
        else:
            user_embed_matrix = self.model.get_user_vectors()
            item_embed_matrix = self.model.get_item_vectors()
        if self.renew:
            user_indices, item_indeces, _ = train_set.uir_tuple
            x_set = user_embed_matrix[user_indices]
            y_set = item_embed_matrix[item_indeces]
            self.drifter.set_reference(np.array(x_set), np.array(y_set))
        
        HR_list, NDCG_list, refits = [], [], []
        # rows = []  # 新增修改，None
        HR_sum, NDCG_sum = 0.0, 0.0
        start_cpu_time = time.process_time()
        for i, row in enumerate(test_df.itertuples(index = False)):
            rows.append(tuple(row)) # !!
            user = uid_map[row.user_id]
            item = iid_map[row.item_id]
            top_k, _ = self.model.rank(user_idx=user)
            top_k = top_k[: self.k]
            hits = int(item in top_k)
            error = 1 - hits
            # calculate hr and ndcg
            hit_value, ndcg_value = 0, 0.0
            if hits:
                hit_value = 1
                position = np.where(top_k == item)[0][0]
                ndcg_value = 1.0 / np.log2(position + 2)
            HR_sum += hit_value
            NDCG_sum += ndcg_value
            HR_list.append(HR_sum / (i+1))
            NDCG_list.append(NDCG_sum/ (i+1))
            # update
            x = user_embed_matrix[user]
            y = item_embed_matrix[top_k]
            rank = np.array([position + 1]) if hits else None
            if self.renew:
                self.drifter.update(x=x, y=y, rank=rank)
            else:
                self.drifter.update(error)
            if self.drifter.drift_detected and self.drifter_refit and len(refits) <= refit_times:
                refit_data = rows[-self.train_size:] # 新增修改 refit_data = rows[-self.train_size:], refit_data = rows
                refit_set = Dataset.build(refit_data, global_uid_map=uid_map, global_iid_map=iid_map, seed=self.seed)
                self.model.fit(refit_set)
                if del_offset:
                    user_embed_matrix = self.model.get_user_vectors()[:, :-1]
                    item_embed_matrix = self.model.get_item_vectors()[:, :-1]
                else:
                    user_embed_matrix = self.model.get_user_vectors()
                    item_embed_matrix = self.model.get_item_vectors()
                refits.append(i)
                # rows = []  # 新增修改， None
        end_cpu_time = time.process_time()
        cpu_elapsed = end_cpu_time - start_cpu_time
        return HR_list, NDCG_list, refits, cpu_elapsed
    
    def start_realdata(self, model, k_fold, refit):
        path = self.load_data()
        data = Dataset.load(path + 'global_dataset.pkl')
        uid_map = data.uid_map
        iid_map = data.iid_map
        num_users = data.num_users
        num_items = data.num_items
        data = Dataset.load(path + 'train_data.pkl')
        self.creat_recmodel(model)
        self.model.fit(data)
        user_embed_matrix = self.model.get_user_vectors()[:, :-1]
        item_embed_matrix = self.model.get_item_vectors()[:, :-1]
        item_bias = self.model.get_item_vectors()[:, -1]
        if self.renew:
            user_ids, item_ids, _ = data.uir_tuple
            x_set = user_embed_matrix[user_ids]
            y_set = item_embed_matrix[item_ids]
            self.drifter.set_reference(np.array(x_set), np.array(y_set))
        HR, NDCG = [], []
        user_ids, item_ids, rating_ids = [], [], []
        refit_times = 0
        start_cpu_time = time.process_time()
        for i in range(k_fold):
            topk_list, item_list = [], []
            data = Dataset.load(path + f'test_data_fold_{i}.pkl')
            for user, (items, ratings, times) in data.chrono_user_data.items():
                for idx in range(len(items)):
                    user_ids.append(user)
                    item_ids.append(items[idx])
                    rating_ids.append(ratings[idx])
                top_k, _ = self.model.rank(user_idx=user)
                top_k = top_k[: self.k]
                topk_list.append(top_k)
                item_list.append(items)
                hits = hit(top_k, items)
                error = 1 - hits
                # update
                top_k = list(top_k)
                x = user_embed_matrix[user]
                y = item_embed_matrix[top_k]
                rank = np.array([top_k.index(item)+1 for item in set(items)&set(top_k)]) if hits else None
                if self.renew:
                    self.drifter.update(x=x, y=y, rank=rank)
                else:
                    self.drifter.update(error)
                if self.drifter.drift_detected and self.drifter_refit and refit_times < refit:
                    refit_times += 1
                    refit_data = (np.array(user_ids), np.array(item_ids), np.array(rating_ids))
                    refit_set = Dataset(num_users=num_users, 
                                        num_items=num_items, 
                                        uid_map=uid_map, 
                                        iid_map=iid_map, 
                                        uir_tuple=refit_data,
                                        seed = self.seed)
                    param = {'U': user_embed_matrix.astype(np.float64), 
                             'V': item_embed_matrix.astype(np.float64), 
                             'Bi': item_bias.astype(np.float64)}
                    self.creat_recmodel(model, init_params=param)
                    self.model.fit(refit_set)
                    user_embed_matrix = self.model.get_user_vectors()[:, :-1]
                    item_embed_matrix = self.model.get_item_vectors()[:, :-1]
                    item_bias = self.model.get_item_vectors()[:, -1]
                    user_ids, item_ids, rating_ids = [], [], []
            if not self.drifter_refit:
                refit_set = data
                param = {'U': user_embed_matrix.astype(np.float64), 
                         'V': item_embed_matrix.astype(np.float64), 
                         'Bi': item_bias.astype(np.float64)}
                self.creat_recmodel(model, init_params=param)
                self.model.fit(refit_set)
                user_embed_matrix = self.model.get_user_vectors()[:, :-1]
                item_embed_matrix = self.model.get_item_vectors()[:, :-1]
                item_bias = self.model.get_item_vectors()[:, -1]
                user_ids, item_ids, rating_ids = [], [], []
            HR.append(compute_hit_rate(item_list, topk_list))
            NDCG.append(compute_ndcg(item_list, topk_list, self.k))
        end_cpu_time = time.process_time()
        cpu_elapsed = end_cpu_time - start_cpu_time
        return HR, NDCG, refit_times, cpu_elapsed

# %%
if __name__ == '__main__':
    control = {
        'path': 'RBF', #RecData, RBF, amazon, ml-10m, twitch
        'synth_control': None,
        'RecData_control': None,
        'k': 25,
        'train_size': 500,
        'vaild_size': 0,
        'test_size': 20000,
        'seed': 42,
        'model': 'MLP',
        'drifter': 'Topk-DDM'} #Topk-DDM, DDM, Static Baseline
    control['synth_control'] = {
        'noise_percentage': 0.01,
        'position': 2500,
        'width': 500}
    control['RecData_control'] = {
        'n_users': 50,
        'n_items': 50,
        'n_features': 5}
    control['drifter'] = 'Topk-DDM'
    
    work = Framework(**control)
    df = work.start_synth()