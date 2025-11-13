# %%
import pandas as pd
import numpy as np
import math
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from sklearn.metrics import accuracy_score
# %%
class DDM:
    
    def __init__(self, warning_threshold: float = 2.0, drift_threshold: float = 3.0, warm_start: int = 30):

        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.warm_start = warm_start
        self.reset()

    def reset(self):

        self.n = 0                  # Number of samples processed
        self.error_mean = 0         # Mean of the error rate
        self.p_min = None           # Minimum recorded error rate
        self.s_min = None           # Standard deviation at the minimum error rate
        # The sum of p_min and s_min, used for comparison
        self.ps_min = float('inf')
        self.warning_detected = False
        self.drift_detected = False

    def update(self, error: int):
        
        self.n += 1
        # Update the mean error rate incrementally
        self.error_mean += (error - self.error_mean) / self.n
        # p is the probability of error (the current error rate)
        p = self.error_mean
        # s is the standard deviation of the error rate
        s = math.sqrt(p * (1 - p) / self.n)
        # Start detection only after the warm-start period
        if self.n > self.warm_start:
            # If the current p + s is the smallest seen so far, update the minimums
            if p + s <= self.ps_min:
                self.p_min = p
                self.s_min = s
                self.ps_min = self.p_min + self.s_min
            # Check for a warning signal
            # A warning is triggered if the current error rate exceeds the minimum by a certain threshold
            if p + s > self.p_min + self.warning_threshold * self.s_min:
                self.warning_detected = True
            else:
                self.warning_detected = False
            # Check for a drift signal
            # A drift is triggered if the error rate exceeds the minimum by a larger threshold
            if p + s > self.p_min + self.drift_threshold * self.s_min:
                self.drift_detected = True
                self.warning_detected = False
# %%
class Classifier_Test:
    
    def __init__(self,
                 alpha: float = 0.01,
                 seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        self.reset()
        
    def reset(self):
        self.P_value = None
        self.statistic = None
        self.detected = False
        
    def test_AUC(self, ref_data: dict, new_data: dict):
        """Use classification AUC as test statistic (C2ST-style)"""
        ref_features = np.hstack([ref_data['x'], ref_data['y'], ref_data['rank']])
        new_features = np.hstack([new_data['x'], new_data['y'], new_data['rank']])
        all_features = np.vstack([ref_features, new_features])
        all_labels = np.hstack([np.zeros(ref_features.shape[0]), np.ones(new_features.shape[0])])
        x_train, x_test, y_train, y_test = train_test_split(
            all_features, all_labels, test_size=0.3, random_state=self.seed, stratify=all_labels)
        classifier = LGBMClassifier(random_state=self.seed, verbosity=-1)
        classifier.fit(x_train, y_train)
        y_pred_proba = classifier.predict_proba(x_test)[:, 1]
        self.statistic = roc_auc_score(y_test, y_pred_proba)
        n_pos = np.sum(y_test == 1)
        n_neg = np.sum(y_test == 0)
        
        if n_pos == 0 or n_neg == 0:
            self.P_value = 1.0
        else:
            mu = 0.5
            variance = (n_pos + n_neg + 1) / (12 * n_pos * n_neg)
            sigma = np.sqrt(variance)
            correction = 1 / (2 * n_pos * n_neg)
            statistic_corrected = self.statistic - correction
            z_score = (statistic_corrected - mu) / sigma
            self.P_value = norm.sf(z_score)
        self.detected = self.P_value < self.alpha
    
    def test(self, ref_data: dict, new_data: dict):
        """Use classification accuracy as test statistic (C2ST-style)"""
        ref_features = np.hstack([ref_data['x'], ref_data['y'], ref_data['rank']])
        new_features = np.hstack([new_data['x'], new_data['y'], new_data['rank']])
        all_features = np.vstack([ref_features, new_features])
        all_labels = np.hstack([np.zeros(ref_features.shape[0]), np.ones(new_features.shape[0])])
        x_train, x_test, y_train, y_test = train_test_split(
            all_features, all_labels, test_size=0.3, random_state=self.seed, stratify=all_labels)
        classifier = LGBMClassifier(random_state=self.seed, verbosity=-1)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        self.statistic = accuracy_score(y_test, y_pred)
        # Under H0: accuracy ~ N(0.5, 0.25/n)
        n = len(y_test)
        mu = 0.5
        sigma = np.sqrt(0.25 / n)
        z_score = (self.statistic - mu) / sigma
        self.P_value = norm.sf(z_score)  # one-sided test
        self.detected = self.P_value < self.alpha
# %%
class Topk_DDM:
    
    def __init__(self,
                 k: int,
                 warning_threshold: float = 2.0,
                 drift_threshold: float = 3.0,
                 warm_start: int = 30,
                 alpha: float = 0.01,
                 seed: int = 42,
                 replace_bool: bool = True,
                 min_batch_samples: int | None = None,
                 Placeholder: int | None = None,
                 verification_test: str = 'Classifier'):
        self.k = k
        self.DDM = DDM(warning_threshold, drift_threshold, warm_start)
        if verification_test == 'Classifier':
            self.test = Classifier_Test(alpha, seed)
        elif verification_test == 'None':
            self.test = None
        self.rng = np.random.default_rng(seed)
        self.replace_bool = replace_bool
        self.min_batch_samples = min_batch_samples
        self.Placeholder = Placeholder if Placeholder is not None else self.k + 1
        self.reference_data = None
        self.reclean()
    def reclean(self):
        self.new_data = []
        self.drift_batch = None
        self.warning_detected = False
        self.drift_detected = False
    def set_reference(self, x: np.ndarray, y: np.ndarray, rank: np.ndarray | None = None):
        if x.ndim != 2:
            raise ValueError(f"`x` must be 2D array, but got shape {x.shape}")
        if y.ndim != 2:
            raise ValueError(f"`y` must be 2D array, but got shape {y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"`x` and `y` must have the same number of rows")
        # 注意，请输入二维的x和y
        if rank is None:
            rank = np.full((x.shape[0], 1), self.Placeholder)
        else: 
            rank = np.asarray(rank, dtype=float)
            rank = np.nan_to_num(rank, nan=self.Placeholder)
        rank = rank.reshape(-1, 1) if rank.ndim == 1 else rank
        self.reference_data = {'x': x, 'y': y, 'rank': rank}
        if self.min_batch_samples is None:
            self.min_batch_samples = len(x)
    def update_monitor(self, rank: np.ndarray | None = None):
        if rank is None:
            error = 1
            self.DDM.update(error)
        else:
            center = self.k // 2
            correctness_probs = 1 / (1 + np.exp(1 * (rank - center)))
            error_probs = 1 - correctness_probs
            errors = (self.rng.random(size=len(rank)) < error_probs).astype(int)
            for error in errors:
                self.DDM.update(error)
    def update(self, x: np.ndarray, y: np.ndarray, rank: np.ndarray | None = None):
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise ValueError(f"`x` must be a 1D numpy array, but got shape {x.shape}")
        if not isinstance(y, np.ndarray) or y.ndim != 2:
            raise ValueError(f"`y` must be a 2D numpy array, but got shape {y.shape}")
        # 请输入一维的x和rank，还有二维的y
        if self.drift_detected:
            if self.replace_bool:
                self.reference_data = self.drift_batch
            self.DDM.reset()
            if self.test is not None:
                self.test.reset()
            self.reclean()
        # 更新DDM
        self.update_monitor(rank)
        # 判断是否添加新数据
        self.warning_detected = self.DDM.warning_detected or self.DDM.drift_detected
        if self.warning_detected:
            if rank is None:
                ranks = np.full(len(y), self.Placeholder)
            else:
                ranks = np.full(len(y), self.Placeholder)
                hit_ranks = np.array(rank, copy=False)
                ranks[hit_ranks - 1] = hit_ranks
            batch_data = [{'x':x, 'y': y[i], 'rank': ranks[i]} for i in range(len(y))]
            self.new_data.extend(batch_data)
        if len(self.new_data) >= self.min_batch_samples and self.DDM.drift_detected:
            df = pd.DataFrame(self.new_data)
            new_data = {
                'x': np.array(df['x'].to_list()),
                'y': np.array(df['y'].to_list()),
                'rank': df['rank'].to_numpy().reshape(-1, 1)}
            # 根据是否有验证机制，来判断是否漂移
            if self.test is None:
                self.drift_detected = True
                self.drift_batch = new_data
            else:
                self.test.test(self.reference_data, new_data)
                if self.test.detected:
                    self.drift_detected = True
                    self.drift_batch = new_data
                else:
                    self.DDM.reset()
                    self.reclean()