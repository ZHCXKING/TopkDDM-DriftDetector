from sklearn.metrics import accuracy_score
from scipy.stats import norm
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

class Classifier_Test:
    
    def __init__(self, alpha: float = 0.01, seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        self.reset()
        
    def reset(self):
        self.P_value = None
        self.statistic = None
        self.detected = False
        
    def test(self, ref_data: dict, new_data: dict):
        """Use AUC as test statistic (existing version)"""
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

    def test_with_accuracy(self, ref_data: dict, new_data: dict):
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
