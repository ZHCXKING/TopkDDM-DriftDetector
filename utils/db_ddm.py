# %%
import math
import pandas as pd
import numpy as np
import torch
from typing import Union
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from scipy.special import zeta
from scipy.spatial.distance import cdist
import warnings
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


class MMD_Test:

    def __init__(self, resample: int = 1000, alpha: float = 0.01, seed: int = 42):
        self.resample = resample
        self.alpha = alpha
        
        # 1. 设备管理：自动选择 GPU 或 CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        # 使用 PyTorch 的随机数生成器
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)
        
        self.reset()

    def reset(self):
        self.gamma_z = None
        self.P_value = None
        self.MMD_obs = None
        self.detected = False

    def set_gamma(self, Z: torch.Tensor):
        # Z 已经是 Tensor 并位于目标设备上
        n_total = Z.shape[0]
        n_samples = min(1000, n_total)
        
        # 使用 torch.randperm 进行无放回采样
        sample_indices = torch.randperm(n_total, generator=self.rng, device=self.device)[:n_samples]
        
        # 2. 距离计算在 GPU 上进行
        dists_sq = torch.cdist(Z[sample_indices], Z[sample_indices], p=2)**2
        
        # 获取上三角矩阵的元素（不包括对角线）
        triu_indices = torch.triu_indices(n_samples, n_samples, offset=1, device=self.device)
        median_dists_sq = torch.median(dists_sq[triu_indices[0], triu_indices[1]])
        
        if median_dists_sq == 0:
            median_dists_sq = 1e-6
            
        gamma = 1.0 / median_dists_sq
        
        return gamma

    def compound_kernel(self, Z: torch.Tensor, F: torch.Tensor):
        # 3. 完整的核计算在 GPU 上进行
        dists_sq_z = torch.cdist(Z, Z, p=2)**2
        K_z = torch.exp(-self.gamma_z * dists_sq_z)
        
        K_f = (F.view(-1, 1) == F.T).float()
        return K_z * K_f

    @staticmethod
    def calculate_mmd(K_XX: torch.Tensor, K_YY: torch.Tensor, K_XY: torch.Tensor, m: int, n: int):
        term1 = K_XX.mean() if m > 0 else 0
        term2 = K_YY.mean() if n > 0 else 0
        term3 = 2 * K_XY.mean()
        return term1 + term2 - term3

    def matrix_test(self, Z: torch.Tensor, F: torch.Tensor, m: int):
        N = Z.shape[0]
        n = N - m
        
        K = self.compound_kernel(Z, F)
        K_XX, K_YY, K_XY = K[:m, :m], K[m:, m:], K[:m, m:]
        
        self.MMD_obs = self.calculate_mmd(K_XX, K_YY, K_XY, m, n)
        
        h_base = torch.cat([torch.ones(m, device=self.device) / m, -torch.ones(n, device=self.device) / n])
        
        # 4. 高效的置换检验在 GPU 上进行
        # 生成所有置换索引（这部分在 CPU 可能更快，但为了简化我们放在 GPU）
        permuted_indices = torch.stack([
            torch.randperm(N, generator=self.rng, device=self.device) for _ in range(self.resample)
        ])
        
        H_perm = h_base[permuted_indices]
        
        # torch.einsum 同样高效
        MMD_bootstrap = torch.einsum('ij,jk,ik->i', H_perm, K, H_perm)
        
        # 5. 最终计算并转回 CPU
        count = torch.sum(MMD_bootstrap >= self.MMD_obs).cpu().item()
        self.P_value = (count + 1) / (self.resample + 1)
        self.detected = self.P_value <= self.alpha
        
    def loop_test(self, Z: torch.Tensor, F: torch.Tensor, m: int):
        N = Z.shape[0]
        n = N - m

        # 1. Pre-calculate the full kernel matrix once.
        K = self.compound_kernel(Z, F)

        # 2. Calculate the observed MMD statistic on the original data.
        K_XX, K_YY, K_XY = K[:m, :m], K[m:, m:], K[:m, m:]
        self.MMD_obs = self.calculate_mmd(K_XX, K_YY, K_XY, m, n)

        # 3. Perform permutation test in a loop.
        count = 0
        for _ in range(self.resample):
            # a. Generate a random permutation of indices.
            indices = torch.randperm(N, generator=self.rng, device=self.device)

            # b. Apply the permutation to the kernel matrix to simulate shuffling the data.
            K_perm = K[indices][:, indices]

            # c. Extract sub-matrices from the permuted kernel matrix.
            K_XX_p = K_perm[:m, :m]
            K_YY_p = K_perm[m:, m:]
            K_XY_p = K_perm[:m, m:]

            # d. Calculate the MMD for this permuted sample.
            mmd_p = self.calculate_mmd(K_XX_p, K_YY_p, K_XY_p, m, n)

            # e. Check if the permuted statistic is greater than or equal to the observed one.
            if mmd_p >= self.MMD_obs:
                count += 1

        # 4. Calculate the final p-value and determine if a drift was detected.
        self.P_value = (count + 1) / (self.resample + 1)
        self.detected = self.P_value <= self.alpha

    def test(self, ref_data: dict, new_data: dict):
        # 准备数据（CPU -> GPU 的转换点）
        Z_X_np = np.hstack([ref_data['x'], ref_data['y']])
        F_X_np = ref_data['feedback']
        Z_Y_np = np.hstack([new_data['x'], new_data['y']])
        F_Y_np = new_data['feedback']
        m = Z_X_np.shape[0]  
        # 将 NumPy 数组转换为 PyTorch Tensors 并移动到目标设备
        Z_X = torch.from_numpy(Z_X_np).float().to(self.device)
        F_X = torch.from_numpy(F_X_np).float().to(self.device)
        Z_Y = torch.from_numpy(Z_Y_np).float().to(self.device)
        F_Y = torch.from_numpy(F_Y_np).float().to(self.device)
        
        Z_combined = torch.cat([Z_X, Z_Y], dim=0)
        F_combined = torch.cat([F_X, F_Y], dim=0)
        
        self.gamma_z = self.set_gamma(Z_combined)
        self.matrix_test(Z_combined, F_combined, m)

# %%
class Classifier_Test:
    def __init__(self,
                 alpha: float = 0.05,
                 seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        self.reset()
        
    def reset(self):
        self.P_value = None
        self.statistic = None
        self.detected = False
        
    def test(self, ref_data: dict, new_data: dict):
        warnings.filterwarnings('ignore', category=UserWarning)
        ref_features = np.hstack([ref_data['x'], ref_data['y'], ref_data['feedback']])
        new_features = np.hstack([new_data['x'], new_data['y'], new_data['feedback']])
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
# %%


class DB_DDM:
    def __init__(self,
                 warning_threshold: float = 2.0,
                 drift_threshold: float = 3.0,
                 warm_start: int = 30,
                 resample: int = 1000,
                 alpha: float = 0.01,
                 seed: int = 42,
                 Placeholder: Union[int, list] = -1,
                 replace: bool = True,
                 test: str = 'Classifier'):
        # Minimum number of samples to collect before re-running MMD after a DDM drift signal.
        self.DDM = DDM(warning_threshold, drift_threshold, warm_start)
        if test == 'MMD':
            self.Test = MMD_Test(resample, alpha, seed)
        elif test == 'Classifier': 
            self.Test = Classifier_Test(alpha=alpha, seed=seed)
        self.replace = replace
        self.test = test
        self.Placeholder = Placeholder
        # State variables
        # Reference data for MMD, stored as (x, y, hit)
        self.reference_data = None
        self.min_mmd_samples = 0
        self.reclean()

    def reclean(self):

        self.newdata = []         # New feature data collected after a DDM warning
        self.drift_batch = None
        self.warning_detected = False
        self.drift_detected = False

    def set_reference(self, x: np.ndarray, y: np.ndarray, feedback: np.ndarray | None = None):

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        self.y_dim = y.shape[1]
        if feedback is None:
            feedback = np.ones((x.shape[0], 1))
        else:
            feedback = feedback.reshape(-1, 1) if feedback.ndim == 1 else feedback
        self.reference_data = {'x': x, 'y': y, 'feedback': feedback}
        self.min_mmd_samples = len(x)

    def update(self, x: list | np.ndarray, feedback: float, y: list | np.ndarray | None = None):

        # If a drift was confirmed in the previous step, reset the system.
        if self.drift_detected:
            if self.replace:
                self.reference_data = self.drift_batch
            self.DDM.reset()
            self.Test.reset()
            self.reclean()
        # Step 1: Update DDM with the error of the new sample.
        error = 1.0 - feedback if feedback in [0.0, 1.0] else (1.0 if feedback < 0.5 else 0.0)
        self.DDM.update(error)
        self.warning_detected = self.DDM.warning_detected or self.DDM.drift_detected
        # Step 2: If DDM issues a warning, start collecting new data.
        y_to_store = y
        if y_to_store is None:
            if isinstance(self.Placeholder, int):
                y_to_store = [self.Placeholder] * self.y_dim
            elif isinstance(self.Placeholder, list):
                y_to_store = self.Placeholder
        if self.warning_detected:
            self.newdata.append({'x': x, 'y': y_to_store, 'feedback': feedback})
        # Step 3: If DDM signals a drift and enough new data is collected, run MMD test.
        if len(self.newdata) >= self.min_mmd_samples:
            df = pd.DataFrame(self.newdata)
            new_data = {
                'x': np.array(df['x'].to_list()),
                'y': np.array(df['y'].to_list()),
                'feedback': df['feedback'].to_numpy().reshape(-1, 1)}
            # Perform MMD test on the joint distribution
            self.Test.test(self.reference_data, new_data)
            # If MMD test confirms the drift, set the final drift flag.
            # This would signal that the model needs retraining.
            if self.Test.detected:
                self.drift_detected = True
                self.drift_batch = new_data
            else:
                # If MMD does not confirm drift, reset DDM.
                self.DDM.reset()
                self.reclean()

# %%


class MMD_drifter:

    def __init__(self,
                 resample: int = 1000,
                 alpha: float = 0.01,
                 seed: int = 42,
                 replace: bool = True,
                 Placeholder: Union[int, list] = -1):
        self.MMD_Test = MMD_Test(resample, alpha, seed)
        self.replace = replace
        self.Placeholder = Placeholder
        self.reference_data = None
        self.reclean()
        
    def reclean(self):
        self.newdata = []
        self.drift_batch = None
        self.drift_detected = False
        self.MMD_Test.reset()

    def set_reference(self, x: np.ndarray, y: np.ndarray, feedback: np.ndarray | None = None):
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        self.y_dim = y.shape[1]
        if feedback is None:
            feedback = np.ones((x.shape[0], 1))
        else:
            feedback = feedback.reshape((-1, 1)) if feedback.ndim == 1 else feedback
        self.reference_data = {'x': x, 'y': y, 'feedback': feedback}
        self.batch_size = len(x)
        self.reclean()

    def update(self, x: list, feedback: float, y: list | None = None):
        if self.drift_detected:
            if self.replace:
                self.reference_data = self.drift_batch
            # Reset state for the next cycle
            self.reclean()
        y_to_store = y
        if y_to_store is None:
            if isinstance(self.Placeholder, int):
                y_to_store = [self.Placeholder] * self.y_dim
            elif isinstance(self.Placeholder, list):
                y_to_store = self.Placeholder
        self.newdata.append({'x': x, 'y': y_to_store, 'feedback': feedback})

        # If the buffer is full, run the MMD test
        if len(self.newdata) >= self.batch_size:
            df = pd.DataFrame(self.newdata)
            newdata = {
                'x': np.array(df['x'].to_list()),
                'y': np.array(df['y'].to_list()),
                'feedback': df['feedback'].to_numpy().reshape(-1, 1)}
            # Perform the MMD test
            self.MMD_Test.test(self.reference_data, newdata)
            if self.MMD_Test.detected:
                self.drift_detected = True
                self.drift_batch = newdata
            else:
                # If no drift is detected, clear the buffer to start collecting the next batch.
                self.newdata = []

# %%
class CS_MMD:
    """
    一个两阶段漂移检测器，结合了 DDM 的预警功能和
    用于验证的流式 MMD 置信序列。
    """
    def __init__(self,
                 warning_threshold: float = 2.0,
                 drift_threshold: float = 3.0,
                 warm_start: int = 30,
                 alpha: float = 0.01,
                 seed: int = 42,
                 eta: float = 2.0,
                 s: float = 1.4,
                 m: float = 50.0,
                 placeholder: Union[int, list] = -1,
                 min_samples_for_adaptation: int = 50):

        self.DDM = DDM(warning_threshold, drift_threshold, warm_start)
        self.alpha = alpha
        self.eta = eta
        self.s = s
        self.m = m # 边界变为有限值所需的最小方差 V_t
        self.placeholder = placeholder
        self.min_samples_for_adaptation = min_samples_for_adaptation

        if self.s <= 1:
            raise ValueError("参数 's' 必须大于 1。")

        # 1. 使用 NumPy 的随机数生成器
        self.rng = np.random.default_rng(seed)

        # 内部状态
        self.gamma_z = None
        self.ref_Z = None
        self.ref_F = None
        self.y_dim = None

        # 预计算边界函数的常数
        self._k1 = (np.power(self.eta, 0.25) + np.power(self.eta, -0.25)) / np.sqrt(2)
        self._k2 = (np.sqrt(self.eta) + 1) / 2
        self._log_eta = np.log(self.eta)
        self._zeta_s = zeta(self.s, 1)

        self.reset() # 初始重置

    def reset(self):
        """ 重置整个检测器的状态。 """
        self.ref_Z = None
        self.ref_F = None
        self.DDM.reset()
        self._reclean_streaming_state()

    def _reclean_streaming_state(self):
        """ 仅重置流式验证部分（例如，在发生假警报后）。 """
        self.in_warning_phase = False # 跟踪我们是否处于验证阶段
        self.drift_detected = False
        self.t = 0
        self.mean_Z_statistic = 0.0
        self.V_t = 0.0
        self.stream_history_Z = []
        self.stream_history_F = []

    def _set_gamma(self, Z: np.ndarray):
        n_total = Z.shape[0]
        if n_total < 2:
            self.gamma_z = 1.0
            return
        n_samples = min(1000, n_total)
        
        # 2. 使用 NumPy 的随机排列
        indices = self.rng.permutation(n_total)[:n_samples]
        
        # 3. 使用 SciPy 的 cdist 计算距离
        dists_sq = cdist(Z[indices], Z[indices], metric='euclidean')**2
        
        # 4. 使用 NumPy 获取上三角索引
        triu_indices = np.triu_indices(n_samples, k=1)
        median_dists_sq = np.median(dists_sq[triu_indices])
        self.gamma_z = 1.0 / median_dists_sq if median_dists_sq > 1e-9 else 1.0

    def _compound_kernel(self, Z1: np.ndarray, F1: np.ndarray, Z2: np.ndarray, F2: np.ndarray):
        dists_sq_z = cdist(Z1, Z2, metric='euclidean')**2
        K_z = np.exp(-self.gamma_z * dists_sq_z)
        # NumPy 的广播机制可以实现同样的效果
        K_f = (F1.reshape(-1, 1) == F2.T)
        return K_z * K_f

    def set_reference(self, x: np.ndarray, y: np.ndarray, feedback: np.ndarray | None = None):
        y_reshaped = y.reshape(-1, 1) if y.ndim == 1 else y
        self.y_dim = y_reshaped.shape[1]
        ref_Z_np = np.hstack([x, y_reshaped])

        if feedback is None:
            feedback_reshaped = np.ones((x.shape[0], 1))
        else:
            feedback_reshaped = feedback.reshape(-1, 1) if feedback.ndim == 1 else feedback
        
        # 5. 直接赋值 NumPy 数组
        self.ref_Z = ref_Z_np.astype(np.float32)
        self.ref_F = feedback_reshaped.astype(np.float32)
        
        self._set_gamma(self.ref_Z)
        self.DDM.reset()
        self._reclean_streaming_state()

    def _u_boundary(self, v: float) -> float:
        if v < self.m: return np.inf
        log_eta_v_m = np.log(max(1e-9, v / self.m)) / self._log_eta
        l_v = self.s * np.log(max(1.0, log_eta_v_m)) + np.log(self._zeta_s / (self.alpha * self._log_eta))
        # Z_t 统计量的取值范围为 [-4, 4]，因此其 sub-gamma 尺度参数 c=4 是合理的。
        c = 4.0
        term1 = (self._k1**2) * v * l_v
        term2 = (self._k2*c*l_v)**2
        term3 = self._k2 * c * l_v
        boundary_value = np.sqrt(term1 + term2) + term3
        return boundary_value

    def update(self, x: Union[list, np.ndarray], feedback: float, y: Union[list, np.ndarray, None] = None) -> bool:
        if self.ref_Z is None:
            raise RuntimeError("检测器未初始化。请先调用 .set_reference()。")

        # --- 自适应步骤 ---
        if self.drift_detected:
            if len(self.stream_history_Z) >= self.min_samples_for_adaptation:
                # 6. 使用 NumPy 拼接数组
                new_ref_Z_np = np.concatenate(self.stream_history_Z)
                new_ref_F_np = np.concatenate(self.stream_history_F)
                self.set_reference(
                    x=new_ref_Z_np[:, :-self.y_dim],
                    y=new_ref_Z_np[:, -self.y_dim:],
                    feedback=new_ref_F_np
                )
                return True # 返回 True 表示发生了自适应
            else:
                pass # 等待收集更多数据
        
        # --- 阶段一: DDM 监控 ---
        error = 1.0 - feedback if feedback in [0.0, 1.0] else (1.0 if feedback < 0.5 else 0.0)
        self.DDM.update(error)
        is_ddm_triggered = self.DDM.warning_detected or self.DDM.drift_detected

        # --- 阶段二: 流式验证 (如果被触发) ---
        if is_ddm_triggered:
            if not self.in_warning_phase:
                self.in_warning_phase = True

            # 准备新的数据点
            x_np = np.array(x).reshape(1, -1)
            if y is not None:
                y_np = np.array(y).reshape(1, -1)
            else:
                y_val = [self.placeholder] * self.y_dim if isinstance(self.placeholder, int) else self.placeholder
                y_np = np.array(y_val).reshape(1, -1)
            
            # 7. 创建 NumPy 数组
            new_Z = np.hstack([x_np, y_np]).astype(np.float32)
            new_F = np.array([[feedback]], dtype=np.float32)
            
            self.stream_history_Z.append(new_Z)
            self.stream_history_F.append(new_F)

            if len(self.stream_history_Z) < 2: return self.drift_detected

            # 这个逻辑现在会在警告阶段为每个样本运行
            y_t, f_t = new_Z, new_F
            
            # 8. 使用 NumPy 生成随机整数
            ref_indices = self.rng.integers(0, len(self.ref_Z), size=2)
            # cdist 需要 2D 输入，所以使用 reshape(-1, 1)
            x_t = self.ref_Z[ref_indices[0]].reshape(1, -1)
            x_prime_t = self.ref_Z[ref_indices[1]].reshape(1, -1)
            f_x_t = self.ref_F[ref_indices[0]].reshape(1, -1)
            f_x_prime_t = self.ref_F[ref_indices[1]].reshape(1, -1)
            
            y_prime_idx = self.rng.integers(0, len(self.stream_history_Z) - 1)
            y_prime_t, f_y_prime_t = self.stream_history_Z[y_prime_idx], self.stream_history_F[y_prime_idx]
            
            # 计算 Z_t
            k_xx = self._compound_kernel(x_t, f_x_t, x_prime_t, f_x_prime_t)
            k_yy = self._compound_kernel(y_t, f_t, y_prime_t, f_y_prime_t)
            k_xy = self._compound_kernel(x_t, f_x_t, y_prime_t, f_y_prime_t)
            k_yx = self._compound_kernel(x_prime_t, f_x_prime_t, y_t, f_t)
            
            # .item() 从 1x1 的数组中提取标量值
            Z_t_statistic = (k_xx + k_yy - k_xy - k_yx).item()

            self.t += 1
            last_mean = self.mean_Z_statistic
            self.mean_Z_statistic += (Z_t_statistic - last_mean) / self.t
            if self.t > 1:
                self.V_t += (Z_t_statistic - last_mean) * (Z_t_statistic - self.mean_Z_statistic)

            u_val = self._u_boundary(self.V_t)
            if self.t > 0 and u_val != np.inf:
                radius = u_val / self.t
                if self.mean_Z_statistic - radius > 0:
                    self.drift_detected = True # 确认漂移!

        # --- 假警报处理 ---
        elif self.in_warning_phase:
            # DDM 信号已清除，但 MMD 未确认漂移
            self.DDM.reset()
            self._reclean_streaming_state()