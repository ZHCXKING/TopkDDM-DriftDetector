# %%
import math
from collections import deque
import numpy as np
# %%


class BDDM:
    def __init__(self, 
                 window_size: int = 30, 
                 d_weight: float = 0.01, 
                 delta: float = 0.25,
                 alpha: float | None = None):
        self.w = window_size
        self.d = d_weight
        self.delta = delta
        self.weights = np.array([1 + i * self.d for i in range(self.w)])
        self.sum_weights = np.sum(self.weights)

        # 如果用户没有提供手动阈值，则根据公式计算
        if alpha is None:
            self.alpha = -math.log((1 + self.d) / 2.0) * self.w / self.sum_weights
        else:
            # 否则，使用用户提供的阈值
            self.alpha = alpha
            
        self.window = deque(maxlen=self.w)
        self._reset()

    def _reset(self):
        """Resets the detector's state after a drift is detected."""
        self.instance_count = 0
        self.window.clear()
        self.p = 0.0
        self.s = 0.0
        self.p_max = 0.0
        self.s_max = 0.0
        self.drift_detected = False

    def update(self, prediction_is_error: int):
        if self.drift_detected:
            self._reset()
        self.instance_count += 1
        self.window.append(prediction_is_error)
        if self.instance_count < self.w:
            return False
        window_array = np.array(self.window)
        self.p = np.sum(self.weights * window_array) / self.sum_weights
        variance = np.sum(self.weights * (window_array - self.p) ** 2) / self.sum_weights
        self.s = math.sqrt(variance)

        if self.p >= self.p_max:
            self.p_max = self.p
            self.s_max = self.s
            return False

        if self.p_max - self.p >= self.p_max * self.delta:
            epsilon = 1e-9
            s_sq = self.s ** 2 + epsilon
            s_max_sq = self.s_max ** 2 + epsilon
            
            term1_inner = (s_max_sq / s_sq) + (s_sq / s_max_sq) + 2
            term1 = 0.25 * math.log(0.25 * term1_inner)
            
            term2_numerator = (self.p_max - self.p) ** 2
            term2_denominator = s_max_sq + s_sq
            term2 = 0.25 * (term2_numerator / term2_denominator)
            
            b_distance = term1 + term2
            
            if b_distance >= self.alpha:
                self.drift_detected = True
                return True
        return False
# %%


class MWDDM:
    """
    Python implementation of the Multi-level Weighted Drift Detection Method (MWDDM)
    from the paper: "A multi-level weighted concept drift detection method" by Chen et al. (2023).
    """

    def __init__(self,
                 long_window_size: int = 500,
                 short_window_size: int = 100,
                 delta: float = 0.01,
                 theta_s: float = 0.78,
                 theta_l: float = 0.85,
                 diff_stable: float = 0.01,
                 diff_warning: float = 5.0,
                 mode: str = 'H'):
        """
        Initializes the drift detector.
        Args:
            long_window_size (int): The size of the long sliding window (n_l).
            short_window_size (int): The size of the short sliding window (n_s).
            delta (float): The confidence level for the statistical bounds.
            theta_s (float): The level transition threshold for the short window (θ_s).
            theta_l (float): The level transition threshold for the long window (θ_l).
            diff_stable (float): The weight difference factor for the 'stable' level.
            diff_warning (float): The weight difference factor for the 'warning' level.
            mode (str): The statistical bound to use. 'H' for Hoeffding, 'M' for McDiarmid.
        """
        if short_window_size >= long_window_size:
            raise ValueError(
                "short_window_size must be smaller than long_window_size.")
        self.n_l = long_window_size
        self.n_s = short_window_size
        self.delta = delta
        self.theta_s = theta_s
        self.theta_l = theta_l
        self.diff_stable = diff_stable
        self.diff_warning = diff_warning
        if mode.upper() not in ['H', 'M']:
            raise ValueError(
                "Mode must be 'H' (Hoeffding) or 'M' (McDiarmid).")
        self.mode = mode.upper()
        self.long_window = deque(maxlen=self.n_l)
        self.short_window = deque(maxlen=self.n_s)
        # Pre-calculate weights to optimize performance
        self.weights_stable_s = self._calculate_weights(
            self.n_s, self.diff_stable)
        self.weights_stable_l = self._calculate_weights(
            self.n_l, self.diff_stable)
        self.weights_warning_s = self._calculate_weights(
            self.n_s, self.diff_warning)
        self.weights_warning_l = self._calculate_weights(
            self.n_l, self.diff_warning)
        self._reset()

    def _reset(self):
        """Resets the detector's state, typically after a drift is detected."""
        self.level = 'stable'
        self.long_window.clear()
        self.short_window.clear()
        # Max weighted accuracies for the stable level
        self.u_max_s = 0.0
        self.u_max_l = 0.0
        # Max weighted accuracies for the warning level
        self.u_max_s_prime = 0.0
        self.u_max_l_prime = 0.0
        self.drift_detected = False

    def _calculate_weights(self, size: int, diff: float) -> np.ndarray:
        """Calculates the linear weights for a window."""
        return 1 + np.arange(size) * diff

    def _calculate_weighted_accuracy(self, window: deque, weights: np.ndarray) -> float:
        """Calculates the weighted average of correct predictions (u.ω)."""
        if not window:
            return 0.0
        window_arr = np.array(list(window))
        weighted_sum = np.sum(window_arr * weights)
        total_weight = np.sum(weights)
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_hoeffding_bound(self, window_size: int) -> float:
        """Calculates the Hoeffding bound (ε_H)."""
        return math.sqrt(math.log(1 / self.delta) / (2 * window_size))

    def _calculate_mcdiarmid_bound(self, weights: np.ndarray) -> float:
        """Calculates the McDiarmid bound (ε_M)."""
        total_weight = np.sum(weights)
        if total_weight == 0:
            return float('inf')
        # Normalized weights v_i = w_i / sum(w)
        v = weights / total_weight
        # Sum of squared normalized weights: sum(v_i^2)
        sum_v_sq = np.sum(v**2)
        return math.sqrt((sum_v_sq * math.log(1 / self.delta)) / 2)

    def update(self, prediction_is_correct: bool) -> bool:
        """
        Adds a new prediction result to the detector and checks for drift.

        Args:
            prediction_is_correct (bool): True if the prediction was correct, False otherwise.

        Returns:
            bool: True if a concept drift was detected, False otherwise.
        """
        if self.drift_detected:
            self._reset()
        val = 1 if prediction_is_correct else 0
        self.long_window.append(val)
        self.short_window.append(val)
        if len(self.long_window) < self.n_l:
            return False
        if self.level == 'stable':
            u_s_omega = self._calculate_weighted_accuracy(
                self.short_window, self.weights_stable_s)
            u_l_omega = self._calculate_weighted_accuracy(
                self.long_window, self.weights_stable_l)
            if u_s_omega > self.u_max_s:
                self.u_max_s = u_s_omega
            if u_l_omega > self.u_max_l:
                self.u_max_l = u_l_omega
            
            lambda_s = u_s_omega / self.u_max_s if self.u_max_s > 0 else 1.0
            lambda_l = u_l_omega / self.u_max_l if self.u_max_l > 0 else 1.0
            
            if lambda_s < self.theta_s or lambda_l < self.theta_l:
                self.level = 'warning'
                self.u_max_s_prime = u_s_omega
                self.u_max_l_prime = u_l_omega
        if self.level == 'warning':
            u_s_omega_prime = self._calculate_weighted_accuracy(
                self.short_window, self.weights_warning_s)
            u_l_omega_prime = self._calculate_weighted_accuracy(
                self.long_window, self.weights_warning_l)
            
            if u_s_omega_prime > self.u_max_s_prime:
                self.u_max_s_prime = u_s_omega_prime
            if u_l_omega_prime > self.u_max_l_prime:
                self.u_max_l_prime = u_l_omega_prime
                
            delta_s = self.u_max_s_prime - u_s_omega_prime
            delta_l = self.u_max_l_prime - u_l_omega_prime
            
            if self.mode == 'H':
                epsilon_s = self._calculate_hoeffding_bound(self.n_s)
                epsilon_l = self._calculate_hoeffding_bound(self.n_l)
            else:  # 'M'
                epsilon_s = self._calculate_mcdiarmid_bound(
                    self.weights_warning_s)
                epsilon_l = self._calculate_mcdiarmid_bound(
                    self.weights_warning_l)

            if delta_s > epsilon_s or delta_l > epsilon_l:
                self.drift_detected = True
                return True
        return False
# %%


class VFDDM:
    """
    Implementation of the Variance Feedback Drift Detection Method (VFDDM)
    as described in the paper: "Variance Feedback Drift Detection Method for
    Evolving Data Streams Mining" by Han et al. (Appl. Sci. 2024, 14, 7157).
    """

    def __init__(self, window_size: int = 50, delta: float = 10e-7, test_type: str = 'H',
                 vp_min_size: int = 30, beta: float = 1.0, diff_0: float = 0.01):
        """
        Args:
            window_size (int): The size of the sliding window (n).
            delta (float): The confidence level for statistical tests (δ).
            test_type (str): The statistical test to use for high-variance streams.
                             Must be one of 'H' (Hoeffding), 'M' (McDiarmid),
                             or 'K' (Kolmogorov).
            vp_min_size (int): The minimum number of samples required in the
                               variance pool before variance feedback is activated.
            beta (float): The mean influence factor for mean adjustment (Eq. 11).
            diff_0 (float): The initial weighting factor (used for McDiarmid's test).
        """
        if test_type not in ['H', 'M', 'K']:
            raise ValueError("test_type must be one of 'H', 'M', or 'K'")
        self.n = window_size
        self.delta = delta
        self.test_type = test_type
        self.vp_min_size = vp_min_size
        self.beta = beta
        self.diff_0 = diff_0
        # Internal state
        self.win = deque(maxlen=self.n)
        self.p = 0.0
        self.p_max = 0.0
        self.drift_detected = False
        # Variance Estimation state
        # Number of 1s (correct predictions) in the variance pool
        self.vp_N1 = 0
        self.vp_m = 0   # Total size of the variance pool
        self.sigma_sq = 0.0  # Estimated variance σ²
        # Calculate initial critical variance σ²_c (it's constant for a given n and delta)
        self.sigma_sq_c = self._calculate_critical_variance()

    def reset(self):
        """Resets the detector's state after a drift is detected."""
        self.win.clear()
        self.p = 0.0
        self.p_max = 0.0
        # As per Algorithm 1, reset variance to the critical threshold
        self.sigma_sq = self.sigma_sq_c
        # The paper doesn't specify resetting the variance pool, but it's logical
        # to clear it to learn the new concept's stability.
        self.vp_N1 = 0
        self.vp_m = 0

    def update(self, prediction: int) -> bool:
        """
        Adds a new prediction result to the detector and checks for drift.

        Args:
            prediction (int): The result of the prediction, 1 for correct, 0 for incorrect.

        Returns:
            bool: True if a drift is detected, False otherwise.
        """
        if self.drift_detected:
            self.drift_detected = False
            self.reset()
        self.win.append(prediction)
        if len(self.win) < self.n:
            return False
        # --- 1. Variance Estimation Strategy (Section 3.1) ---
        p_unweighted = sum(self.win) / self.n
        # Calculate minimum Bernstein bound (ε_B)_min for stability check (Eq. 6)
        epsilon_b_min = (2 * np.log(1 / self.delta)) / (3 * self.n)
        # Check for stability (Eq. 7) to decide whether to sample for variance
        # Note: We use p_unweighted here as the stability of the raw stream is being assessed.
        if p_unweighted > self.p_max - epsilon_b_min:
            # If stable, add the latest instance to the variance pool
            self.vp_m += 1
            if self.win[-1] == 1:
                self.vp_N1 += 1
            # Update estimated variance σ² using the efficient formula (Eq. 9)
            if self.vp_m > 1:
                N1, m = self.vp_N1, self.vp_m
                term1 = N1 * (1 - N1 / m)**2
                term2 = (m - N1) * (N1 / m)**2
                self.sigma_sq = (1 / (m + 1)) * (term1 + term2)
        # --- 2. Variance Feedback & Detection (Section 3.2) ---
        # Calculate adjusted mean 'p'
        if self.test_type == 'M':
            # For McDiarmid, weights are required. We assume linear weighting as in WMDDM.
            # This is an interpretation, as the paper is slightly ambiguous here.
            weights = np.array([self.diff_0 * i for i in range(1, self.n + 1)])
            p_weighted = np.sum(np.array(self.win) * weights) / np.sum(weights)
            self.p = p_weighted
        else:
            # Adjust mean based on variance for H and K tests (Eq. 11)
            p_adjusted = p_unweighted / \
                (1 - self.beta / (1 + np.exp(self.sigma_sq)))
            self.p = p_adjusted
        # Update historical maximum mean p_max (Eq. 22)
        self.p_max = max(self.p_max, self.p)
        # Select statistical test and calculate threshold ε
        # Activate variance-based selection only if variance pool is sufficiently large
        if self.sigma_sq <= self.sigma_sq_c and self.vp_m >= self.vp_min_size:
            # Low variance stream, use Bernstein Test
            epsilon = self._calculate_bernstein_bound()
        else:
            # High variance stream, use the specified test
            if self.test_type == 'H':
                epsilon = self._calculate_hoeffding_bound()
            elif self.test_type == 'M':
                epsilon = self._calculate_mcdiarmid_bound()
            else:  # 'K'
                epsilon = self._calculate_kolmogorov_bound()
        # --- 3. Drift Check ---
        if self.p_max - self.p > epsilon:
            self.drift_detected = True
            return True
        return False
    # --- Helper methods for calculating bounds and critical variance ---

    def _calculate_critical_variance(self):
        """Calculates the critical variance σ²_c for test selection (Eq. 13)."""
        epsilon_h = self._calculate_hoeffding_bound()
        # To avoid division by zero if epsilon_h is somehow 0
        if epsilon_h == 0:
            return 0.25
        return 0.25 - (1 / (3 * epsilon_h))

    def _calculate_bernstein_bound(self):
        """Calculates the Bernstein bound ε_B (Eq. 4)."""
        t = (1 / self.n) * np.log(1 / self.delta)
        # To avoid division by zero if t is 0
        if t == 0:
            return np.inf
        return (1 / t) + np.sqrt((1 / t)**2 + (2 * self.sigma_sq) / t)

    def _calculate_hoeffding_bound(self):
        """Calculates the Hoeffding bound ε_H (Eq. 12)."""
        return np.sqrt(np.log(1 / self.delta) / (2 * self.n))

    def _calculate_mcdiarmid_bound(self):
        """Calculates the McDiarmid bound ε_M (Eq. 18)."""
        # We assume linear weighting for c_i, as this test requires instance weights.
        weights = np.array([self.diff_0 * i for i in range(1, self.n + 1)])
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            return np.inf
        c_i = weights / sum_weights  # c_i = w_i / sum(w)
        sum_c_sq = np.sum(c_i**2)   # sum(c_i^2)
        return np.sqrt((sum_c_sq * np.log(1 / self.delta)) / 2)

    def _calculate_kolmogorov_bound(self):
        """Calculates the Kolmogorov bound ε_K using binary search (from Eq. 21)."""
        # We need to solve: ln(1/δ) = n * (2ε² + (4/3)ε³)
        # Let f(ε) = n * (2ε² + (4/3)ε³) - ln(1/δ)
        # We find ε such that f(ε) = 0.
        target = np.log(1 / self.delta)
        # Binary search for ε in [0, 1]
        low, high = 0.0, 1.0
        for _ in range(100):  # 100 iterations for precision
            mid = (low + high) / 2
            if mid == 0:  # Avoid division by zero
                low = 1e-9
                continue
            try:
                # Value of n * (2ε² + (4/3)ε³)
                val = self.n * (2 * mid**2 + (4/3) * mid**3)
            except OverflowError:
                val = float('inf')
            if val < target:
                low = mid
            else:
                high = mid
        return high
