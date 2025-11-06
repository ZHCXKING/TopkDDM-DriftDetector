import numpy as np
import matplotlib.pyplot as plt
import random

class RankToErrorConverter:
    """
    使用平滑的 Sigmoid 函数将排名转换为一个二元误差信号。
    """
    def __init__(self, k: int, steepness: float = 1.0):
        self.k = k
        self.steepness = steepness
        self.center_point = (k + 1) / 2

    def _get_correctness_prob(self, rank: int) -> float:
        exponent = self.steepness * (rank - self.center_point)
        return 1 / (1 + np.exp(exponent))

    def convert(self, rank: int | None) -> float:
        if rank is None or rank > self.k:
            error_float = 1.0
        else:
            correctness_prob = self._get_correctness_prob(rank)
            error_float = 1.0 - correctness_prob

        if random.random() < error_float:
            return 1.0
        else:
            return 0.0

    def plot_curve(self, savepath="figure/Sigmoid.pdf"):
        """基础曲线绘制"""
        ranks = np.arange(1, self.k + 1)
        error_probs = 1 - np.array([self._get_correctness_prob(r) for r in ranks])

        plt.figure(figsize=(8, 5))
        plt.plot(ranks, error_probs, 'b-o', label=f'Sigmoid (k={self.k})')

        plt.title('Rank to Error Probability Mapping')
        plt.xlabel('Rank in Top-K List')
        plt.ylabel('Probability of being considered an "Error"')
        plt.xticks(ranks)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(savepath, bbox_inches='tight')
        plt.show()

    def plot_curve_with_xi(self, rank_example: int = 4, savepath="figure/Sigmoid_with_xi.pdf"):
        """在曲线上直观体现 xi 的作用"""
        ranks = np.arange(1, self.k + 1)
        error_probs = 1 - np.array([self._get_correctness_prob(r) for r in ranks])

        # 随机采样一个 xi
        xi = random.random()

        # rank_example 的误差概率
        p_err = error_probs[rank_example - 1]

        plt.figure(figsize=(8, 5))
        plt.plot(ranks, error_probs, 'b-o', label=f'Sigmoid (k={self.k})')

        # 水平线 y = xi
        plt.axhline(y=xi, color='r', linestyle='--', label=f"$\\xi={xi:.2f}$")

        # rank_example 的点
        plt.scatter(rank_example, p_err, color='green', s=100, zorder=5,
                    label=f"Rank={rank_example}, p_err={p_err:.2f}")

        # 误差判定结果
        result = "Error=1" if xi < p_err else "Error=0"
        plt.text(rank_example+0.2, p_err, result, color="green", fontsize=12)

        plt.title('Rank to Error Probability Mapping with Random $\\xi$')
        plt.xlabel('Rank in Top-K List')
        plt.ylabel('Probability of being considered an "Error"')
        plt.xticks(ranks)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(savepath, bbox_inches='tight')
        plt.show()


# --- 使用示例 ---
k = 10
converter = RankToErrorConverter(k=k, steepness=1.0)

# 原始曲线
converter.plot_curve()

# 带 xi 的直观解释图
# converter.plot_curve_with_xi(rank_example=4)
