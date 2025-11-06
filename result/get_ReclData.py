# %%
from utils.framework import Framework
import matplotlib.pyplot as plt
import warnings
# %%
def draw_hr_ndcg(results, metrice, k, start_point, model):
    plt.rcParams['font.family'] = 'serif'  # 优先使用衬线字体
    plt.rcParams['font.serif'] = ['Times New Roman', 'Songti SC']  # 设置衬线字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    plt.rcParams['font.size'] = 10  # 全局字体大小
    plt.rcParams['axes.labelsize'] = 10  # 坐标轴标签字体大小
    plt.rcParams['xtick.labelsize'] = 9  # x轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = 9  # y轴刻度字体大小
    plt.rcParams['legend.fontsize'] = 9  # 图例字体大小
    plt.rcParams['figure.dpi'] = 300  # 图像分辨率
    # figsize的单位是英寸，(6.4, 4.8)是默认值，可根据期刊要求调整
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    styles = [
        {'color': 'C0'},  # 蓝色, 实线, 圆点
        {'color': 'C1'},  # 橙色, 虚线, 方块
        {'color': 'C2'},  # 绿色, 点线, 三角
        {'color': 'C3'},  # 红色, 点划线, 菱形
        {'color': 'C4'},  # 紫色, 实线, 倒三角
        {'color': 'C5'},  # 棕色, 虚线, 五边形
        {'color': 'C6'},  # 粉色, 点线, 星号
        {'color': 'C8'} # 灰色, 点划线, X标记
    ]
    x_label = 'Number of samples from test'
    y_label = metrice + '@' + str(k)
    title = model
    path = 'figure/' + model + '/' + model + '-' + y_label + '.pdf'
    for i, data in enumerate(results):
        style = styles[i % len(styles)]
        y = data[f'{metrice}_list'][start_point:]
        x = list(range(start_point, start_point + len(y)))
        label = data['drifter']
        ax.plot(x, y, label=label, **style)
    drift_point = 5000
    ax.axvline(x=drift_point, color='black', linestyle='--', linewidth=1, alpha=0.8, label='Drift Event')
    # 设置坐标轴标签 (包含单位)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # 设置坐标轴范围
    ax.set_xlim(start_point, control['test_size'])
    #ax.set_ylim(0, ylim)
    # 显示图例 (loc='best' 会自动寻找最佳位置)
    ax.legend(loc='upper center',
              bbox_to_anchor = (0.5,-0.2),
              ncol = 5,
              fancybox = True)
    # 添加网格线 (alpha是透明度)
    ax.grid(True, linestyle='--', alpha=0.6)
    # 控制刻度的方向和样式
    ax.tick_params(direction='in', top=True, right=True)
    # 自动调整布局，防止标签重叠
    plt.tight_layout()
    # 保存为pdf, bbox_inches='tight' 可以裁剪掉多余的白边
    # plt.savefig(path, bbox_inches='tight')
    # 显示图形 (在脚本最后调用)
    plt.show()
# %%
def draw(drifters, control, start_point, model):
    results = []
    for drifter in drifters:
        control['drifter'] = drifter
        work = Framework(**control)
        HR_list, NDCG_list, refits, cpu_elapsed = work.start_recdata(model, refit_times)
        results.append({
            'HR_list': HR_list,
            'NDCG_list': NDCG_list,
            'refits': refits,
            'drifter': drifter})
    draw_hr_ndcg(results, 'HR', control['k'], start_point, model)
    # draw_hr_ndcg(results, 'NDCG', control['k'], start_point, model)
# %%
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    model = 'BPR'  # BPR, BiVAECF, HPF, MF, SVD
    drifters = ['Topk-DDM', 'DDM', 'MWDDM-H', 'MWDDM-M', 'VFDDM-H', 'VFDDM-M', 'VFDDM-K', 'EDDM']
    start_point = 1000
    refit_times = 10
    # %%
    control = {
        'path': 'RecData',
        'synth_control': None,
        'RecData_control': None,
        'k': 10,
        'train_size': 1000,
        'test_size': 20000,
        'seed': 42,
        'model': 'NB',
        'drifter': 'DDM'}
    control['RecData_control'] = {
        'n_users': 100,
        'n_items': 50,
        'n_features': 5}
    for i in [6]:
        control['seed'] = i
        draw(drifters, control, start_point, model)