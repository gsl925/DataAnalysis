import pandas as pd
import numpy as np
import glob
from scipy.stats import f_oneway, ks_2samp, variation
from scipy.special import rel_entr
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Microsoft JhengHei'

file_list = sorted(glob.glob('D:\_Document\Others\_FlawDetection\ScrewData\Training\RawData\week*.csv'))
print("找到的CSV檔案", file_list)

if not file_list:
    raise FileNotFoundError("找不到任何符合 week*.csv 格式的檔案，請確認檔名與路徑！")

dfs = [pd.read_csv(f, encoding='big5') for f in file_list]
week_names = [f.split('.')[0] for f in file_list]
columns = dfs[0].columns.tolist()
print('分析欄位:', columns)

print('\n=== 同周期內穩定性CV & ANOVA ===')
for col in columns:
    cvs = []
    print(f'\n【欄位: {col}】')
    for i, df in enumerate(dfs):
        data = df[col].dropna()
        cv = variation(data)
        cvs.append(cv)
        print(f'{week_names[i]}: CV={cv:.4f}')
    # ANOVA 檢查
    all_data = [df[col].dropna() for df in dfs]
    if len(all_data) >= 2:
        fval, pval = f_oneway(*all_data)
        print(f'ANOVA: F={fval:.4f}, p={pval:.4f}')
    else:
        print('ANOVA: 只有一週資料，無法計算')

print('\n=== 跨週期穩定性KS test & KL Divergence ===')
for col in columns:
    print(f'\n【欄位: {col}】')
    for i in range(len(dfs)-1):
        data1 = dfs[i][col].dropna()
        data2 = dfs[i+1][col].dropna()
        # KS test
        ks_stat, ks_p = ks_2samp(data1, data2)
        # KL Divergence（用直方圖估計分布）
        min_bin = min(data1.min(), data2.min())
        max_bin = max(data1.max(), data2.max())
        hist1, bin_edges = np.histogram(data1, bins=10, range=(min_bin, max_bin), density=True)
        hist2, _ = np.histogram(data2, bins=bin_edges, density=True)
        hist1 += 1e-8  # 防止0
        hist2 += 1e-8
        kl_div = np.sum(rel_entr(hist1, hist2))
        print(f'{week_names[i]} vs {week_names[i+1]}: KS={ks_stat:.4f}, KS_p={ks_p:.4f}, KL={kl_div:.4f}')
for col in columns:
    plt.figure(figsize=(8,4))
    for i, df in enumerate(dfs):
        sns.kdeplot(df[col], label=week_names[i])
    plt.title(f'{col} 各週分布比較')
    plt.legend()
    plt.tight_layout()
    plt.show()
