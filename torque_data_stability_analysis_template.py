# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm

# ========== Step 1: 資料合併與欄位標準化 ==========

def load_and_merge(files_dict, encoding='big5'):
    df_all = pd.concat([pd.read_csv(path, encoding=encoding).assign(week=week_label)
                        for week_label, path in files_dict.items()],
                       ignore_index=True)
    df_all.columns = ['頭徑', '頭厚', '牙徑', '牙長', '針深', '槽寬', 'NYLOK', '小槽寬', '硬度', '破壞扭力', 'week']
    return df_all

# ========== Step 2: 同週期穩定性分析 ==========

def analyze_within_week_stability(df):
    results = []
    for week, group in df.groupby("week"):
        mean_torque = group["torque"].mean()
        std_torque = group["torque"].std()
        cv = std_torque / mean_torque
        results.append({"week": week, "mean": mean_torque, "std": std_torque, "cv": cv})
    return pd.DataFrame(results)

def run_anova(df):
    model = sm.OLS(df["torque"], sm.add_constant(pd.get_dummies(df["week"], drop_first=True)))
    return anova_lm(model.fit(), typ=2)

# ========== Step 3: 跨週期穩定性分析 ==========

def ks_kl_analysis(df):
    weeks = sorted(df["week"].unique())
    ks_results = []
    kl_results = []
    for i in range(len(weeks) - 1):
        w1 = df[df["week"] == weeks[i]]["torque"]
        w2 = df[df["week"] == weeks[i+1]]["torque"]
        ks_stat, ks_p = stats.ks_2samp(w1, w2)
        ks_results.append({"from": weeks[i], "to": weeks[i+1], "ks_p": ks_p})

        hist1, _ = np.histogram(w1, bins=20, range=(df["torque"].min(), df["torque"].max()), density=True)
        hist2, _ = np.histogram(w2, bins=20, range=(df["torque"].min(), df["torque"].max()), density=True)
        kl_div = stats.entropy(hist1 + 1e-10, hist2 + 1e-10)
        kl_results.append({"from": weeks[i], "to": weeks[i+1], "kl_div": kl_div})
    return pd.DataFrame(ks_results), pd.DataFrame(kl_results)

# ========== Step 4: 每週 ±3σ 可視化 ==========

def plot_weekly_mean_sigma(df):
    agg = df.groupby("week")["torque"].agg(["mean", "std"])
    agg["upper"] = agg["mean"] + 3 * agg["std"]
    agg["lower"] = agg["mean"] - 3 * agg["std"]
    plt.figure(figsize=(8, 5))
    plt.plot(agg.index, agg["mean"], marker='o', label='Mean')
    plt.fill_between(agg.index, agg["lower"], agg["upper"], alpha=0.3, label='±3σ')
    plt.title("Torque Mean ± 3σ per Week")
    plt.xlabel("Week")
    plt.ylabel("Torque")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ========== Step 5: SHAP 解釋變異趨勢（選用） ==========

def plot_shap_trend(shap_values_df):
    """
    輸入格式：shap_values_df 應包含欄位 ['week', 'feature1', 'feature2', ...]
    每筆為一筆樣本對應的 SHAP 值，會先 groupby 週期再畫出趨勢
    """
    trend = shap_values_df.groupby("week").mean().T
    trend.plot(kind='bar', figsize=(12, 6), legend=True)
    plt.title("SHAP Feature Importance Trend by Week")
    plt.ylabel("Mean |SHAP Value|")
    plt.xlabel("Features")
    plt.tight_layout()
    plt.grid(True)
    plt.show()


# ========== 主程式 ==========
files = {
    "W1": "week603.csv",
    "W2": "week611.csv",
    "W3": "week618.csv"
}
df_all = load_and_merge(files)

cv_df = analyze_within_week_stability(df_all)
anova_df = run_anova(df_all)
ks_df, kl_df = ks_kl_analysis(df_all)
plot_weekly_mean_sigma(df_all)
# 如果有 SHAP 值的 DataFrame，則可以呼叫 plot_shap_trend
# plot_shap_trend(shap_values_df)
# ========== 結果輸出 ==========
print("Within Week Stability (CV):")
print(cv_df)
print("\nANOVA Results:")
print(anova_df)
print("\nKS Test Results:")
print(ks_df)
print("\nKL Divergence Results:")
print(kl_df)
# -*- coding: utf-8 -*-