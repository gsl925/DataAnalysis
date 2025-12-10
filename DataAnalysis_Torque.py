import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf

# 設定 matplotlib 中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
# === Step 1: 自動讀取所有 weekxxx.csv，保留中文欄位名稱 ===

def load_weekly_data(data_dir='./', encoding='big5'):
    csv_paths = glob.glob(os.path.join(data_dir, "week*.csv"))
    files_dict = {f"W{os.path.splitext(os.path.basename(p))[0][4:]}": p for p in csv_paths}

    df_all = pd.concat([
        pd.read_csv(path, encoding=encoding).assign(週期=week_label)
        for week_label, path in files_dict.items()
    ], ignore_index=True)

    return df_all

# === Step 2: 每週內穩定性分析（平均 / 標準差 / CV） ===

def analyze_within_week_stability(df):
    results = []
    for week, group in df.groupby("週期"):
        mean = group["破壞扭力"].mean()
        std = group["破壞扭力"].std()
        cv = std / mean
        results.append({"週期": week, "平均": mean, "標準差": std, "CV": cv})
    return pd.DataFrame(results)

# === Step 3: 跨週期 ANOVA 分析 ===

def run_anova(df):
    # 確保破壞扭力是數值型態
    df = df.copy()
    df["破壞扭力"] = pd.to_numeric(df["破壞扭力"], errors="coerce")

    # 移除空值
    df = df.dropna(subset=["破壞扭力", "週期"])

    # 使用公式語法建立 OLS 模型
    model = smf.ols(formula="破壞扭力 ~ C(週期)", data=df).fit()

    # 回傳 ANOVA 分析結果
    return anova_lm(model, typ=2)



# === Step 4: KS 檢定 & KL divergence（相鄰週） ===

def ks_kl_analysis(df):
    weeks = sorted(df["週期"].unique())
    ks_list = []
    kl_list = []
    for i in range(len(weeks) - 1):
        w1 = df[df["週期"] == weeks[i]]["破壞扭力"]
        w2 = df[df["週期"] == weeks[i+1]]["破壞扭力"]

        ks_stat, ks_p = stats.ks_2samp(w1, w2)
        ks_list.append({"週期1": weeks[i], "週期2": weeks[i+1], "KS_p值": ks_p})

        hist1, _ = np.histogram(w1, bins=20, range=(df["破壞扭力"].min(), df["破壞扭力"].max()), density=True)
        hist2, _ = np.histogram(w2, bins=20, range=(df["破壞扭力"].min(), df["破壞扭力"].max()), density=True)
        kl = stats.entropy(hist1 + 1e-10, hist2 + 1e-10)
        kl_list.append({"週期1": weeks[i], "週期2": weeks[i+1], "KL_divergence": kl})

    return pd.DataFrame(ks_list), pd.DataFrame(kl_list)

# === Step 5: ±3σ 趨勢圖 ===

def plot_weekly_mean_sigma(df):
    agg = df.groupby("週期")["破壞扭力"].agg(["mean", "std"])
    agg["upper"] = agg["mean"] + 3 * agg["std"]
    agg["lower"] = agg["mean"] - 3 * agg["std"]

    plt.figure(figsize=(8, 5))
    plt.plot(agg.index, agg["mean"], marker='o', label="平均")
    plt.fill_between(agg.index, agg["lower"], agg["upper"], alpha=0.3, label="±3σ")
    plt.title("每週破壞扭力平均 ±3σ")
    plt.xlabel("週期")
    plt.ylabel("破壞扭力")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Step 6 (選用): SHAP 趨勢圖（若你有 shap_df） ===

def plot_shap_trend(shap_df):
    trend = shap_df.groupby("週期").mean().T
    trend.plot(kind='bar', figsize=(12, 6), legend=True)
    plt.title("SHAP 各特徵平均貢獻值變化")
    plt.ylabel("平均 |SHAP 值|")
    plt.xlabel("特徵")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Main 主流程 ===

if __name__ == "__main__":
    df_all = load_weekly_data(data_dir=r"D:\_Document\Others\_FlawDetection\ScrewData\Training\RawData")  # 指定資料夾路徑

    # 每週內變異性
    cv_df = analyze_within_week_stability(df_all)
    print("各週 CV 分析：")
    print(cv_df)

    # ANOVA
    anova_result = run_anova(df_all)
    print("\nANOVA 統計結果：")
    print(anova_result)

    # KS/KL
    ks_df, kl_df = ks_kl_analysis(df_all)
    print("\nKS 檢定結果：")
    print(ks_df)
    print("\nKL divergence：")
    print(kl_df)

    # ±3σ 趨勢圖
    plot_weekly_mean_sigma(df_all)

    # 若有 SHAP 值：shap_df = pd.read_csv(...) 並畫圖
    # plot_shap_trend(shap_df)
