import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

import matplotlib.pyplot as plt
# 設定 matplotlib 中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# === 1. 載入資料 ===
def load_weekly_data(data_dir='./', encoding='big5'):
    csv_paths = glob.glob(os.path.join(data_dir, "week*.csv"))
    files_dict = {f"W{os.path.splitext(os.path.basename(p))[0][4:]}": p for p in csv_paths}
    df_all = pd.concat([
        pd.read_csv(path, encoding=encoding).assign(週期=week_label)
        for week_label, path in files_dict.items()
    ], ignore_index=True)
    return df_all

# === 2. 每週內 CV 統計 ===
def analyze_within_week_stability(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    results = []
    for week, group in df.groupby("週期"):
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors='coerce').dropna()
            if len(values) > 1:
                mean = values.mean()
                std = values.std()
                cv = std / mean if mean != 0 else np.nan
                results.append({"週期": week, "欄位": col, "平均": mean, "標準差": std, "CV": cv})
    return pd.DataFrame(results)

# === 3. ANOVA 分析 ===
def run_anova_all_numeric(df):
    results = {}
    df = df.copy()
    for col in df.columns:
        if col != "週期":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    for col in numeric_cols:
        if col == "週期":
            continue
        df_col = df[["週期", col]].dropna()
        if df_col["週期"].nunique() < 2:
            continue
        try:
            model = smf.ols(f"{col} ~ C(週期)", data=df_col).fit()
            anova_table = anova_lm(model, typ=2)
            results[col] = anova_table
            # 儲存為 CSV
            anova_table.to_csv(f"{output_dir}/anova_{col}.csv", encoding='utf_8_sig')
        except Exception as e:
            print(f"ANOVA 錯誤 ({col}): {e}")
    return results

# === 4. KS 與 KL 分析 ===
def ks_kl_analysis_all_features(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    weeks = sorted(df["週期"].unique())
    ks_all = []
    kl_all = []

    for col in numeric_cols:
        for i in range(len(weeks) - 1):
            w1 = pd.to_numeric(df[df["週期"] == weeks[i]][col], errors='coerce').dropna()
            w2 = pd.to_numeric(df[df["週期"] == weeks[i+1]][col], errors='coerce').dropna()
            if len(w1) < 2 or len(w2) < 2:
                continue
            ks_stat, ks_p = stats.ks_2samp(w1, w2)
            ks_all.append({"欄位": col, "週期1": weeks[i], "週期2": weeks[i+1], "KS_p值": ks_p})

            # KL divergence（加上平滑）
            bins = np.linspace(min(w1.min(), w2.min()), max(w1.max(), w2.max()), 20)
            hist1, _ = np.histogram(w1, bins=bins, density=True)
            hist2, _ = np.histogram(w2, bins=bins, density=True)
            kl = stats.entropy(hist1 + 1e-10, hist2 + 1e-10)
            kl_all.append({"欄位": col, "週期1": weeks[i], "週期2": weeks[i+1], "KL_divergence": kl})

    # 儲存成 CSV
    pd.DataFrame(ks_all).to_csv(f"{output_dir}/ks_test_summary.csv", index=False, encoding='utf_8_sig')
    pd.DataFrame(kl_all).to_csv(f"{output_dir}/kl_divergence_summary.csv", index=False, encoding='utf_8_sig')

    return pd.DataFrame(ks_all), pd.DataFrame(kl_all)

# === 5. 控制圖儲存 ===
def plot_sigma_trends(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        agg = df.groupby("週期")[col].agg(["mean", "std"]).dropna()
        agg["upper"] = agg["mean"] + 3 * agg["std"]
        agg["lower"] = agg["mean"] - 3 * agg["std"]

        plt.figure(figsize=(8, 5))
        plt.plot(agg.index, agg["mean"], marker='o', label=f"{col} 平均")
        plt.fill_between(agg.index, agg["lower"], agg["upper"], alpha=0.3, label="±3σ")
        plt.title(f"{col} 每週 ±3σ 控制圖")
        plt.xlabel("週期")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"sigma_plot_{col}.png")
        plt.savefig(save_path)
        plt.close()

# === 主流程 ===
if __name__ == "__main__":
    df_all = load_weekly_data(data_dir=r"D:\_Document\Others\_FlawDetection\ScrewData\Training\RawData")  # 指定資料夾路徑

    # 每週內 CV
    stability_df = analyze_within_week_stability(df_all)
    stability_df.to_csv(f"{output_dir}/stability_summary.csv", index=False, encoding='utf_8_sig')
    print("✅ 每週 CV 統計已儲存")

    # ANOVA
    anova_results = run_anova_all_numeric(df_all)
    print("✅ ANOVA 統計已完成並儲存")

    # KS / KL
    ks_df, kl_df = ks_kl_analysis_all_features(df_all)
    print("✅ KS / KL 統計已儲存")

    # 控制圖
    plot_sigma_trends(df_all)
    print("✅ 所有 ±3σ 控制圖已儲存至 output/")
