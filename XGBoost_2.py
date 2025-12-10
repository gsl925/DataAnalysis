import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import chardet
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 設定 matplotlib 中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ========= 使用者設定區 =========
file_path = r'D:\_Document\Others\_FlawDetection\ScrewData\Training\RawData\603_611_618_630_709.csv'  # ← 修改為你的 CSV 檔案
target_column = '破壞扭力'
use_all_data = False  # True: 全資料訓練與測試；False: 8:2 切分
# ==============================

# 自動偵測檔案編碼
#with open(file_path, 'rb') as f:
#    encoding = chardet.detect(f.read())['encoding']

df = pd.read_csv(file_path, encoding='big5')

# 原始特徵
feature_columns = ['頭徑', '頭厚', '牙徑', '牙長', '針深', '槽寬', 'NYLOK', '小槽寬', '硬度']
# feature_columns = ['頭徑', '頭厚', '牙徑', '牙長', '針深', '槽寬', 'NYLOK', '小槽寬', '硬度', 'T_shear']

# 加入交叉特徵
#df['頭徑x牙徑'] = df['頭徑'] * df['牙徑']
#df['硬度平方'] = df['硬度'] ** 2
#df['NYLOKx牙長'] = df['NYLOK'] * df['牙長']
#feature_columns += ['頭徑x牙徑', '硬度平方', 'NYLOKx牙長']

# 原始特徵（僅選擇部分特徵）
# selected_feature_columns = ['硬度']  # ← 修改為你需要的特徵

X = df[feature_columns]
# X = df[selected_feature_columns]
y = df[target_column]

# 資料切分
if use_all_data:
    X_train, X_test, y_train, y_test = X, X, y, y
    print("⚠️ 使用全部資料做訓練與測試（評估結果可能偏高）")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("✅ 使用 80/20 訓練/測試分割")

# 訓練模型
#model = XGBRegressor(random_state=42)
# === 建立強化 XGBoost 模型 ===
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 XGBoost 評估結果：")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# 預測 vs 實際圖
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, c='blue', alpha=0.6, label='預測點')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='理想線')
plt.xlabel('實際值')
plt.ylabel('預測值')
plt.title('預測 vs 實際')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 殘差圖
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('預測值')
plt.ylabel('殘差 (實際 - 預測)')
plt.title('殘差圖')
plt.grid(True)
plt.tight_layout()
plt.show()

# SHAP 分析
print("📈 SHAP 分析特徵重要性...")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar')
