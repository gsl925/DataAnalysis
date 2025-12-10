import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
import shap
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


# 設定中文字體（支援 Windows 預設字體）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 避免負號顯示錯誤


# === 載入資料 ===
# 請修改為你的實際檔案路徑
file_path = r'D:\_Document\Others\_FlawDetection\ScrewData\Training\RawData\519_603.csv'
df = pd.read_csv(file_path, encoding='big5')

#feature_columns = ['head_len', 'head_thick', 'thread_len', 'thread_dia', 'slot_depth', 'pitch_dia', 'nylok_len', 'other_pitch_dia', 'hardness']
#target_column = 'torque'
feature_columns = ['頭徑', '頭厚', '牙徑', '牙長', '針深', '槽寬', 'NYLOK', '小槽寬', '硬度']
target_column = '扭力'

X = df[feature_columns]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=42)

# === 建立強化 XGBoost 模型 ===
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# === 預測與評估 ===
y_pred = model.predict(X_test)

print("XGBoost 評估結果:")
print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE : {root_mean_squared_error(y_test, y_pred):.4f}")
print(f"R²   : {r2_score(y_test, y_pred):.4f}")
# === 建立 SHAP explainer ===
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# === 總體特徵貢獻圖（全樣本）===
shap.summary_plot(shap_values, X_test, plot_type="bar")

# === 更詳細的分佈圖（每個樣本分布）===
shap.summary_plot(shap_values, X_test)

# === 分析單一樣本（第一筆） ===
#shap.plots.waterfall(shap_values[0])