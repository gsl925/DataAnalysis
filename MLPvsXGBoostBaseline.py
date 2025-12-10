import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import xgboost as xgb

# === 步驟 1：載入資料 ===
# 請修改為你的實際檔案路徑
file_path = 'D:\_Document\Others\FlawDetection\ScrewData\Training\603.csv'
df = pd.read_csv(file_path)

# === 步驟 2：選擇欄位 ===
# 請根據你實際的欄位名稱修改
feature_columns = [
    'head_len', 'head_thick', 'thread_len', 'thread_dia', 'slot_depth', 
    'pitch_dia', 'nylok_len', 'other_pitch_dia' 'hardness'
]
target_column = 'torque'  # 扭力

X = df[feature_columns]
y = df[target_column]

# === 步驟 3：分訓練與測試資料 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 步驟 4：建立與訓練模型 ===
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

xgbr = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
xgbr.fit(X_train, y_train)
y_pred_xgb = xgbr.predict(X_test)

# === 步驟 5：評估模型 ===
def print_scores(y_true, y_pred, name):
    print(f"\n{name} 評估結果:")
    print(f"MAE  : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE : {root_mean_squared_error(y_true, y_pred):.4f}")
    print(f"R²   : {r2_score(y_true, y_pred):.4f}")

print_scores(y_test, y_pred_mlp, "MLP")
print_scores(y_test, y_pred_xgb, "XGBoost")
