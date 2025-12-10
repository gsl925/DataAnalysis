# train_tabtransformer_with_shap.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap
import os
import chardet
import joblib
import itertools

from tab_transformer_pytorch import TabTransformer

# ========== 常調參數 ==========
file_path = r'D:\_Document\Others\_FlawDetection\ScrewData\Training\RawData\611.csv'
target_column = '破壞扭力'
feature_columns = ['頭徑','頭厚','牙徑','牙長','針深','槽寬','NYLOK','小槽寬','硬度']
use_all_data_for_training = False
num_epochs = 500
batch_size = 64
learning_rate = 1e-3
data_split_size = 0.2  # 20% 測試集
model_save_path = 'tabtransformer_model.pth'
scaler_X_path = 'scaler_X.pkl'
scaler_y_path = 'scaler_y.pkl'
shap_save_path = 'shap_values.npy'
shap_sample_index = 0
max_shap_display = 15
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== matplotlib 中文支援 ==========
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = True

# ========== 自動檢測檔案編碼 ==========
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"檢測到的文件編碼: {encoding}")

# ========== 讀取與擴增特徵 ==========
df = pd.read_csv(file_path, encoding='big5')
df.columns = df.columns.str.strip()
df = df.dropna(subset=feature_columns + [target_column])

# # 加入指定交叉特徵
# cross_features = [('針深', '硬度'), ('NYLOK', '硬度')]
# for f1, f2 in cross_features:
#     new_col = f'{f1}x{f2}'
#     df[new_col] = df[f1] * df[f2]
#     feature_columns.append(new_col)

# # 自動產生其他交叉特徵（排除已加入者）
# existing_cross = set([f'{f1}x{f2}' for f1, f2 in cross_features])
# for f1, f2 in itertools.combinations(feature_columns, 2):
#     new_col = f'{f1}x{f2}'
#     if new_col not in existing_cross and new_col not in df.columns and f1 != f2:
#         df[new_col] = df[f1] * df[f2]
#         feature_columns.append(new_col)

# ========== 正規化 ==========
X = df[feature_columns].values.astype(np.float32)
y = df[target_column].values.astype(np.float32).reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

joblib.dump(scaler_X, scaler_X_path)
joblib.dump(scaler_y, scaler_y_path)
print(f"✅ 特徵與標籤已正規化並保存")

# ========== 切分資料 ==========
if use_all_data_for_training:
    X_train, y_train = X, y
    X_test, y_test = X[:50], y[:50]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data_split_size, random_state=42)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# ========== 建立模型 ==========
model = TabTransformer(
    categories=(),
    num_continuous=len(feature_columns),
    dim=64,
    depth=6,
    heads=16,
    dim_out=1,
    mlp_hidden_mults=(4, 2)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
loss_fn = torch.nn.MSELoss()

# ========== 訓練 ==========
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        x_categ = torch.empty((xb.shape[0], 0)).to(device)
        y_pred = model(x_categ, xb)
        loss = loss_fn(y_pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"Epoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")

# ========== 儲存模型 ==========
torch.save(model.state_dict(), model_save_path)
print(f"\n✅ 模型已儲存至 {model_save_path}")

# ========== 預測與反正規化 ==========
model.eval()
y_preds, y_trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        x_categ = torch.empty((xb.shape[0], 0)).to(device)
        y_pred = model(x_categ, xb)
        y_preds += y_pred.cpu().numpy().flatten().tolist()
        y_trues += yb.numpy().flatten().tolist()

y_preds = scaler_y.inverse_transform(np.array(y_preds).reshape(-1, 1)).flatten()
y_trues = scaler_y.inverse_transform(np.array(y_trues).reshape(-1, 1)).flatten()

# ========== 評估 ==========
mae = mean_absolute_error(y_trues, y_preds)
rmse = np.sqrt(mean_squared_error(y_trues, y_preds))
r2 = r2_score(y_trues, y_preds)
print(f"\n📊 TabTransformer 評估:")
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# ========== SHAP 分析 ==========
def model_predict(input_array):
    with torch.no_grad():
        input_tensor = torch.tensor(input_array, dtype=torch.float32).to(device)
        x_categ = torch.empty((input_tensor.shape[0], 0)).to(device)
        return model(x_categ, input_tensor).cpu().numpy()

shap_background = X_test[:100]
explainer = shap.Explainer(model_predict, shap_background, feature_names=feature_columns)
shap_values = explainer(X_test[:100])
np.save(shap_save_path, shap_values.values)
print(f"✅ SHAP 值已儲存至 {shap_save_path}")

# summary plot
shap.plots.bar(shap_values, max_display=max_shap_display)

# 單樣本解釋
try:
    shap.plots.waterfall(shap_values[shap_sample_index], max_display=10)
except:
    print("⚠️ 無法產生 waterfall plot，請確認 shap_sample_index 合理")

# ========== 繪圖 ==========
plt.figure(figsize=(6,6))
plt.scatter(y_trues, y_preds, alpha=0.6)
plt.plot([min(y_trues), max(y_trues)], [min(y_trues), max(y_trues)], 'r--')
plt.xlabel('實際破壞扭力')
plt.ylabel('預測破壞扭力')
plt.title('TabTransformer 預測 vs 實際')
plt.grid(True)
plt.show()
