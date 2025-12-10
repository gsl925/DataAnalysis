# train_tabtransformer_with_shap_optimized.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
import os
import chardet
from sklearn.preprocessing import StandardScaler
from tab_transformer_pytorch import TabTransformer

# ========== 使用者設定 ==========
file_path = r'D:\_Document\Others\_FlawDetection\ScrewData\Training\RawData\603.csv'
target_column = '破壞扭力'
base_feature_columns = ['頭徑','頭厚','牙徑','牙長','針深','槽寬','NYLOK','小槽寬','硬度']
use_all_data_for_training = False
num_epochs = 500
batch_size = 64
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 中文顯示
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False

# ======== 讀取與前處理資料 ========
with open(file_path, 'rb') as f:
    encoding = chardet.detect(f.read())['encoding']
df = pd.read_csv(file_path, encoding='big5')
df.columns = df.columns.str.strip()
df = df.dropna(subset=base_feature_columns + [target_column])

# 加入交叉特徵
if '頭徑' in df.columns and '牙徑' in df.columns:
    df['頭徑x牙徑'] = df['頭徑'] * df['牙徑']
if 'NYLOK' in df.columns and '牙長' in df.columns:
    df['NYLOK_ratio'] = df['NYLOK'] / (df['牙長'] + 1e-3)
feature_columns = base_feature_columns + ['頭徑x牙徑', 'NYLOK_ratio']

# 數據轉換與標準化
X = df[feature_columns].values.astype(np.float32)
y = df[target_column].values.astype(np.float32).reshape(-1, 1)
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)

if use_all_data_for_training:
    X_train, y_train = X, y
    X_test, y_test = X[:50], y[:50]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=batch_size)

# 建立模型
model = TabTransformer(
    categories=(),
    num_continuous=len(feature_columns),
    dim=32,
    depth=4,
    heads=8,
    dim_out=1,
    mlp_hidden_mults=(4, 2)
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

# 訓練
train_losses = []
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
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}")

# 評估
model.eval()
y_preds, y_trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        x_categ = torch.empty((xb.shape[0], 0)).to(device)
        y_pred = model(x_categ, xb)
        y_preds += y_pred.cpu().numpy().flatten().tolist()
        y_trues += yb.numpy().flatten().tolist()

# 反標準化
y_preds = scaler_y.inverse_transform(np.array(y_preds).reshape(-1, 1)).flatten()
y_trues = scaler_y.inverse_transform(np.array(y_trues).reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_trues, y_preds)
rmse = np.sqrt(mean_squared_error(y_trues, y_preds))
r2 = r2_score(y_trues, y_preds)
print(f"\n📊 TabTransformer 評估:")
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# ======== SHAP summary 分析 ========
def model_predict(input_array):
    with torch.no_grad():
        input_tensor = torch.tensor(input_array, dtype=torch.float32).to(device)
        x_categ = torch.empty((input_tensor.shape[0], 0)).to(device)
        return model(x_categ, input_tensor).cpu().numpy()

explainer = shap.Explainer(model_predict, X_test[:100], feature_names=feature_columns)
shap_values = explainer(X_test[:100])
shap.plots.bar(shap_values, max_display=10)

# ======== 預測 vs 實際 ========
plt.figure(figsize=(6,6))
plt.scatter(y_trues, y_preds, alpha=0.6)
plt.plot([min(y_trues), max(y_trues)], [min(y_trues), max(y_trues)], 'r--')
plt.xlabel('實際破壞扭力')
plt.ylabel('預測破壞扭力')
plt.title('TabTransformer 預測 vs 實際')
plt.grid(True)
plt.show()

# TODO: 預留 CNN 對比區塊
