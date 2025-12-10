import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from tab_transformer_pytorch import TabTransformer

# matplotlib 中文支援
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False

# ========== 使用者設定 ==========
file_path = r'D:\_Document\Others\_FlawDetection\ScrewData\Training\RawData\603.csv'
target_column = '破壞扭力'
feature_columns = ['頭徑','頭厚','牙徑','牙長','針深','槽寬','NYLOK','小槽寬','硬度']
use_all_data_for_training = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# =================================

# 資料處理
df = pd.read_csv(file_path, encoding='big5')
df.columns = df.columns.str.strip()
df = df.dropna(subset=feature_columns + [target_column])

X = df[feature_columns].values.astype(np.float32)
y = df[target_column].values.astype(np.float32).reshape(-1, 1)

if use_all_data_for_training:
    X_train, y_train = X, y
    X_test, y_test = X[:50], y[:50]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵選擇
rf = RandomForestRegressor()
rf.fit(X_train, y_train.ravel())
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
N = 5
selected_features = [feature_columns[i] for i in indices[:N]]

# 選擇特徵後的數據
X_train_selected = X_train[:, indices[:N]]
X_test_selected = X_test[:, indices[:N]]

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# 檢查形狀
print(X_train_scaled.shape)
print(y_train.shape)
print(X_test_scaled.shape)
print(y_test.shape)

# # 資料增強
# def add_noise(data, noise_level=0.01):
#     noise = np.random.normal(0, noise_level, data.shape).astype(np.float32)
#     return data + noise

# def scale_features(data, scale_range=(0.9, 1.1)):
#     scale = np.random.uniform(scale_range[0], scale_range[1], data.shape).astype(np.float32)
#     return data * scale

# def mix_features(data, mix_ratio=0.5):
#     mixed_data = data.copy().astype(np.float32)
#     for i in range(data.shape[0]):
#         j = np.random.randint(0, data.shape[0])
#         mixed_data[i] = mix_ratio * data[i] + (1 - mix_ratio) * data[j]
#     return mixed_data

# def drop_features(data, drop_prob=0.1):
#     dropped_data = data.copy().astype(np.float32)
#     mask = np.random.binomial(1, drop_prob, data.shape).astype(bool)
#     dropped_data[mask] = 0
#     return dropped_data

# def augment_data(data, noise_level=0.01, scale_range=(0.9, 1.1), mix_ratio=0.5, drop_prob=0.1):
#     augmented_data = add_noise(data, noise_level)
#     augmented_data = scale_features(augmented_data, scale_range)
#     augmented_data = mix_features(augmented_data, mix_ratio)
#     augmented_data = drop_features(augmented_data, drop_prob)
#         # 生成相同數量的目標數據
#     augmented_labels = np.tile(labels, (2, 1))  # 假設每個增強方法都生成一倍數量的樣本
    
#     return augmented_data, augmented_labels


# X_train_augmented, y_train_augmented = augment_data(X_train_scaled, y_train)
# X_train_combined = np.vstack((X_train_scaled, X_train_augmented)).astype(np.float32)
# y_train_combined = np.vstack((y_train, y_train_augmented)).astype(np.float32)

# 創建 TensorDataset
train_ds = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test_scaled), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# 建立模型
model = TabTransformer(
    categories=(),
    num_continuous=len(selected_features),
    dim=64,
    depth=6,
    heads=16,
    dim_out=1,
    mlp_hidden_mults=(4, 2)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
loss_fn = torch.nn.MSELoss()

# 訓練模型
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        x_categ = torch.empty((xb.shape[0], 0), dtype=torch.float32).to(device)
        y_pred = model(x_categ, xb)
        loss = loss_fn(y_pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")

# 預測
model.eval()
y_preds, y_trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        x_categ = torch.empty((xb.shape[0], 0), dtype=torch.float32).to(device)
        y_pred = model(x_categ, xb)
        y_preds += y_pred.cpu().numpy().flatten().tolist()
        y_trues += yb.numpy().flatten().tolist()

y_preds = np.array(y_preds)
y_trues = np.array(y_trues)

# 評估
mae = mean_absolute_error(y_trues, y_preds)
rmse = np.sqrt(mean_squared_error(y_trues, y_preds))
r2 = r2_score(y_trues, y_preds)
print(f"\n📊 TabTransformer 評估:")
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# SHAP 分析
explainer_data = torch.tensor(X_test_scaled[:100]).to(device)
def model_predict(input_array):
    with torch.no_grad():
        input_tensor = torch.tensor(input_array, dtype=torch.float32).to(device)
        x_categ = torch.empty((input_tensor.shape[0], 0), dtype=torch.float32).to(device)
        return model(x_categ, input_tensor).cpu().numpy()

explainer = shap.Explainer(model_predict, X_test_scaled[:100], feature_names=selected_features)
shap_values = explainer(X_test_scaled[:100])

shap.plots.bar(shap_values, max_display=10)

# 預測 vs 實際
plt.figure(figsize=(6, 6))
plt.scatter(y_trues, y_preds, alpha=0.6)
plt.plot([min(y_trues), max(y_trues)], [min(y_trues), max(y_trues)], 'r--')
plt.xlabel('實際破壞扭力')
plt.ylabel('預測破壞扭力')
plt.title('TabTransformer 預測 vs 實際')
plt.grid(True)
plt.show()
