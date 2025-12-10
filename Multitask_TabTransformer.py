# 多任務 TabTransformer 完整訓練腳本：預測破壞扭力（回歸） + 破壞模式（分類）
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tab_transformer_pytorch import TabTransformer
import shap
import os

# ========= 模型定義 =========
class MultiTaskTabTransformer(nn.Module):
    def __init__(self, 
                 categories=(),
                 num_continuous=9,
                 dim=64,
                 depth=6,
                 heads=16,
                 num_classes=3):
        super().__init__()
        self.tab_transformer = TabTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_out=dim
        )
        self.reg_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, num_classes)
        )

    def forward(self, x_categ, x_cont):
        x = self.tab_transformer(x_categ, x_cont)
        return self.reg_head(x), self.cls_head(x)

# ========= 資料載入與處理 =========
file_path = r'D:\_Document\Others\_FlawDetection\ScrewData\Training\RawData\618_addClass.csv'
target_column = '破壞扭力'
class_column = '破壞模式'#（分類輸出，例如：(滑牙0、斷裂1、彎曲2)）
feature_columns = ['頭徑','頭厚','牙徑','牙長','針深','槽寬','NYLOK','小槽寬','硬度']

num_epochs = 500
batch_size = 64
learning_rate = 1e-3

plt.rcParams['font.family'] = 'Microsoft JhengHei'

df = pd.read_csv(file_path, encoding='big5')
df = df.dropna(subset=feature_columns + [target_column, class_column])

X = df[feature_columns].values.astype(np.float32)
y_reg = df[target_column].values.astype(np.float32).reshape(-1, 1)
y_cls = df[class_column].values.astype(np.int64)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y_reg = scaler_y.fit_transform(y_reg)

joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42
)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_reg_train), torch.tensor(y_cls_train))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_reg_test), torch.tensor(y_cls_test))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# ========= 模型訓練與評估 =========
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MultiTaskTabTransformer(num_continuous=len(feature_columns)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn_reg = nn.MSELoss()
loss_fn_cls = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for xb, yb_reg, yb_cls in train_loader:
        xb, yb_reg, yb_cls = xb.to(device), yb_reg.to(device), yb_cls.to(device)
        x_categ = torch.empty((xb.shape[0], 0)).to(device)

        pred_reg, pred_cls = model(x_categ, xb)
        loss = loss_fn_reg(pred_reg, yb_reg) + loss_fn_cls(pred_cls, yb_cls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")

# ========= 模型評估 =========
model.eval()
y_reg_true, y_reg_pred, y_cls_true, y_cls_pred = [], [], [], []
with torch.no_grad():
    for xb, yb_reg, yb_cls in test_loader:
        xb = xb.to(device)
        x_categ = torch.empty((xb.shape[0], 0)).to(device)
        pred_reg, pred_cls = model(x_categ, xb)

        y_reg_pred.extend(scaler_y.inverse_transform(pred_reg.cpu().numpy()).flatten())
        y_reg_true.extend(scaler_y.inverse_transform(yb_reg.numpy()).flatten())
        y_cls_pred.extend(torch.argmax(pred_cls, dim=1).cpu().numpy())
        y_cls_true.extend(yb_cls.numpy())

# ========== 評估 ==========
mae = mean_absolute_error(y_reg_true, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_true, y_reg_pred))
r2 = r2_score(y_reg_true, y_reg_pred)
acc = accuracy_score(y_cls_true, y_cls_pred)
print(f"""
📊 TabTransformer評估結果：
MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}
分類準確率（破壞模式）= {acc:.2%}""")

# ========= 自定範圍分類準確度 =========
def in_same_bin(val1, val2, bins):
    for i in range(len(bins)-1):
        if bins[i] <= val1 < bins[i+1] and bins[i] <= val2 < bins[i+1]:
            return True
    return False

bin_ranges = [5.1, 5.8, 6.4, 7.0, 7.6, 8.2]
bin_correct = sum(in_same_bin(p, t, bin_ranges) for p, t in zip(y_reg_pred, y_reg_true))
bin_accuracy = bin_correct / len(y_reg_true)
print(f"分段準確率（預測與實際同區段）：{bin_accuracy:.2%}")

# ========= 分段混淆矩陣 =========
def assign_bin(value, bins):
    for i in range(len(bins)-1):
        if bins[i] <= value < bins[i+1]:
            return f"{bins[i]}~{bins[i+1]}"
    return "out"

y_bin_true = [assign_bin(v, bin_ranges) for v in y_reg_true]
y_bin_pred = [assign_bin(v, bin_ranges) for v in y_reg_pred]


os.makedirs("shap_output", exist_ok=True)

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_bin_true, y_bin_pred), annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(y_bin_true)), yticklabels=sorted(set(y_bin_true)))
plt.xlabel("預測區間")
plt.ylabel("實際區間")
plt.title("分段混淆矩陣")
plt.savefig("shap_output/confusion_matrix.png", bbox_inches='tight')
# plt.tight_layout()
# plt.show()

os.makedirs("shap_output", exist_ok=True)

# ========= SHAP 解釋（回歸輸出） =========
def model_predict(input_array):
    with torch.no_grad():
        input_tensor = torch.tensor(input_array, dtype=torch.float32).to(device)
        x_categ = torch.empty((input_tensor.shape[0], 0)).to(device)
        pred, _ = model(x_categ, input_tensor)
        return pred.cpu().numpy()

explainer_reg = shap.Explainer(model_predict, X_test[:100], feature_names=feature_columns)
shap_values_reg = explainer_reg(X_test[:100])

plt.figure()
shap.plots.bar(shap_values_reg, max_display=10, show=False)
plt.title("SHAP - 回歸輸出 (破壞扭力)")
plt.savefig("shap_output/shap_bar_regression.png", bbox_inches='tight')

# Waterfall plot for one sample
sample_index = 0
plt.figure()
shap.plots.waterfall(shap_values_reg[sample_index], max_display=10, show=False)
plt.savefig("shap_output/shap_waterfall_reg_sample0.png", bbox_inches='tight')

# ========= SHAP 解釋（分類輸出） =========
def model_predict_cls(input_array):
    with torch.no_grad():
        input_tensor = torch.tensor(input_array, dtype=torch.float32).to(device)
        x_categ = torch.empty((input_tensor.shape[0], 0)).to(device)
        _, pred_cls = model(x_categ, input_tensor)
        return torch.nn.functional.softmax(pred_cls, dim=1).cpu().numpy()

explainer_cls = shap.Explainer(model_predict_cls, X_test[:100], feature_names=feature_columns)
shap_values_cls = explainer_cls(X_test[:100])

plt.figure()
# shap.plots.bar(shap_values_cls, max_display=10, show=False)
# plt.title("SHAP - 分類輸出 (破壞模式)")
# plt.savefig("shap_output/shap_bar_classification.png", bbox_inches='tight')
for class_index in range(shap_values_cls.shape[-1]):
    shap.plots.bar(shap_values_cls[..., class_index], max_display=10, show=False)
    plt.title(f"SHAP - 分類輸出 (類別 {class_index})")
    plt.savefig(f"shap_output/shap_bar_classification_class{class_index}.png", bbox_inches='tight')


# Waterfall plot for one sample (class 0 by default)
plt.figure()
# shap.plots.waterfall(shap_values_cls[sample_index], max_display=10, show=False)
# plt.savefig("shap_output/shap_waterfall_cls_sample0.png", bbox_inches='tight')
for class_index in range(shap_values_cls.shape[-1]):
    shap.plots.bar(shap_values_cls[..., class_index], max_display=10, show=False)
    plt.title(f"SHAP - 分類輸出 (類別 {class_index})")
    plt.savefig(f"shap_output/shap_waterfall_class{class_index}.png", bbox_inches='tight')

# ========= 模型儲存 =========
torch.save(model.state_dict(), 'multitask_tabtransformer.pth')
print("✅ 模型已儲存，SHAP 圖片已輸出至 shap_output/")
