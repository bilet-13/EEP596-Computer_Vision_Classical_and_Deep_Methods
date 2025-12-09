import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, r2_score

# ================= 設定區 =================
FEATURE_FILE = "album_features_dinov2.pt" 
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30  
HIDDEN_DIM = 512
DROPOUT_RATE = 0.5 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("正在載入特徵資料...")
raw_data = torch.load(FEATURE_FILE, map_location='cpu')

X_list = []
y_list = []

# --- 1. 資料處理：保留連續年份 ---
min_year = 1960 # 用來做基準點 (Normalization)

for item in raw_data:
    embedding = item['embedding'].float()
    X_list.append(embedding)
    
    # 直接取年份數值
    year = item['year']
    
    # 正規化：把 1960 變成 0, 1970 變成 10... 這樣模型比較好學
    # 預測時只要把輸出 + 1960 就能還原
    y_list.append(float(year - min_year))

X = torch.stack(X_list) 
y = torch.tensor(y_list, dtype=torch.float32).view(-1, 1) # Regression 需要 Shape 為 (N, 1)

print(f"總資料筆數: {len(X)}")
print(f"特徵維度: {X.shape[1]}")
print(f"年份範圍: {min_year} + [0 ~ {y.max().item()}]")

# ================= 2. 定義 MLP 模型 (Regression) =================
class YearRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(YearRegressor, self).__init__()
        # 架構: Input -> Linear -> ReLU -> Dropout -> Linear -> Output (1維)
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE) 
        self.layer2 = nn.Linear(hidden_dim, 1) # 輸出只有 1 個數值 (預測年份)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

# ================= 3. 5-Fold Cross-Validation =================
# 回歸問題通常用 KFold (Stratified 是給分類用的，但如果想依年代分層也可以用 StratifiedKFold 搭配 binning，這邊先用簡單的 KFold)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

fold_maes = [] # 記錄每一折的平均誤差 (年)
all_preds = []
all_targets = []

print("\n🚀 開始 5-Fold Cross Validation (Regression)...")

X_numpy = X.numpy()
y_numpy = y.numpy()

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_numpy)):
    print(f"\n--- Fold {fold + 1} / 5 ---")
    
    # 切分資料
    X_train, X_val = X[train_idx].to(device), X[val_idx].to(device)
    y_train, y_val = y[train_idx].to(device), y[val_idx].to(device)
    
    # 初始化模型
    input_dim = X.shape[1]
    model = YearRegressor(input_dim, HIDDEN_DIM).to(device)
    
    # 定義 Loss (MSE 用於訓練，因為它對大誤差懲罰重，收斂快)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 訓練迴圈
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train) # MSE Loss
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, MSE Loss: {loss.item():.4f}")
            
    # 驗證迴圈
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        
        # 還原成年份 (加上 min_year)
        preds_real_year = val_outputs.cpu().numpy() + min_year
        targets_real_year = y_val.cpu().numpy() + min_year
        
        # 計算 MAE (平均絕對誤差) - 這是給人類看的指標
        # "平均預測錯幾年？"
        mae = mean_absolute_error(targets_real_year, preds_real_year)
        fold_maes.append(mae)
        
        print(f"Fold {fold+1} MAE: {mae:.4f} years (平均誤差 {mae:.1f} 年)")
        
        all_preds.extend(preds_real_year.flatten())
        all_targets.extend(targets_real_year.flatten())

# ================= 4. 結果分析 =================
print("\n" + "="*30)
print(f"平均 MAE (Mean Absolute Error): {np.mean(fold_maes):.4f} 年")
print("="*30)

# 繪製 Scatter Plot (真實年份 vs 預測年份)
plt.figure(figsize=(10, 8))
plt.scatter(all_targets, all_preds, alpha=0.3, s=10) # alpha讓重疊點看得出密度

# 畫一條對角線 (完美預測線)
min_val = min(min(all_targets), min(all_preds))
max_val = max(max(all_targets), max(all_preds))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

plt.xlabel('True Year')
plt.ylabel('Predicted Year')
plt.title(f'Regression Result: True vs Predicted Year (MAE: {np.mean(fold_maes):.2f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()