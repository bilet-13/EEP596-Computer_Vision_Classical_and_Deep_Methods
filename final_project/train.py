import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

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
filenames = []

for item in raw_data:
    embedding = item['embedding'].float()
    X_list.append(embedding)
    
    # 取得年份並轉為年代 (Decade)
    # 例如 1963 -> 1960, 2023 -> 2020
    year = item['year']
    decade = (year // 10) * 10 
    y_list.append(decade)

    # 紀錄檔名，若不存在則以索引代替
    filenames.append(item.get('filename', f"idx_{len(filenames)}"))

# 轉成 PyTorch Tensor
X = torch.stack(X_list) # Shape: (N, 768) or (N, 1024)
y_raw = np.array(y_list)
filenames = np.array(filenames)

# 使用 LabelEncoder 把年代 (1960, 1970...) 轉成索引 (0, 1, 2...)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw) # 這是我們訓練用的 Target
y = torch.tensor(y, dtype=torch.long)

# 顯示類別對應關係
class_names = label_encoder.classes_
print(f"總資料筆數: {len(X)}")
print(f"特徵維度: {X.shape[1]}")
print(f"類別對應: {dict(zip(range(len(class_names)), class_names))}")

# ================= 2. 定義 MLP 模型 =================
class DecadeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(DecadeClassifier, self).__init__()
        # 架構: Input -> Linear -> ReLU -> Dropout -> Linear -> Output
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE) 
        self.layer2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

# ================= 3. 5-Fold Cross-Validation 訓練迴圈 =================
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 儲存每一折的結果以便最後平均
fold_accuracies = []
all_preds = []
all_labels = []

# 收集每個年代的高信心案例
best_correct_by_class = {i: [] for i in range(len(class_names))}
best_wrong_by_class = {i: [] for i in range(len(class_names))}

print("\n🚀 開始 5-Fold Cross Validation...")

# 轉回 numpy 做 split 索引 (sklearn 需要 numpy)
X_numpy = X.numpy()
y_numpy = y.numpy()

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_numpy, y_numpy)):
    print(f"\n--- Fold {fold + 1} / 5 ---")
    
    # 切分資料
    X_train, X_val = X[train_idx].to(device), X[val_idx].to(device)
    y_train, y_val = y[train_idx].to(device), y[val_idx].to(device)
    
    # 初始化模型
    input_dim = X.shape[1]
    num_classes = len(class_names)
    model = DecadeClassifier(input_dim, HIDDEN_DIM, num_classes).to(device)
    
    # 定義 Loss 和 Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 訓練迴圈 (Training Loop)
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # (可選) 每 10 epoch 印一次 loss
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            
    # 驗證迴圈 (Validation Loop)
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_probs = torch.softmax(val_outputs, dim=1)

        # Top-1 Accuracy (完全正確)
        val_max_probs, val_preds_top1 = torch.max(val_probs, 1)
        acc_top1 = accuracy_score(y_val.cpu(), val_preds_top1.cpu())
        
        # Off-by-One Accuracy (預測和真實相差 ≤ 1 個類別)
        # 例如: 真實是 1970 (class 1), 預測 1960 (class 0) 或 1980 (class 2) 也算對
        off_by_one_correct = np.abs(val_preds_top1.cpu().numpy() - y_val.cpu().numpy()) <= 1
        acc_off_by_one = off_by_one_correct.mean()
        
        # Top-2 Accuracy (前2名預測中只要有1個對就算對)
        _, val_preds_top2 = torch.topk(val_probs, 2, dim=1)
        acc_top2 = (val_preds_top2 == y_val.unsqueeze(1)).any(dim=1).float().mean().item()
        
        fold_accuracies.append(acc_top1)
        print(f"Fold {fold+1} Top-1 Accuracy: {acc_top1:.4f}, Off-by-One Accuracy: {acc_off_by_one:.4f}, Top-2 Accuracy: {acc_top2:.4f}")
        
        # 收集結果畫 Confusion Matrix (用 Top-1)
        all_preds.extend(val_preds_top1.cpu().numpy())
        all_labels.extend(y_val.cpu().numpy())

        # 收集高信心案例 (依真實年代分組)
        val_true = y_val.cpu().numpy()
        val_pred = val_preds_top1.cpu().numpy()
        val_conf = val_max_probs.cpu().numpy()
        val_files = filenames[val_idx]

        for t, p, conf, fname in zip(val_true, val_pred, val_conf, val_files):
            record = (float(conf), fname, int(p), int(t))
            if p == t:
                best_correct_by_class[t].append(record)
            else:
                best_wrong_by_class[t].append(record)

# ================= 4. 結果分析 =================
print("\n" + "="*30)
print(f"平均準確率 (Mean Accuracy): {np.mean(fold_accuracies):.4f}")
print("="*30)

# 顯示每個年代信心最高的正確 / 錯誤案例 (各取前 10 張)
print("\n📂 每個年代的高信心案例 (Top-10)")
for cls_idx, decade in enumerate(class_names):
    correct_sorted = sorted(best_correct_by_class[cls_idx], key=lambda x: x[0], reverse=True)[:10]
    wrong_sorted = sorted(best_wrong_by_class[cls_idx], key=lambda x: x[0], reverse=True)[:10]

    print(f"\n=== Decade: {decade} ===")
    print("Top-10 正確且信心最高:")
    if correct_sorted:
        for conf, fname, pred_idx, true_idx in correct_sorted:
            pred_decade = class_names[pred_idx]
            print(f"  conf={conf:.3f} | pred={pred_decade} | true={class_names[true_idx]} | file={fname}")
    else:
        print("  (無)")

    print("Top-10 預測錯但信心最高:")
    if wrong_sorted:
        for conf, fname, pred_idx, true_idx in wrong_sorted:
            pred_decade = class_names[pred_idx]
            print(f"  conf={conf:.3f} | pred={pred_decade} | true={class_names[true_idx]} | file={fname}")
    else:
        print("  (無)")

# 繪製 Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Decade')
plt.ylabel('True Decade')
plt.title('Confusion Matrix (All Folds)')
plt.show()