import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# 1. 載入資料
FEATURE_FILE = "album_features_dinov2.pt"
raw_data = torch.load(FEATURE_FILE, map_location='cpu')

X_list = []
y_list = []

# 只取一部分點來畫，不然圖會太密看不清楚 (例如取 1000 點)
# 或者全部畫也可以
sample_data = raw_data[:] 

for item in sample_data:
    X_list.append(item['embedding'].float().numpy())
    year = item['year']
    decade = (year // 10) * 10
    y_list.append(decade)

X = np.array(X_list)
y = np.array(y_list)

print(f"正在進行 t-SNE 降維 (資料筆數: {len(X)})... 這需要一點時間...")

# 2. 執行 t-SNE
# perplexity: 這是關鍵參數，通常設 30~50
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
X_embedded = tsne.fit_transform(X)

# 3. 畫圖
plt.figure(figsize=(12, 10))
scatter = sns.scatterplot(
    x=X_embedded[:, 0], 
    y=X_embedded[:, 1], 
    hue=y, 
    palette="viridis", # 使用漸層色 (藍->綠->黃)，可以看出時間順序
    legend="full",
    alpha=0.6,
    s=60
)

plt.title(f"t-SNE Visualization of Album Styles (DINOv2 Features)", fontsize=16)
plt.legend(title='Decade', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()