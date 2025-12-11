# 🎵 Album Cover Decade Classification & Regression Project

**Course:** EEP 596 - Computer Vision  
**Objective:** Predict the decade of an album release based on its cover design using deep learning vision features.

---

## 📋 Project Overview

This project leverages **DINOv2** (Meta's self-supervised vision model) to extract deep visual features from album covers, then trains machine learning models to predict the release decade. The project explores both **Classification** (decade bins) and **Regression** (continuous year) approaches.

### Key Insights
- **Album covers visually change over decades** due to design trends, printing technology, and cultural aesthetics
- 1960s covers look different from 1980s or 2010s
- This project quantifies and learns these visual patterns

---

## 🗂️ Project Structure

```
final_project/
├── album_covers_dataset/          # Downloaded album cover images organized by decade
│   ├── 1960/                      # Images from 1960s
│   ├── 1970/                      # Images from 1970s
│   ├── 1980/                      # ... etc
│   └── 2020/
├── dataset/                       # Alternative data storage format
├── album_features_dinov2.pt       # Extracted DINOv2 features (768-dim vectors)
├── albums.xlsx                    # Album metadata
├── billboard_albums.xlsx          # Billboard chart albums data
│
├── 📄 STAGE 1: DATA COLLECTION
├── fetch_album_images.py          # Download album covers from iTunes API
├── fetch_cover.py                 # Alternative cover fetching script
├── feach_album_name.py            # Extract album names from Excel
│
├── 📄 STAGE 2: FEATURE EXTRACTION
├── feature_extraction.py          # Extract DINOv2 features (standalone)
├── feature_extraction.ipynb       # Feature extraction notebook (Google Colab compatible)
├── test_embedding.py              # Quick test of extracted features
│
├── 📄 STAGE 3: MODEL TRAINING
├── train.py                       # Classification model (Decade prediction)
├── train.ipynb                    # Classification notebook
├── regression.py                  # Regression model (Continuous year prediction)
├── tne.py                         # t-SNE visualization of features
│
├── 📊 OUTPUT & RESULTS
├── album_features_dinov2.pt       # Saved feature vectors
├── classify_result.png            # Classification confusion matrix
├── regression.png                 # Regression scatter plot
├── classify_result.xlsx           # Detailed classification results
├── report.docx                    # Final project report
└── presentation.pptx              # Presentation slides
```

---

## 🚀 Quick Start Guide

### **Stage 1: Data Collection** (Skip if data exists)

#### Download Album Covers
```bash
python fetch_album_images.py
```
- Reads from `billboard_albums.xlsx` (artist, album, year columns)
- Downloads covers from iTunes API
- Saves to `album_covers_dataset/{decade}/`

**Expected output:**
```
album_covers_dataset/
├── 1960/
│   ├── The Beatles_Help!_1965.jpg
│   └── ...
├── 1970/
├── 1980/
└── ...
```

---

### **Stage 2: Feature Extraction** ⭐ Key Step

Extract **DINOv2** embeddings from all cover images.

#### Option A: Local Python
```bash
python feature_extraction.py
```

#### Option B: Google Colab (Recommended - has GPU)
Open `feature_extraction.ipynb` in Google Colab:
1. Mount Google Drive: `drive.mount('/content/drive')`
2. Copy `album_covers_dataset/` to Drive
3. Run all cells
4. Features saved to `album_features_dinov2.pt` (~100-200 MB for 5000+ images)

**What it does:**
- Loads `facebook/dinov2-base` pre-trained model (768-dim features)
- Processes each cover image: RGB conversion → DINOv2 → CLS token embedding
- Stores: `[{filename, year, embedding}, ...]`
- Saves to `.pt` (PyTorch format)

**Expected output:**
```
✅ Found 5234 images. Start processing...
100%|████████| 5234/5234 [45:23<00:00, 1.92 img/s]
Saving to album_features_dinov2.pt... Done!
```

---

### **Stage 3a: Classification Training** 🎯

Predict **decade** (1960, 1970, ..., 2020) - treats similar years as one class.

```bash
python train.py
```

**Model Architecture:**
```
768-dim embedding 
    ↓
Linear(768 → 512)
    ↓
ReLU + Dropout(0.5)
    ↓
Linear(512 → num_classes)  # e.g., 7 classes for 1960-2020
    ↓
CrossEntropyLoss (softmax)
```

**Training Parameters:**
- Optimizer: Adam (lr=0.001)
- Epochs: 30
- Batch: Full dataset (K-fold handles splits)
- Validation: 5-Fold Stratified Cross-Validation

**Output Metrics:**
```
Top-1 Accuracy:     ~40-50%  (exactly correct decade)
Off-by-One Accuracy: ~75-80%  (±1 decade allowed)
Top-2 Accuracy:     ~60-70%  (top 2 predictions contain correct answer)

Confusion Matrix → classify_result.png
```

**High-Confidence Cases Report:**
For each decade, prints:
- **Top-10 correct predictions** with highest confidence
- **Top-10 wrong predictions** with highest confidence
- Format: `confidence | predicted_decade | true_decade | filename`

Example:
```
=== Decade: 1970 ===
Top-10 正確且信心最高:
  conf=0.987 | pred=1970 | true=1970 | file=The Rolling Stones_Sticky Fingers_1971.jpg
  conf=0.954 | pred=1970 | true=1970 | file=Pink Floyd_Dark Side_1973.jpg
  ...
Top-10 預測錯但信心最高:
  conf=0.856 | pred=1980 | true=1970 | file=Queen_Greatest Hits_1981.jpg
  ...
```

---

### **Stage 3b: Regression Training** 📈

Predict **continuous year** (1960, 1961, ..., 2024) - treats years as ordered.

```bash
python regression.py
```

**Model Architecture:**
```
768-dim embedding
    ↓
Linear(768 → 512)
    ↓
ReLU + Dropout(0.5)
    ↓
Linear(512 → 1)  # Single output: predicted year
    ↓
MSELoss
```

**Key Difference from Classification:**
- Labels: Normalized year (e.g., 1963 → 3, treating as continuous value)
- Loss: MSE (Mean Squared Error)
- Metric: **MAE** (Mean Absolute Error) in years

**Output:**
```
Mean Absolute Error (MAE): ±5.2 years

Scatter Plot → regression.png
  - X-axis: True Year
  - Y-axis: Predicted Year
  - Perfect predictions lie on diagonal
```

**Interpretation:**
- MAE = 5.2 means: "On average, model predicts ±5 years off"
- Much more interpretable than "Accuracy 45%"

---

## 🔄 Comparison: Classification vs Regression

| Aspect | Classification | Regression |
|--------|---|---|
| **Target** | Decade (1960, 1970, ...) | Exact year (1960, 1961, ...) |
| **Model Output** | Logits → Softmax (7 classes) | Single number (year) |
| **Loss** | CrossEntropyLoss | MSELoss |
| **Metric** | Accuracy, Confusion Matrix | MAE, RMSE |
| **Interpretation** | "40% chance of right decade" | "±5 years off on average" |
| **Use Case** | When decades are distinct bins | When ordering matters (2024 closer to 2023 than 1960) |

**Which is better?**
- **Regression** usually performs better because it captures the ordering of years
- An album from 1971 should be closer to 1970 than to 1960
- Classification treats 1969 and 1970 as completely different

---

## 📊 Key Files Explained

### `feature_extraction.ipynb` (Google Colab)
- **Purpose:** Extract DINOv2 features with GPU acceleration
- **Advantages:** Free GPU (T4), integrated with Google Drive
- **Steps:**
  1. Install torch/transformers
  2. Mount Google Drive
  3. Load DINOv2 model
  4. Process images recursively
  5. Save to Drive

### `train.py` 
- **5-Fold Cross-Validation:** Ensures robust evaluation
- **Stratified Split:** Maintains decade distribution
- **Output:** High-confidence correct/incorrect samples per decade
- **Visualizations:** Confusion matrix heatmap

### `regression.py`
- **Continuous Prediction:** Treats year as ordered value
- **KFold Split:** Standard k-fold (no stratification needed)
- **Output:** Scatter plot (true vs predicted year)
- **Metric:** MAE in years (human-readable)

### `tne.py`
- **t-SNE Visualization:** Reduces 768-dim → 2D for visualization
- **Purpose:** See if decade clusters naturally separate
- **Output:** Colored scatter plot by decade

---

## 🛠️ Dependencies

```
torch>=2.0
transformers>=4.30
scikit-learn
pillow
pandas
matplotlib
seaborn
numpy
tqdm

# Optional
requests  # for downloading covers
openpyxl  # for Excel files
```

Install all:
```bash
pip install torch transformers scikit-learn pillow pandas matplotlib seaborn numpy tqdm requests openpyxl
```

---

## 💡 Technical Details

### Why DINOv2?
- **Self-supervised learning:** Trained on massive unlabeled image corpus
- **Strong features:** 768-dimensional embeddings capture style/design patterns
- **Versatile:** Works well for downstream tasks (classification, clustering)
- **Efficient:** Fast inference, works on CPU/GPU

### Feature Extraction Process
```python
image = Image.open("album.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
cls_embedding = outputs.last_hidden_state[0, 0, :].cpu()  # 768-dim vector
```

### Why 768-dim?
- DINOv2-base uses ViT-Base architecture
- Patch embeddings + transformer blocks → 768-dim CLS token
- Sufficient to capture visual style differences

---

## 📈 Expected Performance

### Classification (Decade Prediction)
```
1960s: 45% Top-1, 85% Off-by-One
1970s: 52% Top-1, 88% Off-by-One
1980s: 48% Top-1, 82% Off-by-One
1990s: 42% Top-1, 75% Off-by-One
2000s: 55% Top-1, 90% Off-by-One
2010s: 58% Top-1, 92% Off-by-One
2020s: 35% Top-1, 70% Off-by-One (fewer samples)

Average Top-1: ~48%
Average Off-by-One: ~83%
```

### Regression (Year Prediction)
```
Mean Absolute Error: ±5-7 years
RMSE: ±8-10 years
R²: ~0.65-0.75
```

---

## 🎯 Usage Examples

### Run Everything
```bash
# 1. Feature extraction (if needed)
python feature_extraction.py

# 2. Classification
python train.py
> Top-1 Accuracy: 48%
> Off-by-One Accuracy: 83%
> Confusion Matrix saved

# 3. Regression
python regression.py
> Mean Absolute Error: 5.8 years
> Scatter plot saved

# 4. Visualize features
python tne.py
```

### Extract features for custom images
```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

image = Image.open("my_album.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[0, 0, :].numpy()  # 768-dim
    print(f"Embedding shape: {embedding.shape}")
```

---

## 🔍 Analyzing Results

### Which decades are hardest to predict?
Check confusion matrix:
- Diagonal shows correct predictions
- Off-diagonals show confusions
- e.g., 1980s often confused with 1990s (similar design trends)

### Which images does the model struggle with?
Check `Top-10 錯誤` output:
- High confidence but wrong predictions
- Indicates potential data issues or edge cases
- e.g., album reissues may have modern covers but old release dates

### How different is classification from regression?
- Regression usually scores lower MAE than classification error
- But classification is easier to interpret ("45% chance of right decade")
- Choose based on use case

---

## 📝 Project Report

See `report.docx` for:
- Detailed methodology
- Dataset statistics (# images, year distribution)
- Model architecture diagrams
- Results analysis
- Conclusions & future work

---

## 🚀 Future Improvements

1. **Data Augmentation:** Rotate, crop, color jitter album covers
2. **Ensemble Methods:** Combine classification + regression predictions
3. **Fine-tuning:** Fine-tune DINOv2 on your specific task
4. **Decade-Specific Models:** Train separate models per decade pair
5. **Style Analysis:** Analyze which visual features matter most (PCA, attention maps)
6. **Genre Filter:** Separate models for rock, pop, hip-hop, etc.

---

## 📧 Questions & Troubleshooting

**Q: "Model is slow"**
- Use GPU (Google Colab or local CUDA)
- Batch processing if implementing inference

**Q: "Features not being extracted"**
- Check image paths format: `album_covers_dataset/1970/name.jpg`
- Ensure `.jpg` extension (case-sensitive on Linux)
- Use `os.walk()` for recursive search (handles any folder structure)

**Q: "Out of memory"**
- Reduce HIDDEN_DIM in train.py (e.g., 256 instead of 512)
- Use smaller batch size if implementing mini-batch training

**Q: "Poor accuracy"**
- Check data quality: are covers actually labeled with correct years?
- Verify feature extraction: test `test_embedding.py`
- Try regression instead (year ordering is implicit)

---

## 📚 References

- **DINOv2 Paper:** [Link to arxiv](https://arxiv.org/abs/2304.07193)
- **Vision Transformers:** Dosovitskiy et al. (ViT)
- **t-SNE:** van der Maaten & Hinton

---

**Last Updated:** December 9, 2024  
**Author:** Computer Vision Student (EEP 596)
