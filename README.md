# brain-tumor-classification
SVM, CNN, and ViT models for brain tumor classification using MRI images


## 🧠 Brain Tumor Classification using SVM (Support Vector Machine)

### 👤 Contributor: Richa Patel  
Part of a collaborative project for our Machine Learning final assignment.  
This section focuses on a lightweight and interpretable model using Support Vector Machines (SVM) with handcrafted image features (HOG + LBP).

---

### 🗂️ Dataset
We used the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), which includes:
- 4 classes: `glioma`, `meningioma`, `pituitary`, and `no tumor`
- Separate `Training/` and `Testing/` folders
- Grayscale MRI brain scan images

---

### 🧪 My Approach: SVM + Feature Engineering

#### ✅ Preprocessing
- Converted images to grayscale
- Resized to 128×128 pixels

#### ✅ Feature Extraction
- **HOG (Histogram of Oriented Gradients):** captures edge and shape information
- **LBP (Local Binary Patterns):** captures texture features
- Features were concatenated into a single vector per image

#### ✅ Model Training
- Used `scikit-learn`’s `SVC` with both **Linear** and **RBF** kernels
- Evaluated using accuracy, precision, recall, F1-score, and confusion matrix

---

### 📊 Results

| Model        | Features     | Kernel | Accuracy |
|--------------|--------------|--------|----------|
| SVM          | HOG          | Linear | 94.74%   |
| SVM          | HOG + LBP    | Linear | **94.81%** ✅ |
| SVM          | HOG + LBP    | RBF    | 93.06%   |

- Best performance achieved with Linear SVM and combined features.
- Excellent classification on “no tumor” and “pituitary” classes.

---

### 📁 Files Included
- `brain_tumor_svm.ipynb`: Full notebook with code, evaluations, and visualizations
- `confusion_matrix_linear.png`, `confusion_matrix_rbf.png`: Heatmaps of model predictions
- `accuracy_comparison_graph.png`: Comparison of model accuracies
- `rbf_metrics_by_class.png`: Precision, Recall, F1 for RBF
- `all_models_class_metrics_comparison.png`: Final comparison of all models

---

### 💾 Model Saving
The best model (`Linear SVM with HOG + LBP`) was saved using `joblib`:

```python
joblib.dump(model, 'svm_hog_lbp_linear.pkl')
