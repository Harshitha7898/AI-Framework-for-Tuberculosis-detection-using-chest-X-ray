# AI-Framework-for-Tuberculosis-detection-using-chest-X-ray
🫁 Tuberculosis Detection from Chest X-rays using Deep Learning
📌 Project Title:
AI Framework for Tuberculosis Diagnosis using Chest X-rays

📖 Overview
This project aims to develop a deep learning-based automatic Tuberculosis (TB) detection system using chest X-ray images. It is designed to assist healthcare professionals in early and accurate diagnosis of TB, especially in resource-constrained environments. The system uses multiple deep learning models including ConvNeXt-Tiny, Vision Transformer (ViT), MobileNetV3, and an ensemble model combining the best-performing architectures.

🧠 Objectives
Build an accurate and robust TB classification model using chest X-ray images.

Handle class imbalance through augmentation, weighted loss functions, and regularization techniques.

Compare and evaluate performance across different architectures.

Deploy and make the model scalable for real-world healthcare applications.

📂 Dataset
Source: Tuberculosis (TB) Chest X-ray Dataset - Kaggle

Structure:
```plaintext
TB_Data/
  ├── Train/
  │   ├── Normal/
  │   └── Tuberculosis/
  ├── Validation/
  │   ├── Normal/
  │   └── Tuberculosis/
  └── Test/
      ├── Normal/
      └── Tuberculosis/
```
Class Imbalance:

Normal: ~3500 images

Tuberculosis: ~700 images

⚙️ Methodology
1. Data Preprocessing
Resizing images to 224x224

Normalization using ImageNet mean and std

Data augmentation: Random horizontal/vertical flip, rotation, color jitter

Split into Train, Validation, Test sets

Used PyTorch ImageFolder and DataLoader

2. Models Used
✅ ConvNeXt-Tiny (fine-tuned)

✅ Vision Transformer (ViT-base)

✅ MobileNetV3 (Lightweight and efficient)

✅ Ensemble Model (ConvNeXt + ViT)

3. Loss Function & Optimization
Weighted Cross-Entropy Loss to handle class imbalance

AdamW optimizer

Learning rate scheduling

Early stopping to avoid overfitting

4. Evaluation Metrics
Accuracy

Precision, Recall, F1-Score

ROC-AUC Curve

Confusion Matrix

Loss & Accuracy Curves

🔍 Results
| Model         | Accuracy  | Precision | Recall | F1-Score  | ROC-AUC   |
| ------------- | --------- | --------- | ------ | --------- | --------- |
| ConvNeXt-Tiny | 96.8%     | 95.1%     | 93.2%  | 94.1%     | 0.972     |
| ViT           | 97.2%     | 95.6%     | 94.0%  | 94.8%     | 0.978     |
| MobileNetV3   | 95.4%     | 93.3%     | 91.0%  | 92.1%     | 0.961     |
| **Ensemble**  | **98.1%** | **97.2%** | 96.0%  | **96.6%** | **0.987** |

🖼️ Visualizations
Training vs Validation Accuracy & Loss

Confusion Matrix

ROC Curves

(Optional) Grad-CAM heatmaps for explainability

🧪 How to Run
PyTorch

torchvision

scikit-learn

matplotlib

numpy

🚀 Training the Model
# Run training for ConvNeXt
python train_convnext.py

# Run training for Vision Transformer
python train_vit.py

# Run ensemble model
python ensemble_model.py
📊 Evaluation

python evaluate_model.py
📈 Sample Output
Include screenshots or plots:

Accuracy/Loss curves

ROC curve

Confusion Matrix

✅ Key Features
End-to-end automated TB detection pipeline

Handles class imbalance effectively

Ensemble learning for performance boost

Model interpretability ready for deployment

Lightweight MobileNetV3 for edge deployment

🔮 Future Work
Incorporate Explainable AI (XAI) using Grad-CAM

Deploy via Flask web app or Streamlit

Integrate into mobile health apps

Expand dataset to multi-class lung disease classification

👩‍⚕️ Impact
This system can be used as a clinical decision support tool to assist radiologists and healthcare workers in early TB diagnosis, particularly in rural or underserved regions lacking access to specialized care.

