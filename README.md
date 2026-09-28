# UCF-Crime ResNet18 — Abnormal Activity Detection

A deep learning-based abnormal activity detection system using the **UCF-Crime dataset** and **ResNet18 transfer learning** to classify surveillance video frames as **Normal** or **Abnormal**.

---

## 📌 Project Overview

Abnormal activity detection is an important computer vision task for intelligent surveillance systems. Traditional surveillance requires continuous human monitoring, which can be time-consuming and difficult to scale.

This project develops a deep learning system that analyzes surveillance footage frame-by-frame and classifies each frame into:

- **Normal**
- **Abnormal**

The project uses a pretrained **ResNet18** convolutional neural network with **ImageNet weights** and fine-tunes the final layers using data derived from the UCF-Crime dataset.

The system can also process GIF/video frames and generate frame-level predictions, confidence scores, prediction statistics, and visualization results.

> **Important:** This implementation is a ResNet18 transfer-learning approach inspired by the UCF-Crime problem. It is **not an exact reproduction of the original UCF-Crime Deep Multiple Instance Learning (MIL) architecture**.

---

## 🎯 Objectives

- Detect abnormal activities in surveillance footage.
- Convert multi-class crime categories into binary **Normal vs Abnormal** classification.
- Apply transfer learning using ResNet18.
- Fine-tune the model on UCF-Crime data.
- Evaluate the model using multiple classification metrics.
- Analyze surveillance footage frame-by-frame.
- Generate prediction statistics and visualizations.
- Build a reproducible deep learning workflow using Kaggle GPU.

---

## 🗂️ Dataset

### UCF-Crime Dataset

The project uses the **UCF-Crime** surveillance video dataset.

The dataset contains normal surveillance videos and multiple abnormal/crime categories.

### Classes Used

#### Normal

- NormalVideos

#### Abnormal

- Abuse
- Arrest
- Arson
- Assault
- Burglary
- Explosion
- Fighting
- RoadAccidents
- Robbery
- Shooting
- Shoplifting
- Stealing
- Vandalism

For this project, all abnormal categories are combined into a single **Abnormal** class.

### Dataset Sampling

| Dataset | Normal | Abnormal | Total |
|---|---:|---:|---:|
| Training | 1,000 | 13,000 | 14,000 |
| Testing | 300 | 3,897 | 4,197 |

The training data was sampled using a fixed random seed for reproducibility.

---

## 🧠 Methodology

The overall workflow is:

```text
UCF-Crime Dataset
        ↓
Select Normal + Abnormal Samples
        ↓
Binary Labeling
        ↓
Image/Frame Preprocessing
        ↓
ResNet18 with ImageNet Weights
        ↓
Initial Transfer Learning
        ↓
Fine-Tuning Layer4 + Fully Connected Layer
        ↓
Binary Classification
        ↓
Model Evaluation
        ↓
Frame-by-Frame Video/GIF Analysis
        ↓
Prediction Statistics & Visualization
```

---

## 🔬 Model Architecture

### ResNet18

The project uses **ResNet18**, a convolutional neural network based on residual learning.

A pretrained ResNet18 model was loaded with **ImageNet weights**.

The original classification layer was replaced with:

```text
Dropout(0.3)
      ↓
Linear(512 → 1)
```

The single output represents the abnormality score.

A sigmoid operation converts the output into an abnormal probability.

### Training Strategy

#### Stage 1 — Transfer Learning

Initially, the pretrained ResNet18 backbone was frozen and only the final fully connected layer was trained.

#### Stage 2 — Fine-Tuning

The final ResNet block (`layer4`) and classification layer were unfrozen and fine-tuned on the UCF-Crime dataset.

This allowed the model to adapt pretrained visual features to surveillance-related patterns.

---

## 🖼️ Image Preprocessing

Input images are resized to:

```text
224 × 224 pixels
```

Training augmentation:

- Resize
- Random horizontal flip
- Tensor conversion
- ImageNet normalization

Testing preprocessing:

- Resize
- Tensor conversion
- ImageNet normalization

ImageNet normalization was used because the model was initialized with ImageNet pretrained weights.

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Model | ResNet18 |
| Pretrained weights | ImageNet |
| Input size | 224 × 224 |
| Batch size | 32 |
| Loss function | BCEWithLogitsLoss |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Fine-tuning learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Dropout | 0.3 |
| Initial training epochs | 5 |
| Fine-tuning epochs | 5 |
| Hardware | NVIDIA Tesla T4 |
| Framework | PyTorch |
| Python environment | Kaggle |

---

## 📊 Training Results

### Initial Transfer Learning

After the first training stage:

```text
Training Accuracy: 94.22%
```

### Fine-Tuning

After fine-tuning `layer4` and the final classification layer:

```text
Training Accuracy: 99.81%
```

The final evaluation was performed on a separate test set.

---

## 📈 Final Test Results

| Metric | Score |
|---|---:|
| Accuracy | **87.30%** |
| Precision | **96.62%** |
| Recall | **89.45%** |
| F1-Score | **92.90%** |
| ROC-AUC | **86.78%** |

### Interpretation

- **Accuracy** measures the overall percentage of correctly classified test samples.
- **Precision** indicates how many samples predicted as abnormal were actually abnormal.
- **Recall** measures how many actual abnormal samples were detected.
- **F1-score** balances precision and recall.
- **ROC-AUC** measures the model's ability to distinguish between normal and abnormal samples across classification thresholds.

---

## 🧩 Confusion Matrix

The final confusion matrix was:

```text
                    Predicted
                  Normal  Abnormal

Actual Normal       178      122
Actual Abnormal     411     3486
```

Therefore:

- True Normal = 178
- False Abnormal = 122
- False Normal = 411
- True Abnormal = 3486

---

## 🎥 Frame-by-Frame Analysis

The trained model was also used to analyze surveillance footage frame-by-frame.

For each frame, the system calculates:

- Normal probability
- Abnormal probability
- Predicted class
- Frame number

The frame predictions can then be summarized to understand the overall activity in the footage.

### Example Analysis

A sample `arrest1.gif` was analyzed frame-by-frame.

```text
Total frames: 241

Normal frames:   239
Abnormal frames:   2

Normal:    99.17%
Abnormal:   0.83%

Average normal probability:   96.01%
Average abnormal probability:  3.99%

Final prediction: NORMAL
```

The frame-level results are stored in:

```text
results/arrest1_all_frame_predictions.csv
```

---

## 📊 Result Visualizations

The repository contains:

### Prediction Timeline

```text
results/arrest1_normal_abnormal_timeline.png
```

This visualization shows how the predicted class changes across frames.

### Prediction Distribution

```text
results/arrest1_prediction_distribution.png
```

This visualization shows the distribution of normal and abnormal predictions.

---

## 📁 Project Structure

```text
UCF-Crime-ResNet18-Detection/
│
├── UCF_Crime_ResNet18_Final.ipynb
│
├── models/
│   └── resnet18_ucf_crime_final.pth
│
├── results/
│   ├── final_results.csv
│   ├── arrest1_all_frame_predictions.csv
│   ├── arrest1_normal_abnormal_timeline.png
│   └── arrest1_prediction_distribution.png
│
├── README.md
│
└── .gitattributes
```

---

## 💾 Trained Model

The trained model is stored as:

```text
models/resnet18_ucf_crime_final.pth
```

Because the model file is approximately 45 MB, it is stored using **Git Large File Storage (Git LFS)**.

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Deep Learning

- PyTorch
- Torchvision
- ResNet18

### Data Processing

- NumPy
- Pandas
- Pillow

### Computer Vision

- OpenCV

### Visualization

- Matplotlib
- Seaborn

### Development Environment

- Kaggle
- Jupyter Notebook
- VS Code
- Git
- GitHub
- Git LFS

### Hardware

- NVIDIA Tesla T4 GPU

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/dhruveesalian/UCF-Crime-ResNet18-Detection.git
```

Move into the project directory:

```bash
cd UCF-Crime-ResNet18-Detection
```

Install the required Python libraries:

```bash
pip install torch torchvision pandas numpy pillow opencv-python matplotlib seaborn scikit-learn
```

If Git LFS is installed, retrieve the model:

```bash
git lfs pull
```

---

## ▶️ Running the Project

Open:

```text
UCF_Crime_ResNet18_Final.ipynb
```

The notebook contains the complete workflow:

1. Dataset configuration
2. Dataset loading
3. Binary label creation
4. Image preprocessing
5. DataLoader creation
6. ResNet18 initialization
7. Transfer learning
8. Fine-tuning
9. Model evaluation
10. Confusion matrix
11. ROC curve
12. Model saving
13. Frame-by-frame analysis
14. Prediction visualization

---

## 📄 Output Files

### Final Metrics

```text
results/final_results.csv
```

Contains the final evaluation metrics.

### Frame-Level Predictions

```text
results/arrest1_all_frame_predictions.csv
```

Contains prediction information for every analyzed frame.

### Timeline

```text
results/arrest1_normal_abnormal_timeline.png
```

Visual representation of frame-level predictions.

### Prediction Distribution

```text
results/arrest1_prediction_distribution.png
```

Visual representation of prediction probabilities/classes.

---

## 🔍 Limitations

This project has several limitations:

- The model performs frame-level classification rather than temporal video understanding.
- A single frame may not contain enough information to determine whether an activity is truly abnormal.
- The approach does not reproduce the temporal MIL architecture proposed in the original UCF-Crime research.
- Class imbalance exists between normal and abnormal samples.
- Frame-level predictions can produce false positives or false negatives.
- The system should not be considered a production-ready security or law-enforcement system without further validation.

---

## 🚀 Future Improvements

Possible future work includes:

- Temporal modeling using LSTM/GRU.
- 3D CNN-based video understanding.
- Transformer-based video models.
- Video-level rather than frame-level classification.
- Deep Multiple Instance Learning (MIL).
- Attention mechanisms.
- Temporal feature extraction.
- Better handling of class imbalance.
- Real-time webcam/video inference.
- Confidence threshold calibration.
- Explainable AI techniques such as Grad-CAM.
- Deployment as a web application or surveillance dashboard.

---

## 📚 Research Context

The project is based on the abnormal activity detection problem addressed by the **UCF-Crime dataset** introduced by Sultani, Chen, and Shah.

Reference:

> Sultani, W., Chen, C., & Shah, M. (2018). Real-World Anomaly Detection in Surveillance Videos. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

The original research introduced a large-scale real-world surveillance anomaly detection dataset and a weakly supervised deep learning approach based on Multiple Instance Learning.

This project uses the same general problem domain and dataset but implements a **binary ResNet18 transfer-learning approach** instead of the original MIL architecture.

---

## 👩‍💻 Author

**Dhruvee Salian**

B.E. Artificial Intelligence & Data Science  
Smt. Indira Gandhi College of Engineering (SIGCE), Navi Mumbai

### Areas of Interest

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Data Science
- AI Research & Development

---

## ⭐ Project Highlights

- ✅ UCF-Crime surveillance dataset
- ✅ Binary Normal/Abnormal classification
- ✅ ResNet18 transfer learning
- ✅ ImageNet pretrained model
- ✅ Fine-tuning of ResNet18 Layer4
- ✅ 14,000 training samples
- ✅ 4,197 test samples
- ✅ 87.30% test accuracy
- ✅ 92.90% F1-score
- ✅ 86.78% ROC-AUC
- ✅ Frame-by-frame analysis
- ✅ Prediction visualizations
- ✅ GPU training on NVIDIA Tesla T4
- ✅ Model versioned using Git LFS
- ✅ Reproducible Kaggle notebook workflow

---

## 📌 Disclaimer

This project is developed for **academic and research purposes**. Predictions should not be treated as definitive evidence of criminal or abnormal behavior. Real-world deployment would require extensive testing, validation, privacy safeguards, fairness evaluation, and human oversight.
