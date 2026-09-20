# 🧠 Brain Tumor Classification using ResNet18

A deep learning project for classifying brain MRI images into four classes using **transfer learning with ResNet18** and PyTorch.

## 📌 Classes

- Glioma
- Meningioma
- No Tumor
- Pituitary

## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook

## 🔬 Approach

The project follows a complete image-classification workflow:

1. Dataset exploration
2. Stratified train/validation split
3. Image preprocessing and augmentation
4. Transfer learning using pretrained ResNet18
5. Training with a frozen backbone
6. Partial fine-tuning of ResNet18
7. Model evaluation using accuracy, precision, recall and confusion matrices
8. Misclassification analysis

Three ResNet18 configurations were evaluated:

| Model | Test Accuracy |
|---|---:|
| Frozen ResNet18 | **80.69%** |
| Layer4 + FC Fine-tuning | **91.69%** |
| Layer3 + Layer4 + FC Fine-tuning | **93.19%** |

The final model achieved **93.19% accuracy on the 1,600-image test set**.

## 📂 Project Structure

```text
Directory structure:
└── Brain-Tumor-Classification/
    ├── images/
    │   ├── CompleteFT CM.png
    │   ├── CompleteFT Metrics.png
    │   ├── Frozen Backbone CM.png
    │   ├── Frozen Backbone Metrics.png
    │   ├── Misclassified Images.png
    │   ├── PartialFT CM.png
    │   └── PartialFT Metrics.png
    ├── models/
    │   ├── ResNET_Frozen.pt
    │   ├── ResNET_Frozen_FT.pt
    │   └── ResNET_Frozen_PartialFT.pt
    ├── notebooks/
    │   ├── 00. Data Exploration.ipynb
    │   ├── 01. ResNET.ipynb
    │   └── 02. ResNET Evaluation.ipynb
    ├── README.md
    ├── requirements.txt
    ├── split.py
    └── src/
        ├── data.py
        ├── engine.py
        ├── evaluation.py
        ├── metrics.py
        └── plots.py
```

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Brain-Tumor-Classification
```

### 2️⃣ Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Prepare the dataset

Place the original dataset inside the `archive/` directory:

```text
archive/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Run the dataset splitting script:

```bash
python split.py
```

This creates the following structure:

```text
dataset/
├── train/
├── val/
└── test/
```

The original test set is kept untouched.

### 5️⃣ Run the notebooks

Launch Jupyter:

```bash
jupyter notebook
```

Open the notebooks inside the `notebooks/` directory to reproduce the analysis and experiments.

## 📦 Pretrained Models

Trained ResNet18 checkpoints are available in the `models/` directory.

The final model is the **Layer3 + Layer4 + FC fine-tuned ResNet18**.

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. It is not a medical diagnostic system and should not be used for clinical decision-making.

---