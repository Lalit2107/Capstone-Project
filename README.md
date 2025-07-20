# 🫁 Lung Disease Detection Using Chest X-ray Images

An AI-powered backend system to detect and classify lung diseases — including Pneumonia, Pneumothorax, and Normal conditions — using deep learning on chest X-ray images. This project focuses on automated medical image classification with enhanced preprocessing for improved accuracy.

---

## 🔍 Overview

This project leverages a Convolutional Neural Network (CNN) trained on chest X-ray datasets to diagnose lung conditions. It uses advanced image preprocessing techniques such as **histogram equalization** and **fuzzy logic-based contrast enhancement** to improve feature visibility and classification accuracy.

---

## 🧠 Key Features

- 🏥 Classifies X-ray images into:
  - **Pneumonia**
  - **Pneumothorax**
  - **Normal**
- 🧠 Built using a custom CNN model
- 🔧 Preprocessing pipeline includes:
  - Histogram Equalization
  - Fuzzy Logic-based contrast enhancement
- ⚡ GPU acceleration enabled (PyTorch)
- 📈 Includes training and testing accuracy visualization
- 💾 Saves trained model for inference: `lung_disease_cnn_model.pth`

---

## 🛠️ Tech Stack

| Component        | Technology            |
|------------------|------------------------|
| Language         | Python                 |
| Deep Learning    | PyTorch                |
| Image Processing | OpenCV, NumPy          |
| Visualization    | Matplotlib, tqdm       |
| Dataset          | Chest X-ray images (3 classes) |

---

## ⚙️ How It Works

1. Preprocess X-ray images with histogram equalization and fuzzy logic.
2. Load and split dataset into training and testing sets.
3. Train a CNN model on the enhanced images.
4. Evaluate model performance (accuracy, loss).
5. Save the model for later use in prediction.

---

## 🧪 Model Performance

> 📊 Accuracy: (Add your actual result here, e.g. ~94%)  
> 📉 Loss: (Add your test loss here, e.g. 0.16)  
> 🧬 Evaluation done using separate test set and confusion matrix.

---
