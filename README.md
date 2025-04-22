# NIH Chest X-ray 14 - Multi-Label Classification

## 📌 Project Overview

This project focuses on developing a **deep learning model** to classify **14 different thoracic diseases** from the **NIH Chest X-ray 14 dataset**. The dataset contains **112,120 frontal-view X-ray images** with **multi-label annotations**, making it a challenging medical imaging task.

## 🚀 Key Features

- **Multi-label classification** using **deep learning**
- **Custom data generator** for efficient training & preprocessing
- **Weighted binary cross-entropy loss** to handle class imbalance
- **Model architecture inspired by research paper**
- **Evaluation using AUC-ROC scores for each disease**

## 🏥 Dataset

- **Source:** NIH Chest X-ray 14 dataset ([Download Here](https://nihcc.app.box.com/v/ChestXray-NIHCC))
- **Images:** 112,120 frontal chest X-ray images
- **Labels:** 14 thoracic disease conditions
- **Multi-label Classification:** Each image can have **multiple diseases** or be normal
- **Multi-hot Encoded Vector:** Converted each disease into a multi-hot encoded vector of 14 dimensions. For normal X-Ray the vector will have zeroes.

## 📊 Performance Metrics

| Disease             | AUC-ROC Score |
| ------------------- | ------------- |
| Atelectasis         | 0.9315        |
| Cardiomegaly        | 0.9742        |
| Consolidation       | 0.9541        |
| Edema               | 0.9788        |
| Effusion            | 0.9508        |
| Emphysema           | 0.9822        |
| Fibrosis            | 0.9631        |
| Hernia              | 0.9967        |
| Infiltration        | 0.8751        |
| Mass                | 0.9549        |
| Nodule              | 0.9382        |
| Pleural Thickening  | 0.9586        |
| Pneumonia           | 0.9530        |
| Pneumothorax        | 0.9737        |
| **Overall AUC-ROC** | **0.9561**    |

## 🛠️ Model Architecture

The model follows a **CNN-based architecture**, leveraging a **custom-designed neural network** for high accuracy:

- **Base Model:** Pretrained **ResNet50** without fine-tuning
- **Custom Layers:** Fully connected layers for multi-label classification
- **Activation:** Sigmoid function for multi-label probabilities
- **Loss Function:** Weighted **Binary Cross-Entropy Loss**
- **Optimizer:** Adam

## 📜 Research Paper Reference

- Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases," IEEE CVPR, 2017. [[Paper](https://arxiv.org/abs/1705.02315)]

## 🤝 Contributions

Feel free to contribute! Open an issue or create a pull request if you'd like to improve the project. 🚀

## 📧 Contact

For any queries, reach out via:

- 🔗 LinkedIn: [Shubham Singh Sainger](https://www.linkedin.com/in/shubham-sainger/)
- 📧 Email: [shubhamsainger97@gmail.com](mailto\:shubhamsainger97@gmail.com)

---
