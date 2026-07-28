# ♻️ EcoSort AI — Recyclable & Household Waste Classification Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-Transfer%20Learning-orange)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?logo=flask)
![HTML5](https://img.shields.io/badge/HTML5-Frontend-E34F26?logo=html5)
![CSS3](https://img.shields.io/badge/CSS3-Styling-1572B6?logo=css3)
![JavaScript](https://img.shields.io/badge/JavaScript-Interactive-F7DF1E?logo=javascript)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render)
![License](https://img.shields.io/badge/License-MIT-yellow)

> AI-powered web application that automatically classifies recyclable and household waste into **30 categories** using **MobileNetV2** and provides disposal guidance with confidence scores through a modern Flask web application.

---

# 🔗 Live Demo

**Website:** https://recycle-and-house-hold-waste-b4o1.onrender.com/

---

# 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Model Architecture](#model-architecture)
- [Project Workflow](#project-workflow)
- [Folder Structure](#folder-structure)
- [Deployment](#deployment)
- [Challenges & Solutions](#challenges--solutions)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Installation](#installation)
- [Author](#author)
- [License](#license)

## 📖 Overview

EcoSort AI is a Deep Learning based waste classification system developed to automatically identify recyclable and household waste from images. The application assists users in proper waste segregation by predicting the waste category and providing disposal recommendations.

The project uses **Transfer Learning** with **MobileNetV2**, enabling accurate image classification while maintaining low computational requirements suitable for web deployment.

---

## ✨ Features

- Classifies 30 waste categories
- MobileNetV2 Transfer Learning model
- Real-time image prediction
- Top-5 prediction probabilities
- Confidence score display
- Disposal guidance
- Responsive web interface
- Drag & Drop image upload
- Mobile-friendly design
- Flask REST API
- Render deployment

---

## 🏗 System Architecture

```
User Uploads Image
        │
        ▼
 Flask Web Application
        │
        ▼
 Image Preprocessing
 (Resize → Normalize)
        │
        ▼
 MobileNetV2 Model
        │
        ▼
 Softmax Classification
        │
        ▼
 Prediction + Confidence
        │
        ▼
 Disposal Recommendation
```

---

## 📂 Dataset

Dataset Size

- Approximately 15,000 Images
- 30 Waste Categories
- Default Images
- Real-world Images

Example Classes

- Plastic Water Bottles
- Plastic Bags
- Cardboard Boxes
- Newspapers
- Food Waste
- Coffee Grounds
- Tea Bags
- Glass Bottles
- Aluminum Cans
- Shoes
- Clothing
- Styrofoam Containers
- Eggshells
- Office Paper

---

## 🤖 Model Architecture

Model Used

**MobileNetV2 (Transfer Learning)**

Additional Layers

- GlobalAveragePooling2D
- Dense(128, ReLU)
- Dropout
- Dense(30, Softmax)

Training Configuration

- Optimizer : Adam
- Loss Function : Categorical Crossentropy
- Epochs : 3
- Batch Size : 16
- Input Size : 128 × 128

---

## ⚙️ Tech Stack

| Layer | Technology |
|--------|------------|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Model | MobileNetV2 |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Image Processing | Pillow, NumPy |
| Training | Google Colab |
| Deployment | Render |
| Version Control | GitHub |

---

## 🔄 Project Workflow

1. Upload waste image
2. Flask receives image
3. Image preprocessing
4. MobileNetV2 extracts features
5. Softmax predicts class
6. Top-5 predictions generated
7. Disposal guidance displayed

---

## 📁 Project Structure

```
EcoSort-AI/

│── app.py
│── requirements.txt
│── class_names.json
│── waste_classification_model.h5

│── templates/
│      index.html

│── static/
│      uploads/

│── dataset/

│── README.md
```

---

# 🚀 Deployment

Deployment Pipeline

Google Colab

↓

GitHub

↓

Render

↓

Live Web Application

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Model | MobileNetV2 |
| Classes | 30 |
| Dataset | ~15,000 Images |
| Validation Accuracy | 82.40% |
| Framework | TensorFlow |
| Deployment | Render |

---

## ⚡ Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Large dataset | Image resizing and batching |
| Long training time | Used T4 GPU in Google Colab |
| Memory limitations | MobileNetV2 lightweight architecture |
| Render timeout | Optimized TensorFlow and Gunicorn configuration |
| Incorrect dataset path | Auto-detected dataset structure |

---

## 🌱 Future Enhancements

- Camera-based live detection
- Android application
- IoT Smart Dustbin Integration
- Multi-language support
- Offline prediction
- Higher accuracy models
- Cloud database integration

---

## 🛠 Installation

```bash

gh repo clone farzeen-05/recycle-and-house-hold-waste-classification-using-deep-learning

cd recycle-and-house-hold-waste-classification-using-deep-learning

pip install -r requirements.txt

python app.py
```

Open

```
http://localhost:5000
```

---

## Author

**Farzeen Abdul Khadir**
ECE Graduate | ML & Full-Stack Developer | MLOps & Cloud

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://www.linkedin.com/in/farzeen-abdul-khadir-8921ba2a1)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github)](https://github.com/farzeen-05)


---

## 📄 License

This project is licensed under the MIT License.
