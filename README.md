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

> **EcoSort AI** is an AI-powered web application that classifies recyclable and household waste into **30 categories** using **MobileNetV2 Transfer Learning**. It provides real-time predictions, confidence scores, and disposal guidance through an intuitive Flask-based web interface.

---

## 🌐 Live Demo

**🔗 Website:** https://recycle-and-house-hold-waste-b4o1.onrender.com/

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Model Architecture](#model-architecture)
- [Project Workflow](#project-workflow)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Results](#results)
- [Challenges & Solutions](#challenges--solutions)
- [Future Enhancements](#future-enhancements)
- [Installation](#installation)
- [Author](#author)
- [License](#license)

---

## Overview

EcoSort AI is a Deep Learning-based waste classification system designed to identify recyclable and household waste from uploaded images. The application promotes proper waste segregation by predicting the waste category and providing suitable disposal recommendations.

The model is built using **MobileNetV2 Transfer Learning**, offering high accuracy with a lightweight architecture suitable for real-time web deployment.

---

## Features

- ♻️ Classifies 30 waste categories
- 🤖 MobileNetV2 Transfer Learning model
- 📷 Real-time image classification
- 📈 Displays Top-5 predictions
- 🎯 Confidence score for every prediction
- 🗑️ Waste disposal recommendations
- 🌐 Responsive web interface
- 📱 Mobile-friendly design
- 📤 Drag & Drop image upload
- ⚡ Flask REST API
- ☁️ Deployed on Render

---

## Architecture

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
 Top-5 Predictions
        │
        ▼
 Disposal Recommendation
```

---

## Dataset

### Dataset Details

- Approximately **15,000 images**
- **30 waste categories**
- Default images
- Real-world images

### Sample Classes

- Plastic Water Bottles
- Plastic Bags
- Cardboard Boxes
- Newspapers
- Food Waste
- Coffee Grounds
- Tea Bags
- Glass Bottles
- Aluminium Cans
- Shoes
- Clothing
- Styrofoam Containers
- Eggshells
- Office Paper

---

## Tech Stack

| Layer | Technology |
|--------|------------|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| CNN Model | MobileNetV2 |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Image Processing | Pillow, NumPy |
| Training Platform | Google Colab |
| Deployment | Render |
| Version Control | GitHub |

---

## Model Architecture

### Base Model

- MobileNetV2 (Transfer Learning)

### Classification Layers

- GlobalAveragePooling2D
- Dense (128, ReLU)
- Dropout
- Dense (30, Softmax)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Epochs | 3 |
| Batch Size | 16 |
| Input Size | 128 × 128 |

---

## Project Workflow

1. User uploads an image.
2. Flask receives the uploaded file.
3. Image preprocessing (resize and normalization).
4. MobileNetV2 extracts image features.
5. Softmax layer predicts the waste category.
6. Top-5 predictions with confidence scores are generated.
7. Disposal guidance is displayed to the user.

---

## Project Structure

```
EcoSort-AI/
│
├── app.py
├── requirements.txt
├── waste_classification_model.h5
├── class_names.json
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   ├── images/
│   └── uploads/
│
└── dataset/
```

---

## Deployment

### Deployment Pipeline

```
Google Colab
      │
      ▼
   GitHub
      │
      ▼
    Render
      │
      ▼
 Live Web Application
```

**Live Demo:** https://recycle-and-house-hold-waste-b4o1.onrender.com/

---

## Results

| Metric | Value |
|--------|-------|
| Model | MobileNetV2 |
| Dataset Size | ~15,000 Images |
| Number of Classes | 30 |
| Validation Accuracy | 82.40% |
| Framework | TensorFlow |
| Deployment | Render |

---

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Large dataset | Image resizing and batch processing |
| Long training time | Used Google Colab T4 GPU |
| Memory limitations | Lightweight MobileNetV2 architecture |
| Render timeout | Optimised TensorFlow and Gunicorn configuration |
| Dataset path issues | Structured dataset with automatic loading |

---

## Future Enhancements

- 📹 Live camera-based waste detection
- 📱 Android mobile application
- 🤖 IoT Smart Dustbin integration
- 🌍 Multi-language support
- 💾 Offline prediction support
- 📈 Higher accuracy using advanced CNN architectures
- ☁️ Cloud database integration
- 📊 Prediction history and analytics dashboard

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/farzeen-05/recycle-and-house-hold-waste-classification-using-deep-learning.git
```

### Navigate to the Project

```bash
cd recycle-and-house-hold-waste-classification-using-deep-learning
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://localhost:5000
```

---

## Author

**Farzeen Abdul Khadir**

ECE Graduate | Machine Learning Engineer | Full-Stack Developer | MLOps & Cloud Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://www.linkedin.com/in/farzeen-abdul-khadir-8921ba2a1)

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github)](https://github.com/farzeen-05)

[![Email](https://img.shields.io/badge/Email-farzeen98453@gmail.com-EA4335?style=flat&logo=gmail)](mailto:farzeen98453@gmail.com)
 

---

## License

This project is licensed under the **MIT License**.
