# Bench-Press-Detection
Bench Press Phase Classification with DisenGCN and MediaPipe

An AI-powered system for real-time bench press analysis, phase classification, and rep counting using pose estimation and graph neural networks.

📌 Overview

This project leverages 2D pose estimation (via MediaPipe) and Disentangled Graph Convolutional Networks (DisenGCN) to classify bench press movement phases ("UP" or "DOWN") and count repetitions from video input. It is designed for fitness enthusiasts, trainers, and researchers to analyze exercise form, track performance, and prevent injuries.

🚀 Features

Real-Time Pose Estimation: Extracts 2D coordinates of shoulders, elbows, and wrists using MediaPipe.

Biomechanical Feature Engineering: Computes joint angles (e.g., elbow flexion/extension) and spatial relationships between landmarks.

Graph-Based Phase Classification: Implements DisenGCN to model joint interactions and classify movement phases with 93.99% accuracy.

Rep Counting & Feedback: Tracks repetitions and validates form consistency in real time.

Interactive Web Interface: Built with Streamlit for easy video upload and results visualization.

🛠️ Tech Stack

Pose Estimation: MediaPipe, OpenCV

Deep Learning: PyTorch, PyTorch Geometric (DisenGCN implementation)

Data Processing: Pandas, NumPy, scikit-learn

Frontend: Streamlit

📂 Project Structure

  bench-press-analysis/  
  ├── data/  
  │   ├── train.csv            # Labeled training data (2D coordinates + phase labels)  
  │   └── test.csv             # Test dataset  
  ├── models/  
  │   ├── disentangled_gcn.py  # DisenGCN model implementation  
  │   └── train.py             # Training script  
  ├── utils/  
  │   ├── pose_estimator.py    # MediaPipe-based pose extraction  
  │   └── preprocess.py        # Data normalization and feature engineering  
  ├── app/  
  │   └── main.py              # Streamlit web application  
  └── requirements.txt         # Dependency list  

🖥️ Usage
1. Run the Streamlit App

bash

streamlit run app.py  
Upload a video file or use a webcam for real-time analysis.

View classified phases ("UP"/"DOWN"), rep counts, and form feedback.

🧠 Model Architecture

Disentangled Graph Convolutional Network (DisenGCN)
Input: 24-dimensional feature vector (2D coordinates + visibility for 6 upper-body landmarks).

Graph Construction: Joints as nodes, biomechanical relationships as edges.

Layers:
Disentangled graph convolution to model latent factors (landmarks).
Global pooling and fully connected layers for phase classification.

Loss: Cross-entropy loss with Adam optimizer.

📊 Results

Metric	Value

Test Accuracy	93.99%

Precision	93.4%

Recall	95.04%

Team: 
1) Musaddik Ibrahim Karanje

      🔗 Connect: www.linkedin.com/in/musaddik19
      
      📧 Contact: musaddikkaranje@gmail.com

2) Samrat Ganguly
   
      🔗 Connect: www.linkedin.com/in/samratganguly03
      
      📧 Contact: blueoctopus@gmail.com

3) Sahil Bhadane
  
      🔗 Connect: www.linkedin.com/in/04sahil
      
      📧 Contact: sahilbhadane04@gmail.com
