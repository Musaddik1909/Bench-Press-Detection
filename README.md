**# Bench-Press-Detection**
Bench Press Phase Classification with DisenGCN and MediaPipe
An AI-powered system for real-time bench press analysis, phase classification, and rep counting using pose estimation and graph neural networks.

***📌 Overview**
This project leverages 3D pose estimation (via MediaPipe) and Disentangled Graph Convolutional Networks (DisenGCN) to classify bench press movement phases ("UP" or "DOWN") and count repetitions from video input. It is designed for fitness enthusiasts, trainers, and researchers to analyze exercise form, track performance, and prevent injuries.

**🚀 Features**
Real-Time Pose Estimation: Extracts 3D coordinates of shoulders, elbows, and wrists using MediaPipe.

Biomechanical Feature Engineering: Computes joint angles (e.g., elbow flexion/extension) and spatial relationships between landmarks.

Graph-Based Phase Classification: Implements DisenGCN to model joint interactions and classify movement phases with 93.99% accuracy.

Rep Counting & Feedback: Tracks repetitions and validates form consistency in real time.

Interactive Web Interface: Built with Streamlit for easy video upload and results visualization.

**🛠️ Tech Stack**
Pose Estimation: MediaPipe, OpenCV

Deep Learning: PyTorch, PyTorch Geometric (DisenGCN implementation)

Data Processing: Pandas, NumPy, scikit-learn

Frontend: Streamlit

**📂 Project Structure**

  bench-press-analysis/  
  ├── data/  
  │   ├── train.csv            # Labeled training data (3D coordinates + phase labels)  
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
