import cv2
import torch
import numpy as np
import pandas as pd
import mediapipe as mp
import argparse
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import pickle

# Define DisenGCN Model (same as in training)
class DisenGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, dropout=0.4):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim // 2)
        self.conv4 = GCNConv(hidden_dim // 2, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Disentangled graph convolution operations
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(F.relu(x1))
        x1 = self.dropout(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(F.relu(x2))
        x2 = self.dropout(x2)

        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(F.relu(x3))
        x3 = self.dropout(x3)

        x_out = self.conv4(x3, edge_index)
        return F.log_softmax(x_out, dim=1)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Best Model & Scaler
# Initialize the model with the correct input dimension
input_dim = 24  # 6 landmarks with x,y,z,visibility (4 values each)
model = DisenGCN(input_dim=input_dim).to(device)
# Load the state dictionary
model.load_state_dict(torch.load("./model/disengcn.pth"))
model.eval()  # Set to evaluation mode

with open("./model/input_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define Landmarks Used in Training
FEATURES = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST"
]

# Function to Calculate Angle
def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

# Function to create graph from input features
def create_graph(X):
    # Convert to tensor
    x = torch.tensor(X, dtype=torch.float32).to(device)
    
    # Create a simple fully connected graph for a single sample
    # This is a simplified approach - you might need to adjust based on your training
    num_nodes = 1
    edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
    
    return Data(x=x, edge_index=edge_index)

# Function to Process Video (Live or Uploaded)
def process_video(video_path=None):
    if video_path is None:
        cap = cv2.VideoCapture(0)  # Webcam Mode
    else:
        cap = cv2.VideoCapture(video_path)  # Uploaded Video Mode

    total_frames = 0
    correct_predictions = 0
    rep_count = 0
    last_label = None  # Track last movement phase (UP/DOWN)

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            # Convert to RGB for Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                row = []

                # Extract Features Consistent with Training
                for lm in FEATURES:
                    keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
                    row.extend([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])

                # Convert Data & Predict
                X_input = np.array(row).reshape(1, -1)
                X_scaled = scaler.transform(X_input)
                
                # Create graph data
                graph_data = create_graph(X_scaled)
                
                # Make prediction with the model
                with torch.no_grad():
                    output = model(graph_data)
                    prediction = output.argmax(dim=1).item()

                label = "UP" if prediction == 0 else "DOWN"
                color = (0, 255, 0) if prediction == 0 else (0, 0, 255)

                # Accuracy Calculation
                if prediction == 0:
                    correct_predictions += 1

                # Calculate Elbow Angle for Rep Counting
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Classify Movement Phase & Count Reps
                elbow_threshold = 90  # Define elbow bend threshold
                if left_elbow_angle > elbow_threshold and right_elbow_angle > elbow_threshold:
                    current_label = "UP"
                else:
                    current_label = "DOWN"

                if last_label == "DOWN" and current_label == "UP":
                    rep_count += 1  # Count rep when moving from DOWN to UP
                
                last_label = current_label

                # Draw Pose on Image
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Display Prediction & Reps
                # cv2.putText(image, f"Bench Press: {label}", (30, 50),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.putText(image, f"Reps: {rep_count}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Show Video Output
            cv2.imshow("Bench Press Detection", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate Accuracy for Uploaded Video
    if video_path:
        accuracy = (correct_predictions / total_frames) * 100 if total_frames > 0 else 0
        print(f"Bench Press Accuracy: {accuracy:.2f}% | Total Reps: {rep_count}")
        
        # Write accuracy and reps to file
        with open("accuracy.txt", "w") as f:
            f.write(f"Accuracy: {accuracy:.2f}%\nTotal Reps: {rep_count}")


# Argument Parsing for CLI
parser = argparse.ArgumentParser(description="Live or Video Upload Bench Press Detection")
parser.add_argument("--video", type=str, help="Path to the video file (leave blank for live detection)")
args = parser.parse_args()

if args.video:
    process_video(args.video)
else:
    process_video()
