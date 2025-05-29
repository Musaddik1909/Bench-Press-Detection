import streamlit as st
import subprocess
import tempfile

# Page Configuration
st.set_page_config(page_title="Bench Press Form Analyzer", layout="wide")

# Sidebar Section
st.sidebar.title("Upload & Select Mode")
option = st.sidebar.radio("Select Mode", ("Upload Video"))

# 🎥 Video Processing Section
col1, col2 = st.columns([1, 3])  # Sidebar (1) | Video Output (3)

if option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        with col1:
            st.sidebar.success("Video Uploaded Successfully!")

        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        with col2:
            st.video(video_path)  # Show the uploaded video

        st.sidebar.write("Processing video... Please wait.")
        
        # Run `live.py` with uploaded video
        subprocess.run(["python", "5detection.py", "--video", video_path])

        # Display Accuracy & Total Reps
        try:
            with open("accuracy.txt", "r") as f:
                accuracy_data = f.readlines()

            accuracy = accuracy_data[0].strip() if len(accuracy_data) > 0 else "N/A"
            total_reps = accuracy_data[1].strip() if len(accuracy_data) > 1 else "N/A"

            with col2:
                st.success(f"{accuracy}")
                st.success(f"{total_reps}")

        except:
            with col2:
                st.error("Error retrieving accuracy and reps.")

elif option == "Live Webcam Detection":
    with col1:
        st.sidebar.write("Starting Live Detection...")

    # Run `live.py` in Live Mode
    subprocess.run(["python", "live.py", "--live"])

    # Display Accuracy & Total Reps
    try:
        with open("accuracy.txt", "r") as f:
            accuracy_data = f.readlines()

        accuracy = accuracy_data[0].strip() if len(accuracy_data) > 0 else "N/A"
        total_reps = accuracy_data[1].strip() if len(accuracy_data) > 1 else "N/A"

        with col2:
            st.success(f"Live Bench Press Accuracy: {accuracy}")
            st.success(f"Total Reps: {total_reps}")

    except:
        with col2:
            st.error("Error retrieving accuracy and reps.")
