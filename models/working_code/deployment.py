import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

# Load custom YOLO model for object detection and pose estimation
custom_model = YOLO(
    r"E:\Optimizing Workplace Compliance and Safety\models\runs\detect\train\weights\best.pt")  # Replace with your trained model path
pose_model = YOLO(r"E:\Optimizing Workplace Compliance and Safety\models\yolov8n-pose.pt")  # YOLOv8 Pose Estimation Model


# Function to calculate angles between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


# Function to classify poses
def classify_pose(keypoints, frame_height):
    if keypoints is None or len(keypoints) < 17:
        return ["Unknown"]

    nose, left_eye, right_eye, left_ear, right_ear = keypoints[:5]
    left_shoulder, right_shoulder, left_elbow, right_elbow = keypoints[5:9]
    left_wrist, right_wrist, left_hip, right_hip = keypoints[9:13]
    left_knee, right_knee, left_ankle, right_ankle = keypoints[13:17]

    detected_actions = []

    # Calculate key angles
    left_arm_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
    right_arm_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
    bending_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    leaning_angle = calculate_angle(left_shoulder, right_shoulder, right_hip)
    climbing_angle = calculate_angle(left_elbow, left_knee, left_ankle)
    left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # Detect Bending
    if bending_angle < 100:
        detected_actions.append("Bending")

    # Detect arm raising
    if left_arm_angle > 50:
        detected_actions.append("Left Arm Raised")
    if right_arm_angle > 50:
        detected_actions.append("Right Arm Raised")

    # Detect Running (based on knee height difference)
    if ((110 <= left_leg_angle <= 170 and right_knee[1] < left_knee[1]) or
        (110 <= right_leg_angle <= 170 and left_knee[1] < right_knee[1])) and \
            (50 <= left_arm_angle <= 140 or 50 <= right_arm_angle <= 140):
        detected_actions.append("Running")

    # Detect Lying on the Floor (if hips are close to the ground)
    hip_avg_y = (left_hip[1] + right_hip[1]) / 2
    shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
    if hip_avg_y < 0.8 * frame_height and shoulder_avg_y < 0.8 * frame_height:
        detected_actions.append("Lying on the Floor")

    # Detect Touching Face (if wrist is close to face)
    if (np.linalg.norm(left_wrist - nose) < 50 or np.linalg.norm(right_wrist - nose) < 50 or
            np.linalg.norm(left_wrist - left_eye) < 50 or np.linalg.norm(right_wrist - right_eye) < 50 or
            np.linalg.norm(left_wrist - left_ear) < 50 or np.linalg.norm(right_wrist - right_ear) < 50):
        detected_actions.append("Touching Face")

    # Detect Jumping (if ankles are above normal position)
    if left_ankle[1] > 0.1 * frame_height and right_ankle[1] > 0.1 * frame_height:
        detected_actions.append("Jumping")

    # Detect Leaning
    if leaning_angle < 80:
        detected_actions.append("Leaning Forward")
    elif leaning_angle > 100:
        detected_actions.append("Leaning Backward")

    # Detect Climbing
    if climbing_angle < 100 and abs(left_knee[1] - left_hip[1]) > 50:
        detected_actions.append("Climbing")

    return detected_actions if detected_actions else ["Standing"]


# Function to process video
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    stframe = st.empty()  # Placeholder for video display
    action_text = st.empty()  # Placeholder for detected actions

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run object detection
        obj_results = custom_model(frame)
        frame_with_boxes = obj_results[0].plot()  # Draw bounding boxes

        # Run pose estimation
        pose_results = pose_model(frame_with_boxes)

        for result in pose_results:
            if hasattr(result, "keypoints") and result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()

                for kp in keypoints:
                    actions = classify_pose(kp, frame_height)

                    # Display action text on video
                    y_offset = 50
                    for action in actions:
                        cv2.putText(frame_with_boxes, action, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                    2)
                        y_offset += 40

                    # Draw keypoints
                    for x, y in kp:
                        cv2.circle(frame_with_boxes, (int(x), int(y)), 5, (0, 255, 0), -1)

                    action_text.text(f"Detected Actions: {', '.join(actions)}")

        out.write(frame_with_boxes)

        # Convert to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    out.release()


# Streamlit UI
st.title("YOLO Object Detection & Pose Estimation 🎥")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    with st.spinner("Processing video... Please wait."):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        output_path = r"E:\project_data\Output_Folder"

        process_video(video_path, output_path)  # Process video

        # Cleanup temp file
        st.success("Processing completed!")

        with open(output_path, "rb") as file:
            st.download_button(label="Download Processed Video 🎬", data=file, file_name="output_video.mp4",
                               mime="video/mp4")
