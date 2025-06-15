import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from scipy.spatial import distance as dist

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Verified MediaPipe landmark indices [2][9]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Smoothing parameters
EAR_SMOOTHING_WINDOW = 5  # Number of frames for moving average
left_ear_queue = deque(maxlen=EAR_SMOOTHING_WINDOW)
right_ear_queue = deque(maxlen=EAR_SMOOTHING_WINDOW)

def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Dynamic threshold parameters
BASE_EAR_THRESHOLD = 0.21
DYNAMIC_THRESHOLD_FACTOR = 0.75
calibration_frames = 30
calibration_values = []

# Blink detection parameters
CONSEC_FRAMES = 3
left_counter = right_counter = both_counter = 0
left_blinks = right_blinks = both_blinks = 0

cap = cv2.VideoCapture(0)

# Calibration phase
print("Calibrating... Keep eyes open and face still")
while len(calibration_values) < calibration_frames:
    ret, frame = cap.read()
    if not ret:
        continue
    
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        left_eye = [(int(face_landmarks.landmark[i].x * w), 
                    int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(face_landmarks.landmark[i].x * w), 
                     int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
        
        left_ear = calculate_EAR(left_eye)
        right_ear = calculate_EAR(right_eye)
        calibration_values.append((left_ear + right_ear) / 2)

dynamic_threshold = np.mean(calibration_values) * DYNAMIC_THRESHOLD_FACTOR
print(f"Calibration complete. Dynamic threshold: {dynamic_threshold:.2f}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        left_eye = [(int(face_landmarks.landmark[i].x * w), 
                    int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(face_landmarks.landmark[i].x * w), 
                     int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

        # Calculate and smooth EAR values
        left_ear = calculate_EAR(left_eye)
        right_ear = calculate_EAR(right_eye)
        left_ear_queue.append(left_ear)
        right_ear_queue.append(right_ear)
        
        smoothed_left_ear = np.mean(left_ear_queue)
        smoothed_right_ear = np.mean(right_ear_queue)

        # Eye state detection
        left_closed = smoothed_left_ear < dynamic_threshold
        right_closed = smoothed_right_ear < dynamic_threshold

        # Blink detection logic
        if left_closed and right_closed:
            both_counter += 1
        else:
            if both_counter >= CONSEC_FRAMES:
                both_blinks += 1
            both_counter = 0

        if left_closed and not right_closed:
            left_counter += 1
        else:
            if left_counter >= CONSEC_FRAMES:
                left_blinks += 1
            left_counter = 0

        if right_closed and not left_closed:
            right_counter += 1
        else:
            if right_counter >= CONSEC_FRAMES:
                right_blinks += 1
            right_counter = 0

        # Visualization
        cv2.putText(frame, f"Left EAR: {smoothed_left_ear:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(frame, f"Right EAR: {smoothed_right_ear:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"Left: {left_blinks}", (w-200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(frame, f"Right: {right_blinks}", (w-200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"Both: {both_blinks}", (w-200, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow('Stable Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
