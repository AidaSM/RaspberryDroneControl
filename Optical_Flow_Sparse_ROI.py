import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# === Setup working directories (optional) ===
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)

# === Initialize camera ===
cap = cv2.VideoCapture(0)  # 0 = default camera
if not cap.isOpened():
    raise Exception("Camera not accessible")

# Warm-up: grab a few frames to stabilize video feed
for _ in range(10):
    cap.read()

# === Shi-Tomasi corner detection parameters ===
shitomasi_params = dict(
    maxCorners=100,         # Maximum number of features
    qualityLevel=0.5,       # Minimum quality of corners
    minDistance=7           # Minimum distance between corners
)

# === Lucas-Kanade Optical Flow parameters ===
lk_params = dict(
    winSize=(15, 15),       # Window size for searching
    maxLevel=2,             # Number of pyramid levels
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# === Read the first frame and extract initial feature points ===
ret, frame = cap.read()
if not ret:
    raise Exception("Failed to read from camera")

frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.goodFeaturesToTrack(frame_gray_init, mask=None, **shitomasi_params)
mask = np.zeros_like(frame)  # Used for drawing flow vectors

# === Define the Region of Interest (ROI) ===
frame_height, frame_width = frame.shape[:2]
middle_x, middle_y = frame_width // 2, frame_height // 2
roi = [middle_x - 180, middle_y - 150, middle_x + 160, middle_y + 250]  # [x1, y1, x2, y2]

# === Motion detection threshold ===
displacement_threshold = 5

# === Lists to store displacement values for plotting ===
avg_displacement_x_list = []
avg_displacement_y_list = []
displacement_x_list = []
displacement_y_list = []

frame_number = 0  # Frame counter

# === Main processing loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reinitialize tracking points if lost
    if edges is None or len(edges) < 5:
        print("Reinitializing features...")
        edges = cv2.goodFeaturesToTrack(frame_gray, mask=None, **shitomasi_params)
        frame_gray_init = frame_gray.copy()
        continue

    # === Calculate optical flow using Lucas-Kanade method ===
    new_edges, status, errors = cv2.calcOpticalFlowPyrLK(
        frame_gray_init, frame_gray, edges, None, **lk_params)

    if new_edges is None or status is None:
        print("Optical flow failed — skipping frame.")
        continue

    good_old = edges[status == 1]
    good_new = new_edges[status == 1]

    if len(good_new) == 0:
        print("No good points tracked — skipping frame.")
        continue

    # === Overlay frame info and ROI ===
    cv2.putText(frame, f"Frame: {frame_number}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 0), 2)

    obstacle_detected = False

    # === Analyze displacement of each tracked point ===
    for new, old in zip(good_new, good_old):
        x1, y1 = new.ravel()
        x2, y2 = old.ravel()
        dx = x1 - x2
        dy = y1 - y2

        if abs(dx) > displacement_threshold or abs(dy) > displacement_threshold:
            obstacle_detected = True
            mask = cv2.arrowedLine(mask, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(x1), int(y1)), 3, (0, 255, 0), thickness=-1)
            displacement_x_list.append(dx)
            displacement_y_list.append(dy)

    # === Check average motion in the ROI only ===
    roi_corners = good_new[
        (good_new[:, 0] >= roi[0]) & (good_new[:, 1] >= roi[1]) &
        (good_new[:, 0] <= roi[2]) & (good_new[:, 1] <= roi[3])
    ]

    if len(roi_corners) > 2:
        avg_dx = np.mean(roi_corners[:, 0] - roi_corners[0, 0])
        avg_dy = np.mean(roi_corners[:, 1] - roi_corners[0, 1])
        avg_displacement_x_list.append(avg_dx)
        avg_displacement_y_list.append(avg_dy)

        if abs(avg_dx) > displacement_threshold or abs(avg_dy) > displacement_threshold:
            obstacle_detected = True
            for corner in roi_corners:
                x, y = corner.ravel()
                frame = cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 0), thickness=-1)

    # === Display results ===
    output = cv2.add(frame, mask)
    cv2.imshow('Obstacle Detection', output)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Update reference for next frame
    frame_gray_init = frame_gray.copy()
    edges = good_new.reshape(-1, 1, 2)

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()

# === Plot displacement statistics ===
# plt.plot(displacement_x_list, label='Displacement X')
# plt.plot(displacement_y_list, label='Displacement Y')
# plt.plot(avg_displacement_x_list, label='Average Displacement X')
# plt.plot(avg_displacement_y_list, label='Average Displacement Y')
# plt.xlabel('Frame')
# plt.ylabel('Displacement')
# plt.legend()
# plt.title('Motion Displacement Over Time')
# plt.show()
