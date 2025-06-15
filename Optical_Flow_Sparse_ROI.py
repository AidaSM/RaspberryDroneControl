import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# === Setare directoare pentru fișiere video ===
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)

# === Inițializare cameră ===
cap = cv2.VideoCapture(0)  # 0 = camera implicită
if not cap.isOpened():
    raise Exception("Camera not accessible")

# Warming-up: se citesc câteva cadre pentru a stabiliza fluxul video
for _ in range(10):
    cap.read()

# === Parametri Shi-Tomasi pentru detecția colțurilor ===
shitomasi_params = dict(
    maxCorners=100,         # număr maxim de puncte de interes
    qualityLevel=0.5,       # calitatea minimă a colțului (0.0–1.0)
    minDistance=7           # distanța minimă între colțuri
)

# === Parametri Lucas-Kanade pentru urmărirea fluxului optic ===
lk_params = dict(
    winSize=(15, 15),       # dimensiunea ferestrei de căutare
    maxLevel=2,             # numărul nivelurilor în piramida Gaussiană
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# === Citirea primului cadru și extragerea colțurilor inițiale ===
ret, frame = cap.read()
if not ret:
    raise Exception("Failed to read from camera")

frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.goodFeaturesToTrack(frame_gray_init, mask=None, **shitomasi_params)
mask = np.zeros_like(frame)  # pentru desenarea vectorilor

# === Definirea zonei de interes (ROI) pentru decizie ===
frame_height, frame_width = frame.shape[:2]
middle_x, middle_y = frame_width // 2, frame_height // 2
roi = [middle_x - 180, middle_y - 150, middle_x + 160, middle_y + 250]

# Prag de detecție pentru mișcare semnificativă
displacement_threshold = 5

# Liste pentru stocarea deplasărilor
avg_displacement_x_list = []
avg_displacement_y_list = []
displacement_x_list = []
displacement_y_list = []

frame_number = 0  # Contor cadru

# === Buclă principală ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reinițializare puncte dacă se pierd
    if edges is None or len(edges) < 5:
        print("Reinitializing features...")
        edges = cv2.goodFeaturesToTrack(frame_gray, mask=None, **shitomasi_params)
        frame_gray_init = frame_gray.copy()
        continue

    # === Calculul fluxului optic (Lucas-Kanade) ===
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

    # === Suprapunere: număr cadru și ROI ===
    cv2.putText(frame, f"Frame: {frame_number}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 0), 2)

    obstacle_detected = False  # Indicator obstacol

    # === Analiză deplasări punctuale ===
    for new, old in zip(good_new, good_old):
        x1, y1 = new.ravel()
        x2, y2 = old.ravel()
        dx = x1 - x2
        dy = y1 - y2

        # Detectare mișcare semnificativă
        if abs(dx) > displacement_threshold or abs(dy) > displacement_threshold:
            obstacle_detected = True
            mask = cv2.arrowedLine(mask, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(x1), int(y1)), 3, (0, 255, 0), thickness=-1)
            displacement_x_list.append(dx)
            displacement_y_list.append(dy)

    # === Evaluarea mișcării în zona ROI ===
    roi_corners = good_new[
        (good_new[:, 0] >= roi[0]) & (good_new[:, 1] >= roi[1]) &
        (good_new[:, 0] <= roi[2]) & (good_new[:, 1] <= roi[3])
    ]

    if len(roi_corners) > 2:
        avg_dx = np.mean(roi_corners[:, 0] - roi_corners[0, 0])
        avg_dy = np.mean(roi_corners[:, 1] - roi_corners[0, 1])
        avg_displacement_x_list.append(avg_dx)
        avg_displacement_y_list.append(avg_dy)

        # Semnalizare obstacol în ROI
        if abs(avg_dx) > displacement_threshold or abs(avg_dy) > displacement_threshold:
            obstacle_detected = True
            for corner in roi_corners:
                x, y = corner.ravel()
                frame = cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 0), thickness=-1)

    # === Combinare mască + cadru pentru vizualizare ===
    output = cv2.add(frame, mask)
    cv2.imshow('Obstacle Detection', output)

    # Ieșire la apăsarea tastei 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Actualizare referințe pentru următorul cadru
    frame_gray_init = frame_gray.copy()
    edges = good_new.reshape(-1, 1, 2)

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()

# === Vizualizare grafică a deplasărilor detectate ===
plt.plot(displacement_x_list, label='Displacement X')
plt.plot(displacement_y_list, label='Displacement Y')
plt.plot(avg_displacement_x_list, label='Average Displacement X')
plt.plot(avg_displacement_y_list, label='Average Displacement Y')
plt.xlabel('Frame')
plt.ylabel('Displacement')
plt.legend()
plt.show()
