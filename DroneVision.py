import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from Camera import Camera

# === Setup ===
os.makedirs("Output", exist_ok=True)
output_video_path = "Output/output_opticalflow.avi"
output_csv_path = "Output/displacements.csv"
output_plot_path = "Output/displacement_plot.png"

# === Ini?ializare camera ===
camera = Camera(resolution=(640, 480), framerate=30)
camera.start_preview(fullscreen=False)
time.sleep(1)

# Parametri
K = 2
shi_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
criteria_kmeans = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Captura prim cadru
frame = camera.get_frame()
print("Cadru capturat:", frame.shape)

h, w = frame.shape[:2]
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pixels = rgb.reshape((-1, 3)).astype(np.float32)

_, labels, centers = cv2.kmeans(pixels, K, None, criteria_kmeans, 10, cv2.KMEANS_PP_CENTERS)
labels_img = labels.reshape((h, w))
object_cluster = np.argmin(np.bincount(labels.flatten()))
mask_kmeans = np.uint8((labels_img == object_cluster) * 255)

prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask_kmeans, **shi_params)
prev_gray = gray.copy()
mask = np.zeros_like(frame)

# Stocare date deplasare
disp_x, disp_y = [], []
frame_count = 0

while True:
    frame = camera.get_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria_kmeans, 10, cv2.KMEANS_PP_CENTERS)
    labels_img = labels.reshape((h, w))
    mask_kmeans = np.uint8((labels_img == object_cluster) * 255)

    if prev_pts is None or len(prev_pts) < 5:
        prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask_kmeans, **shi_params)
        prev_gray = gray.copy()
        continue

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

    if next_pts is not None and status is not None:
        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]

        dx_total, dy_total = 0, 0
        valid_vectors = 0

        for new, old in zip(good_new, good_old):
            x1, y1 = new.ravel()
            x2, y2 = old.ravel()
            dx, dy = x1 - x2, y1 - y2
            if abs(dx) > 1 or abs(dy) > 1:
                mask = cv2.arrowedLine(mask, (int(x2), int(y2)), (int(x1), int(y1)), (0, 255, 0), 1)
                frame = cv2.circle(frame, (int(x1), int(y1)), 2, (0, 0, 255), -1)
                dx_total += dx
                dy_total += dy
                valid_vectors += 1

        if valid_vectors > 0:
            avg_dx = dx_total / valid_vectors
            avg_dy = dy_total / valid_vectors
            disp_x.append(avg_dx)
            disp_y.append(avg_dy)
        else:
            disp_x.append(0)
            disp_y.append(0)

        output = cv2.add(frame, mask)
        video_writer.write(output)
        cv2.imshow("Flux optic segmentat", output)
        cv2.startWindowThread()

        frame_count += 1

        prev_gray = gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
camera.stop_preview()
video_writer.release()
cv2.destroyAllWindows()

# === Salvare date ?i grafic ===
df = pd.DataFrame({
    "Frame": list(range(len(disp_x))),
    "Displacement_X": disp_x,
    "Displacement_Y": disp_y
})
df.to_csv(output_csv_path, index=False)

plt.figure()
plt.plot(disp_x, label="Displacement X")
plt.plot(disp_y, label="Displacement Y")
plt.xlabel("Cadru")
plt.ylabel("Deplasare medie")
plt.legend()
plt.title("Evolu?ia deplasarii xn timp")
plt.savefig(output_plot_path)
plt.show()
