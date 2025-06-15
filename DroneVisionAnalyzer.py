import cv2
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from threading import Thread

class DroneVisionAnalyzer(Thread):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = True

        self.K = 2
        self.shi_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.criteria_kmeans = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        self.disp_x = []
        self.disp_y = []

    def run(self):
        frame = self.camera.get_frame()
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixels = rgb.reshape((-1, 3)).astype(np.float32)

        _, labels, _ = cv2.kmeans(pixels, self.K, None, self.criteria_kmeans, 10, cv2.KMEANS_PP_CENTERS)
        labels_img = labels.reshape((h, w))
        object_cluster = np.argmin(np.bincount(labels.flatten()))
        mask_kmeans = np.uint8((labels_img == object_cluster) * 255)

        prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask_kmeans, **self.shi_params)
        prev_gray = gray.copy()
        mask = np.zeros_like(frame)

        while self.running:
            frame = self.camera.get_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pixels = rgb.reshape((-1, 3)).astype(np.float32)

            _, labels, _ = cv2.kmeans(pixels, self.K, None, self.criteria_kmeans, 10, cv2.KMEANS_PP_CENTERS)
            labels_img = labels.reshape((h, w))
            mask_kmeans = np.uint8((labels_img == object_cluster) * 255)

            if prev_pts is None or len(prev_pts) < 5:
                prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask_kmeans, **self.shi_params)
                prev_gray = gray.copy()
                continue

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **self.lk_params)

            if next_pts is not None and status is not None:
                good_new = next_pts[status == 1]
                good_old = prev_pts[status == 1]

                dx_total, dy_total, valid_vectors = 0, 0, 0

                for new, old in zip(good_new, good_old):
                    dx, dy = new.ravel() - old.ravel()
                    if abs(dx) > 1 or abs(dy) > 1:
                        dx_total += dx
                        dy_total += dy
                        valid_vectors += 1

                if valid_vectors > 0:
                    self.disp_x.append(dx_total / valid_vectors)
                    self.disp_y.append(dy_total / valid_vectors)
                else:
                    self.disp_x.append(0)
                    self.disp_y.append(0)

                prev_gray = gray.copy()
                prev_pts = good_new.reshape(-1, 1, 2)

            time.sleep(0.03)  # ~30 FPS

    def stop(self):
        self.running = False
        self.join()

        df = pd.DataFrame({
            "Frame": list(range(len(self.disp_x))),
            "Displacement_X": self.disp_x,
            "Displacement_Y": self.disp_y
        })
        df.to_csv("Output/displacements.csv", index=False)

        plt.figure()
        plt.plot(self.disp_x, label="Displacement X")
        plt.plot(self.disp_y, label="Displacement Y")
        plt.xlabel("Frame")
        plt.ylabel("Avg Displacement")
        plt.legend()
        plt.title("Motion Analysis")
        plt.savefig("Output/displacement_plot.png")
        plt.close()
