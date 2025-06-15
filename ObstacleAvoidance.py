import cv2
import numpy as np

class ObstacleAvoidance:
    def __init__(self, camera_index=0, width=320, height=240):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Camera nu poate fi accesat.")

        self.width = width
        self.height = height

        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.shi_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        self.K = 2

        self._init_tracking()

    def _init_tracking(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Nu s-a putut citi primul cadru.")

        self.prev_frame = cv2.resize(frame, (self.width, self.height))
        rgb = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2RGB)
        pixels = rgb.reshape((-1, 3)).astype(np.float32)

        _, labels, _ = cv2.kmeans(pixels, self.K, None, self.criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels_img = labels.reshape((self.height, self.width))

        object_cluster = np.argmax(np.bincount(labels.flatten()))
        self.mask_kmeans = np.uint8((labels_img == object_cluster) * 255)

        gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=self.mask_kmeans, **self.shi_params)
        self.prev_gray = gray

    def check_and_avoid(self):
        ret, frame = self.cap.read()
        if not ret:
            return False

        frame_small = cv2.resize(frame, (self.width, self.height))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        if self.prev_pts is None or len(self.prev_pts) < 5:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=self.mask_kmeans, **self.shi_params)
            self.prev_gray = gray.copy()
            return False

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)

        if next_pts is not None and status is not None:
            good_old = self.prev_pts[status == 1]
            good_new = next_pts[status == 1]

            for new, old in zip(good_new, good_old):
                dx, dy = new.ravel() - old.ravel()
                if abs(dx) > 2 or abs(dy) > 2:
                    print("[AVOID] Obstacle detected.")
                    self.prev_gray = gray.copy()
                    self.prev_pts = good_new.reshape(-1, 1, 2)
                    return True

        self.prev_gray = gray.copy()
        self.prev_pts = next_pts.reshape(-1, 1, 2) if next_pts is not None else None
        return False

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()