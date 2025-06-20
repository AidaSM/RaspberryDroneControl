import cv2
import numpy as np

class ObstacleAvoidance:
    def __init__(self, camera_index=0, width=320, height=240):
        """
        Initializes the obstacle avoidance system using optical flow and color clustering.
        - camera_index: index of the camera (default 0)
        - width, height: frame dimensions for processing
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Camera nu poate fi accesat.")  # Camera not accessible

        self.width = width
        self.height = height

        # Parameters for Lucas-Kanade Optical Flow
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Parameters for Shi-Tomasi corner detection
        self.shi_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)

        # Criteria for K-means clustering
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        self.K = 2  # Number of clusters

        self._init_tracking()  # Initialize with first frame and tracking points

    def _init_tracking(self):
        """
        Captures the first frame, applies K-means clustering to identify a mask,
        and detects good features to track.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Nu s-a putut citi primul cadru.")  # Could not read first frame

        # Resize and preprocess frame
        self.prev_frame = cv2.resize(frame, (self.width, self.height))
        rgb = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2RGB)
        pixels = rgb.reshape((-1, 3)).astype(np.float32)

        # Apply K-means clustering to segment image
        _, labels, _ = cv2.kmeans(pixels, self.K, None, self.criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels_img = labels.reshape((self.height, self.width))

        # Select dominant cluster as the object mask
        object_cluster = np.argmax(np.bincount(labels.flatten()))
        self.mask_kmeans = np.uint8((labels_img == object_cluster) * 255)

        # Convert to grayscale and detect initial features
        gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=self.mask_kmeans, **self.shi_params)
        self.prev_gray = gray

    def check_and_avoid(self):
        """
        Checks for motion between frames using optical flow.
        If significant motion is detected, returns True (obstacle detected).
        """
        ret, frame = self.cap.read()
        if not ret:
            return False

        # Resize and convert to grayscale
        frame_small = cv2.resize(frame, (self.width, self.height))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # If we lose tracking points, reinitialize them
        if self.prev_pts is None or len(self.prev_pts) < 5:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=self.mask_kmeans, **self.shi_params)
            self.prev_gray = gray.copy()
            return False

        # Calculate optical flow between previous and current frame
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)

        if next_pts is not None and status is not None:
            good_old = self.prev_pts[status == 1]
            good_new = next_pts[status == 1]

            # Check for significant motion in any tracked point
            for new, old in zip(good_new, good_old):
                dx, dy = new.ravel() - old.ravel()
                if abs(dx) > 2 or abs(dy) > 2:
                    print("[AVOID] Obstacle detected.")
                    self.prev_gray = gray.copy()
                    self.prev_pts = good_new.reshape(-1, 1, 2)
                    return True

        # Update tracking state for next iteration
        self.prev_gray = gray.copy()
        self.prev_pts = next_pts.reshape(-1, 1, 2) if next_pts is not None else None
        return False

    def release(self):
        """
        Releases the camera and closes any OpenCV windows.
        """
        self.cap.release()
        cv2.destroyAllWindows()
