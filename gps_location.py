import cv2
import numpy as np

cap = cv2.VideoCapture(0)
K = 2  # Număr clustere

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_small = cv2.resize(frame, (320, 240))
    img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    segmented = centers[labels.flatten()].reshape(img_rgb.shape).astype(np.uint8)
    combined = np.hstack((frame_small, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)))

    cv2.imshow("K-means Segmentare (Original | Segmentat)", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
