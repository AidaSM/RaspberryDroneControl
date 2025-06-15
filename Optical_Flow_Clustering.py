import numpy as np
import cv2

# --- Configuration ---
K = 2  # Number of color clusters
resize_width = 320  # Width to resize frames for faster processing
resize_height = 240
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
attempts = 10

# --- Open webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open webcam.")

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize for speed
    frame_small = cv2.resize(frame, (resize_width, resize_height))

    # Convert to RGB
    img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    # Flatten image
    pixel_data = img_rgb.reshape((-1, 3)).astype(np.float32)

    # Apply k-means clustering
    _, labels, centers = cv2.kmeans(pixel_data, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    # Reconstruct segmented image
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    result_image = segmented.reshape(img_rgb.shape)

    # Convert back to BGR for OpenCV display
    result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    # Show original and segmented side-by-side
    combined = np.hstack((cv2.resize(frame, (resize_width, resize_height)), result_bgr))
    cv2.imshow("Original (Left) | Segmented (Right)", combined)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
