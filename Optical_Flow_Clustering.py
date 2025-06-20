import numpy as np
import cv2

# --- Configuration ---
K = 2  # Number of color clusters to separate the image into
resize_width = 320  # Width for resizing frames (for speed)
resize_height = 240  # Height for resizing frames
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # K-means stopping criteria
attempts = 10  # Number of times k-means algorithm will be executed using different initial labellings

# --- Open webcam ---
cap = cv2.VideoCapture(0)  # Use default webcam (camera index 0)
if not cap.isOpened():
    raise Exception("Could not open webcam.")

print("Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize the frame for faster processing
    frame_small = cv2.resize(frame, (resize_width, resize_height))

    # Convert the image from BGR to RGB (since OpenCV loads in BGR)
    img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    # Flatten the image to a 2D array of pixels for k-means input
    pixel_data = img_rgb.reshape((-1, 3)).astype(np.float32)

    # Apply k-means clustering to segment colors
    _, labels, centers = cv2.kmeans(pixel_data, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    # Convert centers to uint8 and reconstruct segmented image
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    result_image = segmented.reshape(img_rgb.shape)

    # Convert back to BGR for OpenCV display
    result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    # Stack original and segmented images side-by-side for comparison
    combined = np.hstack((cv2.resize(frame, (resize_width, resize_height)), result_bgr))
    cv2.imshow("Original (Left) | Segmented (Right)", combined)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close any OpenCV windows
