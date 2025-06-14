import cv2
import numpy as np

class DroneVision:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Camera not accessible")

    def process_frame(self, frame):
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours (could indicate obstacles)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If large contours are found, assume obstacle
        obstacle_detected = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Tune this threshold
                obstacle_detected = True
                cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

        return frame, obstacle_detected

    def run(self):
        print("Starting vision system. Press 'q' to quit.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame, obstacle = self.process_frame(frame)
            if obstacle:
                print("🚨 Obstacle Detected!")

            cv2.imshow("Drone Vision", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    vision = DroneVision()
    vision.run()
