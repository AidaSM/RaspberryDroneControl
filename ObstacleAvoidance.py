import cv2
import numpy as np
import time
from dronekit import connect, VehicleMode
from picamera2 import Picamera2


class ObstacleAvoidance:
    def __init__(self):
        print("Connecting to ArduPilot...")
        self.vehicle = connect('/dev/serial0', baud=115200, wait_ready=True)
        print("Connected to vehicle.")

        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(
            main={"format": "BGR888", "size": (640, 480)}))
        self.picam2.start()
        time.sleep(1)

        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.displacement_threshold = 5

        self.mode_changed = False
        self.last_mode = self.vehicle.mode.name

        self.init_first_frame()

    def init_first_frame(self):
        frame = self.picam2.capture_array()
        if frame is None:
            raise Exception("Can't read from camera")
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)

    def run(self):
        print("Starting vision-based obstacle detection. Press 'q' to quit.")
        no_motion_start_time = None
        clear_delay = 5  # seconds of confirmed "clear path" before resuming

        try:
            while True:
                frame = self.picam2.capture_array()
                if frame is None:
                    continue

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if self.p0 is None or len(self.p0) < 10:
                    self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
                    self.old_gray = frame_gray.copy()
                    continue

                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    self.old_gray, frame_gray, self.p0, None, **self.lk_params)

                if p1 is None or st is None:
                    continue

                good_new = p1[st == 1]
                good_old = self.p0[st == 1]

                obstacle_detected = False
                for new, old in zip(good_new, good_old):
                    dx = new[0] - old[0]
                    dy = new[1] - old[1]
                    if abs(dx) > self.displacement_threshold or abs(dy) > self.displacement_threshold:
                        obstacle_detected = True
                        break

                current_time = time.time()

                if obstacle_detected:
                    no_motion_start_time = None
                    if not self.mode_changed:
                        print("Obstacle detected! Switching to BRAKE mode.")
                        self.vehicle.mode = VehicleMode("BRAKE")
                        self.mode_changed = True

                else:
                    if self.mode_changed:
                        if no_motion_start_time is None:
                            no_motion_start_time = current_time
                        elif (current_time - no_motion_start_time) >= clear_delay:
                            print(f"Path clear for {clear_delay} seconds. Resuming mode:", self.last_mode)
                            self.vehicle.mode = VehicleMode(self.last_mode)
                            self.mode_changed = False
                            no_motion_start_time = None

                # Draw optical flow
                for new, old in zip(good_new, good_old):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 1)

                cv2.imshow("Obstacle Detection View", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                self.old_gray = frame_gray.copy()
                self.p0 = good_new.reshape(-1, 1, 2)

        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            self.shutdown()

    def shutdown(self):
        self.picam2.stop()
        self.vehicle.close()
        cv2.destroyAllWindows()
        print("System shut down cleanly.")

