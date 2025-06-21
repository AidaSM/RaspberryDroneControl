import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import logging
import time

from Camera import Camera as PiCamera
from DroneControl import DroneControl

logging.basicConfig(level=logging.INFO)


class PersonDetector:
    def __init__(self, camera, model_path="detect.tflite", threshold=0.5):
        self.camera = camera
        self.threshold = threshold

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

    def preprocess(self, frame):
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        return np.expand_dims(resized, axis=0).astype(np.uint8)

    def detect_person(self, frame):
        h, w, _ = frame.shape
        input_data = self.preprocess(frame)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        results = []
        for i in range(len(scores)):
            if scores[i] > self.threshold and int(classes[i]) == 0:
                ymin, xmin, ymax, xmax = boxes[i]
                x1, y1, x2, y2 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
                results.append((x1, y1, x2, y2, scores[i]))
        return results


class PersonFollowDetector(PersonDetector):
    def __init__(self, camera, drone_control, model_path="detect.tflite", threshold=0.5,
                 move_speed=0.3, dead_zone=80):
        super().__init__(camera, model_path, threshold)
        self.drone = drone_control
        self.vehicle = self.drone
        self.move_speed = move_speed
        self.dead_zone = dead_zone
        self.no_detect_counter = 0
        self.detect_streak = 0

    def detect_person(self, frame):
        detections = super().detect_person(frame)
        if not detections:
            return None
        best = max(detections, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
        x1, y1, x2, y2, conf = best
        area = (x2 - x1) * (y2 - y1)
        return (x1, y1, x2, y2, conf, area)

    def follow(self):
        print("Starting person-following simulation. Press 'q' to quit.")
        video_writer = None

        while True:
            frame = self.camera.get_frame()
            if frame is None:
                logging.error("Frame not captured.")
                continue

            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

            start_time = time.time()
            h, w, _ = frame.shape
            cx_frame = w // 2
            cy_frame = h // 2

            box_width = self.dead_zone * 2
            box_height = self.dead_zone * 2
            x_dead1 = cx_frame - self.dead_zone
            x_dead2 = cx_frame + self.dead_zone
            y_dead1 = cy_frame - self.dead_zone
            y_dead2 = cy_frame + self.dead_zone
            cv2.rectangle(frame, (x_dead1, y_dead1), (x_dead2, y_dead2), (0, 0, 255), 2)

            detection = self.detect_person(frame)
            if detection:
                self.detect_streak += 1
                self.no_detect_counter = 0

                if self.detect_streak < 3:
                    continue

                x1, y1, x2, y2, conf, area = detection
                cx_person = (x1 + x2) // 2
                cy_person = (y1 + y2) // 2
                offset_x = cx_person - cx_frame
                offset_y = cy_person - cy_frame

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                vy = 0
                vx = 0

                if cx_person < x_dead1 or cx_person > x_dead2:
                    vy = np.clip((offset_x / cx_frame) * self.move_speed, -self.move_speed, self.move_speed)

                target_area = 120000
                vx = np.clip(((target_area - area) / target_area) * self.move_speed, -self.move_speed, self.move_speed)

                self.drone.send_velocity(vx, vy, 0, duration=0.1)
                print(f"[COMMAND] Sent velocity ? vx: {vx:.2f}, vy: {vy:.2f}, duration: 0.1")
                logging.info(f"Person tracked. Offset: {offset_x}, Area: {area}")
            else:
                self.detect_streak = 0
                self.no_detect_counter += 1
                self.drone.send_velocity(0, 0, 0, duration=0.1)
                print("[COMMAND] Sent stop velocity (vx: 0, vy: 0)")

                if self.no_detect_counter > 100:
                    logging.warning("Person lost for too long. Exiting.")
                    self.safe_shutdown()
                    break

            fps = 1.0 / (time.time() - start_time + 1e-6)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            video_writer.write(frame)
            cv2.imshow("Follow Me (Simulated)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.safe_shutdown()
                break

        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

    def safe_shutdown(self):
        print("[SIM] Safe shutdown triggered.")
        self.drone.send_velocity(0, 0, 0)
        self.vehicle.set_mode("LOITER")


if __name__ == "__main__":
    cam = PiCamera()
    drone = DroneControl()

    print("Starting camera preview. Press 'q' to continue to detection...")
    while True:
        frame = cam.get_frame()
        if frame is not None:
            cv2.imshow("Camera Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    detector = FollowPersonDetector(cam, drone, model_path="detect.tflite")
    detector.follow()
