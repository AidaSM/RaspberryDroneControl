import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import logging
import time
from dronekit import Vehicle, VehicleMode
from pymavlink import mavutil

logging.basicConfig(filename="follow_me.log", level=logging.INFO)
class PersonFollowDetector:
    def __init__(self, drone_control, camera, model_path="yolov5-person.tflite",
                 threshold=0.5, move_speed=0.3, dead_zone=40):
        self.drone = drone_control
        self.camera = camera
        self.vehicle = self.drone.vehicle
        self.threshold = threshold
        self.move_speed = move_speed
        self.dead_zone = dead_zone

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_width = self.input_details[0]['shape'][2]
        self.input_height = self.input_details[0]['shape'][1]

        self.no_detect_counter = 0
        self.detect_streak = 0

    def preprocess(self, frame):
        input_img = cv2.resize(frame, (self.input_width, self.input_height))
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, axis=0)
        return input_img

    def detect_person(self, frame):
        input_data = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        best_det = None
        best_conf = 0

        for det in output:
            x1, y1, x2, y2, conf, class_id = det[:6]
            if conf > self.threshold and int(class_id) == 0:
                if conf > best_conf:
                    best_conf = conf
                    best_det = (int(x1), int(y1), int(x2), int(y2), conf)

        if best_det:
            x1, y1, x2, y2, conf = best_det
            area = (x2 - x1) * (y2 - y1)
            return (x1, y1, x2, y2, conf, area)
        return None

    def follow(self):
        print("Starting person-following loop...")
        while True:
            frame = self.camera.get_frame()
            if frame is None:
                logging.error("Frame not captured.")
                continue

            start_time = time.time()
            h, w, _ = frame.shape
            cx_frame = w // 2

            detection = self.detect_person(frame)
            if detection:
                self.detect_streak += 1
                self.no_detect_counter = 0

                if self.detect_streak < 3:
                    continue

                x1, y1, x2, y2, conf, area = detection
                cx_person = (x1 + x2) // 2
                offset = cx_person - cx_frame

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                vy = 0
                if abs(offset) > self.dead_zone:
                    vy = np.clip((offset / cx_frame) * self.move_speed, -self.move_speed, self.move_speed)

                target_area = 10000
                vx = np.clip(((target_area - area) / target_area) * self.move_speed, -self.move_speed, self.move_speed)

                self.drone.send_velocity(vx, vy, 0, duration=0.1)
                logging.info(f"Person tracked at offset {offset}, area {area}")
            else:
                self.detect_streak = 0
                self.no_detect_counter += 1
                self.drone.send_velocity(0, 0, 0, duration=0.1)

                if self.no_detect_counter > 100:
                    logging.warning("Person lost for too long. Exiting.")
                    self.safe_shutdown()
                    break

            fps = 1.0 / (time.time() - start_time + 1e-6)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Follow Me", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.safe_shutdown()
                break

        cv2.destroyAllWindows()

    def safe_shutdown(self):
        self.drone.send_velocity(0, 0, 0)
        self.vehicle.mode = VehicleMode("LOITER")
        logging.info("Switched to LOITER mode for safety.")
