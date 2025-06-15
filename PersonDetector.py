import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

class PersonDetector:
    def __init__(self, model_path="yolov5-person.tflite", threshold=0.5, camera_index=0):
        self.threshold = threshold
        self.cap = cv2.VideoCapture(camera_index)
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Dimensiuni așteptate de model
        self.input_width = self.input_details[0]['shape'][2]
        self.input_height = self.input_details[0]['shape'][1]

    def preprocess(self, frame):
        input_img = cv2.resize(frame, (self.input_width, self.input_height))
        input_img = np.expand_dims(input_img, axis=0).astype(np.uint8)
        return input_img

    def detect(self, frame):
        input_data = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        detections = []
        for det in output:
            confidence = det[4]
            if confidence > self.threshold:
                x1, y1, x2, y2 = map(int, det[:4])
                detections.append((x1, y1, x2, y2, confidence))
        return detections

    def run(self, display=True):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            detections = self.detect(frame)
            for x1, y1, x2, y2, conf in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if display:
                cv2.imshow("Person Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
