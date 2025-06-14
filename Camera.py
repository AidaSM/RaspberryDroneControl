from picamera2 import Picamera2, Preview
import time
import os

class Camera:
    def __init__(self, resolution=(1920, 1080), framerate=30):
        self.picam2 = Picamera2()
        self.resolution = resolution
        self.framerate = framerate
        self.configure_camera()

    def configure_camera(self):
        camera_config = self.picam2.create_still_configuration(
            main={"size": self.resolution}
        )
        self.picam2.configure(camera_config)

    def start_preview(self, fullscreen=True, window=None):
        self.picam2.start_preview(Preview.QT if fullscreen else Preview.NULL)
        self.picam2.start()
        print("Preview started...")

    def stop_preview(self):
        self.picam2.stop_preview()
        self.picam2.stop()
        print("Preview stopped.")

    def capture_image(self, output_path="image.jpg"):
        self.picam2.capture_file(output_path)
        print(f"Image saved to {output_path}")

    def start_recording(self, output_path="video.h264"):
        video_config = self.picam2.create_video_configuration(main={"size": self.resolution, "fps": self.framerate})
        self.picam2.configure(video_config)
        self.picam2.start_recording(output_path)
        print(f"Recording started to {output_path}")

    def stop_recording(self):
        self.picam2.stop_recording()
        print("Recording stopped.")

# Example usage
if __name__ == "__main__":
    camera = Camera()
    camera.start_preview()
    time.sleep(2)  # Let the camera warm up
    camera.capture_image("test.jpg")
    camera.stop_preview()
