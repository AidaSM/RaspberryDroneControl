import time
import os
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

class GeoTrigger:
    def __init__(self, drone, camera, target_lat, target_lon, radius=5.0, capture_mode='image'):
        self.drone = drone
        self.camera = camera
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.radius = radius
        self.capture_mode = capture_mode  # 'image' or 'video'

        # Ensure directories exist
        if self.capture_mode == 'image':
            os.makedirs("images", exist_ok=True)
        elif self.capture_mode == 'video':
            os.makedirs("videos", exist_ok=True)

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        return R * c

    def monitor_and_capture(self):
        print(f"[INFO] Monitoring position. Target radius: {self.radius:.1f} m")

        while True:
            location = self.drone.vehicle.location.global_relative_frame
            distance = self.haversine_distance(
                location.lat, location.lon,
                self.target_lat, self.target_lon
            )

            print(f"[INFO] Distance to target: {distance:.2f} m")

            if distance <= self.radius:
                #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if self.capture_mode == 'image':
                    filename = f"images/capture.jpg"
                    self.camera.capture_image(filename)
                elif self.capture_mode == 'video':
                    filename = f"videos/record.h264"
                    self.camera.start_recording(filename)
                    time.sleep(5)  # record 5 seconds
                    self.camera.stop_recording()

                time.sleep(10)  # Wait before next capture

            else:
                time.sleep(1)  # Check again after a second
