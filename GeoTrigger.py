import time
import os
from datetime import datetime
from math import radians, cos, sin, asin, sqrt


class GeoTrigger:
    def __init__(self, drone, camera, target_lat, target_lon, radius=5.0, capture_mode='image'):
        """
        Initialize a geographic trigger that activates when the drone enters a specified radius
        around a GPS coordinate and captures an image or video using the attached camera.

        Parameters:
        - drone: an instance of DroneControl (must have .vehicle with GPS access)
        - camera: an instance of the Camera class (must support capture_image and start/stop_recording)
        - target_lat, target_lon: coordinates for the trigger zone
        - radius: distance threshold in meters for activation
        - capture_mode: 'image' or 'video'
        """
        self.drone = drone
        self.camera = camera
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.radius = radius
        self.capture_mode = capture_mode  # 'image' or 'video'
        self.triggered = False  # Flag to ensure one-time activation

        # Create directories to save captured media
        if self.capture_mode == 'image':
            os.makedirs("images", exist_ok=True)
        elif self.capture_mode == 'video':
            os.makedirs("videos", exist_ok=True)

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the great-circle distance between two points on the Earth using the Haversine formula.
        Returns distance in meters.
        """
        R = 6371000  # Earth radius in meters
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        return R * c

    def monitor_and_capture(self):
        """
        Continuously monitor the drone's GPS location. When the drone enters the
        trigger radius, capture a photo or video using the camera.
        """
        print(f"[INFO] Monitoring position. Target radius: {self.radius:.1f} m")

        while not self.triggered:
            # Get current drone location
            location = self.drone.vehicle.location.global_relative_frame

            # Compute distance to target coordinates
            distance = self.haversine_distance(
                location.lat, location.lon,
                self.target_lat, self.target_lon
            )

            print(f"[INFO] Distance to target: {distance:.2f} m")

            # Check if within capture radius
            if distance <= self.radius:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Capture image or video
                if self.capture_mode == 'image':
                    filename = f"images/capture_{timestamp}.jpg"
                    self.camera.capture_image(filename)
                elif self.capture_mode == 'video':
                    filename = f"videos/record_{timestamp}.h264"
                    self.camera.start_recording(filename)
                    time.sleep(5)  # Record 5 seconds of video
                    self.camera.stop_recording()

                self.triggered = True
                print(f"[SUCCESS] Capture complete at {timestamp}")

            time.sleep(1)  # Poll every second
