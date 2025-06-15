from time import sleep
from GeoTrigger import haversine_distance

class MissionPlanner:
    def __init__(self, drone, camera, obstacle_avoidance, waypoints, waypoint_radius=2.0):
        self.drone = drone
        self.camera = camera
        self.avoidance = obstacle_avoidance
        self.waypoints = waypoints  # List of (lat, lon, alt)
        self.radius = waypoint_radius

    def start_mission(self):
        print("[MISSION] Starting mission...")

        for idx, (lat, lon, alt) in enumerate(self.waypoints):
            print(f"\n[NAV] Heading to waypoint {idx + 1}: {lat}, {lon}, {alt}m")
            self.drone.goto_location(lat, lon, alt)

            while True:
                current = self.drone.vehicle.location.global_relative_frame
                distance = haversine_distance(current.lat, current.lon, lat, lon)
                print(f"[INFO] Distance to WP{idx+1}: {distance:.2f} m")

                # Obstacle check
                if self.avoidance.check_and_avoid():
                    print("[AVOID] Obstacle handled. Resuming course...")

                if distance < self.radius:
                    print(f"[ARRIVED] Reached waypoint {idx + 1}")
                    self.camera.capture_image(f"images/waypoint_{idx + 1}.jpg")
                    break
                sleep(1)

        print("\n[MISSION] Mission complete.")
