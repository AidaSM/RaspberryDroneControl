from time import sleep
from GeoTrigger import haversine_distance

class MissionPlanner:
    def __init__(self, drone, camera, obstacle_avoidance, waypoints, waypoint_radius=2.0):
        """
        Initialize the mission planner with:
        - drone: a DroneControl instance
        - camera: a Camera instance
        - obstacle_avoidance: an ObstacleAvoidance instance
        - waypoints: list of tuples (lat, lon, alt)
        - waypoint_radius: proximity threshold in meters for "arrival"
        """
        self.drone = drone
        self.camera = camera
        self.avoidance = obstacle_avoidance
        self.waypoints = waypoints
        self.radius = waypoint_radius

    def start_mission(self):
        """
        Start the autonomous mission:
        - For each waypoint, fly the drone to it
        - Continuously check position and obstacle status
        - Capture an image upon arrival
        """
        print("[MISSION] Starting mission...")

        for idx, (lat, lon, alt) in enumerate(self.waypoints):
            print(f"\n[NAV] Heading to waypoint {idx + 1}: {lat}, {lon}, {alt}m")
            self.drone.goto_location(lat, lon, alt)

            # Loop until the drone reaches the waypoint
            while True:
                current = self.drone.vehicle.location.global_relative_frame
                distance = haversine_distance(current.lat, current.lon, lat, lon)
                print(f"[INFO] Distance to WP{idx+1}: {distance:.2f} m")

                # Check for obstacles and perform avoidance if needed
                if self.avoidance.check_and_avoid():
                    print("[AVOID] Obstacle handled. Resuming course...")

                # If within radius, consider the waypoint reached
                if distance < self.radius:
                    print(f"[ARRIVED] Reached waypoint {idx + 1}")
                    self.camera.capture_image(f"images/waypoint_{idx + 1}.jpg")
                    break

                sleep(1)  # Wait before checking again

        print("\n[MISSION] Mission complete.")
