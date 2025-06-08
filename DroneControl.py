import Camera
class DroneController:
    def __init__(self):
        self.altitude = 0
        self.is_flying = False
        self.position = [0, 0]
        self.gps_coords = (0.0, 0.0)
        self.camera = Camera()

    def takeoff(self, target_altitude=10):
        if self.is_flying:
            print("Drone is already flying.")
        else:
            self.altitude = target_altitude
            self.is_flying = True
            print(f"Taking off to {self.altitude} meters.")

    def land(self):
        if not self.is_flying:
            print("Drone is already landed.")
        else:
            self.altitude = 0
            self.is_flying = False
            print("Landing... Drone is on the ground.")

    def move(self, direction, distance):
        if not self.is_flying:
            print("Drone is not flying. Can't move.")
            return

        if direction == "forward":
            self.position[1] += distance
        elif direction == "backward":
            self.position[1] -= distance
        elif direction == "left":
            self.position[0] -= distance
        elif direction == "right":
            self.position[0] += distance
        else:
            print("Unknown direction.")
            return

        print(f"Moving {direction} by {distance} meters. Current position: {self.position}")

    def change_altitude(self, delta):
        if not self.is_flying:
            print("Drone is not flying. Can't change altitude.")
            return

        self.altitude += delta
        if self.altitude < 0:
            self.altitude = 0
        print(f"Changing altitude by {delta} meters. Current altitude: {self.altitude}")

    def hover(self):
        if self.is_flying:
            print(f"Hovering at altitude {self.altitude} meters, position {self.position}.")
        else:
            print("Drone is not flying.")


    def update_gps(self, latitude, longitude):
        self.gps_coords = (latitude, longitude)
        print(f"GPS updated: Latitude {latitude}, Longitude {longitude}")

    def get_gps_location(self):
        print(f"Current GPS location: {self.gps_coords}")
        return self.gps_coords