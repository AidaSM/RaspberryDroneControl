from dronekit import connect, VehicleMode, Command
import time

class DroneControl:
    def __init__(self, connection_string='/dev/ttyAMA0', baud=115200):
        print("Connecting to drone...")
        self.vehicle = connect(connection_string, baud=baud, wait_ready=True)
        print("Connected.")

    def arm_and_takeoff(self, target_altitude=0.5):
        print("Arming motors...")
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True

        while not self.vehicle.armed:
            print("Waiting for arming...")
            time.sleep(1)

        print("Taking off!")
        self.vehicle.simple_takeoff(target_altitude)

        while True:
            alt = self.vehicle.location.global_relative_frame.alt
            print(f"Altitude: {alt:.1f}")
            if alt >= target_altitude * 0.95:
                print("Reached target altitude.")
                break
            time.sleep(1)

    def land(self):
        print("Landing...")
        self.vehicle.mode = VehicleMode("LAND")

    def disarm(self):
        print("Disarming...")
        self.vehicle.armed = False

    def goto_location(self, lat, lon, alt):
        from dronekit import LocationGlobalRelative
        point = LocationGlobalRelative(lat, lon, alt)
        self.vehicle.simple_goto(point)

    def close(self):
        self.vehicle.close()
        print("Connection closed.")

