from dronekit import connect, VehicleMode, Command
import time
from pymavlink import mavutil
from math import radians, cos, sin, asin, sqrt

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

    def return_to_launch(self):
        print("Returning to Launch...")
        self.vehicle.mode = VehicleMode("RTL")

    def send_velocity(self, vx, vy, vz, duration=1):
        """
        Move vehicle in direction based on specified velocity vectors.
        vx: forward/backward (m/s). + = forward
        vy: left/right (m/s). + = right
        vz: up/down (m/s). + = down (not intuitive!)
        duration: time to move in that direction
        """
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,  # time_boot_ms (not used)
            0, 0,  # target system, target component
            mavutil.mavlink.MAV_FRAME_BODY_NED,  # relative to drone's current direction
            0b0000111111000111,  # type mask (only velocities enabled)
            0, 0, 0,  # x, y, z positions (not used)
            vx, vy, vz,  # velocities in m/s
            0, 0, 0,  # acceleration (not used)
            0, 0)  # yaw, yaw_rate (not used)

        for _ in range(int(duration * 10)):
            self.vehicle.send_mavlink(msg)
            time.sleep(0.1)

    def move_forward(self, speed=0.5, duration=1):
        self.send_velocity(vx=speed, vy=0, vz=0, duration=duration)

    def move_backward(self, speed=0.5, duration=1):
        self.send_velocity(vx=-speed, vy=0, vz=0, duration=duration)

    def move_right(self, speed=0.5, duration=1):
        self.send_velocity(vx=0, vy=speed, vz=0, duration=duration)

    def move_left(self, speed=0.5, duration=1):
        self.send_velocity(vx=0, vy=-speed, vz=0, duration=duration)

    def close(self):
        self.vehicle.close()
        print("Connection closed.")



