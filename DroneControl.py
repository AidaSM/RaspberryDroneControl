from dronekit import connect, VehicleMode, Command
import time
from pymavlink import mavutil
from math import radians, cos, sin, asin, sqrt

class DroneControl:
    def __init__(self, connection_string='/dev/ttyAMA0', baud=115200):
        # Connect to the drone via serial (default is UART on Raspberry Pi)
        print("Connecting to drone...")
        self.vehicle = connect(connection_string, baud=baud, wait_ready=True)
        print("Connected.")

    def arm_and_takeoff(self, target_altitude=0.5):
        # Arms the drone and initiates takeoff to a target altitude
        print("Arming motors...")
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True

        # Wait until drone is armed
        while not self.vehicle.armed:
            print("Waiting for arming...")
            time.sleep(1)

        # Begin takeoff
        print("Taking off!")
        self.vehicle.simple_takeoff(target_altitude)

        # Monitor altitude until target is reached
        while True:
            alt = self.vehicle.location.global_relative_frame.alt
            print(f"Altitude: {alt:.1f}")
            if alt >= target_altitude * 0.95:
                print("Reached target altitude.")
                break
            time.sleep(1)

    def land(self):
        # Commands the drone to land
        print("Landing...")
        self.vehicle.mode = VehicleMode("LAND")

    def disarm(self):
        # Disarms the drone (typically after landing)
        print("Disarming...")
        self.vehicle.armed = False

    def goto_location(self, lat, lon, alt):
        # Commands the drone to fly to a specific GPS coordinate
        from dronekit import LocationGlobalRelative
        point = LocationGlobalRelative(lat, lon, alt)
        self.vehicle.simple_goto(point)

    def return_to_launch(self):
        # Commands the drone to return to its launch location
        print("Returning to Launch...")
        self.vehicle.mode = VehicleMode("RTL")

    def send_velocity(self, vx, vy, vz, duration=1):
        """
        Sends velocity command to the drone in the body frame (relative to drone's orientation).
        vx: Forward/backward velocity (+ is forward)
        vy: Right/left velocity (+ is right)
        vz: Up/down velocity (+ is down — note inverted z axis)
        duration: Duration to apply velocity command (in seconds)
        """
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,  # time_boot_ms (ignored)
            0, 0,  # target system, target component
            mavutil.mavlink.MAV_FRAME_BODY_NED,  # velocities relative to drone's current heading
            0b0000111111000111,  # bitmask: enable velocity components only
            0, 0, 0,  # position (not used)
            vx, vy, vz,  # velocity (m/s)
            0, 0, 0,  # acceleration (not used)
            0, 0  # yaw, yaw_rate (not used)
        )

        # Repeat the velocity command for the duration specified
        for _ in range(int(duration * 10)):
            self.vehicle.send_mavlink(msg)
            time.sleep(0.1)

    # Convenience movement methods using velocity control
    def move_forward(self, speed=0.5, duration=1):
        self.send_velocity(vx=speed, vy=0, vz=0, duration=duration)

    def move_backward(self, speed=0.5, duration=1):
        self.send_velocity(vx=-speed, vy=0, vz=0, duration=duration)

    def move_right(self, speed=0.5, duration=1):
        self.send_velocity(vx=0, vy=speed, vz=0, duration=duration)

    def move_left(self, speed=0.5, duration=1):
        self.send_velocity(vx=0, vy=-speed, vz=0, duration=duration)

    def close(self):
        # Closes the connection to the drone
        self.vehicle.close()
        print("Connection closed.")
