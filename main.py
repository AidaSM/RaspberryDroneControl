from DroneControl import DroneControl
from Camera import Camera
from ObstacleAvoidance import ObstacleAvoidance
from MissionPlanner import MissionPlanner
from DroneVisionAnalyzer import DroneVisionAnalyzer

import atexit

# ========== INITIALIZATION ==========
print("[SYSTEM] Initializing components...")
drone = DroneControl()
camera = Camera()
obstacle_avoidance = ObstacleAvoidance()
vision_analyzer = DroneVisionAnalyzer(camera)

# Ensure clean shutdown
atexit.register(lambda: drone.close())
atexit.register(lambda: vision_analyzer.stop() if vision_analyzer.is_alive() else None)
atexit.register(lambda: obstacle_avoidance.release())

# ========== START VISION ANALYSIS THREAD ==========
print("[VISION] Starting DroneVisionAnalyzer thread...")
vision_analyzer.start()

# ========== DEFINE WAYPOINTS ==========
# Replace with real coordinates if needed
waypoints = [
    (47.3769, 8.5417, 5),
    (47.3770, 8.5420, 5),
    (47.3771, 8.5422, 5)
]

# ========== START MISSION ==========
print("[MISSION] Starting autonomous mission...")
mission = MissionPlanner(drone, camera, obstacle_avoidance, waypoints)
mission.start_mission()

# ========== RETURN TO LAUNCH AND SHUTDOWN ==========
print("[MISSION] Returning to Launch and cleaning up...")
drone.return_to_launch()
drone.land()

# Wait for vision analyzer to finish saving
vision_analyzer.stop()

print("[SYSTEM] Mission complete. All systems shut down.")