import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, Attitude

async def connect_drone():
    drone = System()
    await drone.connect(system_address="serial:///dev/serial0")
    return drone

async def start_offboard_mode(drone):
    await drone.offboard.set_attitude(Attitude(roll_deg=0, pitch_deg=0, yaw_deg=0, thrust_value=0.5))
    try:
        await drone.offboard.start()
        print("Offboard mode started")
    except OffboardError as e:
        print(f"Failed to start offboard: {e}")

async def turn_left(drone, duration=1.0):
    await drone.offboard.set_attitude(Attitude(yaw_deg=-30, thrust_value=0.5))
    await asyncio.sleep(duration)
    await drone.offboard.set_attitude(Attitude(yaw_deg=0, thrust_value=0.5))

async def turn_right(drone, duration=1.0):
    await drone.offboard.set_attitude(Attitude(yaw_deg=30, thrust_value=0.5))
    await asyncio.sleep(duration)
    await drone.offboard.set_attitude(Attitude(yaw_deg=0, thrust_value=0.5))

async def move_forward(drone, duration=1.0):
    await drone.offboard.set_attitude(Attitude(pitch_deg=-10, thrust_value=0.6))
    await asyncio.sleep(duration)
    await drone.offboard.set_attitude(Attitude(pitch_deg=0, thrust_value=0.5))

async def move_backward(drone, duration=1.0):
    await drone.offboard.set_attitude(Attitude(pitch_deg=10, thrust_value=0.6))
    await asyncio.sleep(duration)
    await drone.offboard.set_attitude(Attitude(pitch_deg=0, thrust_value=0.5))

async def main():
    drone = await connect_drone()
    await start_offboard_mode(drone)

    await move_forward(drone, 2)
    await turn_left(drone, 1)
    await move_backward(drone, 2)
    await turn_right(drone, 1)

asyncio.run(main())
