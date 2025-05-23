import time
import pigpio

ESC_GPIO = 4  # GPIO pin number
pi = pigpio.pi()

# Set the ESC to its minimum throttle
pi.set_servo_pulsewidth(ESC_GPIO, 1000)
time.sleep(2)

# Set the ESC to its maximum throttle
pi.set_servo_pulsewidth(ESC_GPIO, 2000)
time.sleep(2)

# Set the ESC to a mid-range throttle
pi.set_servo_pulsewidth(ESC_GPIO, 1500)
time.sleep(2)

# Stop sending pulses to the ESC
pi.set_servo_pulsewidth(ESC_GPIO, 0)
pi.stop()
