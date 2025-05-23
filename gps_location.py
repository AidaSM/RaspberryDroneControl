import serial
import pynmea2

ser = serial.Serial('/dev/serial0', 9600, timeout=1)
while True:
    line = ser.readline().decode('ascii', errors='replace')
    if line.startswith('$GPGGA'):
        msg = pynmea2.parse(line)
        print(f'Lat: {msg.latitude} {msg.lat_dir}, Lon: {msg.longitude} {msg.lon_dir}')
