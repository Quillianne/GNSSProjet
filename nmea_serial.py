import serial
import pynmea2

serial_port = 'COM3'
baud_rate = 4800

try:
    with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
        while True:
            line = ser.readline().decode('ascii', errors='replace')
            if line.startswith('$GPGSV'):
                try:
                    
                    
                    msg = pynmea2.parse(line)
                    print("Available attributes:", dir(msg))
                    print(msg)

                except pynmea2.nmea.ParseError as e:
                    print(f"Parse error: {e}")
                except AttributeError as e:
                    print(f"Attribute error: {e}")
except serial.SerialException as e:
    print(f"Error: {e}")
