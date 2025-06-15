#!/usr/bin/env python3
import evdev

# List all input devices
devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
for device in devices:
    print(f"Device: {device.path}, {device.name}, {device.phys}")

# Use the first gamepad found
gamepad = None
for device in devices:
    if any(keyword in device.name.lower() for keyword in ['joy', 'gamepad', 'controller']):
        gamepad = device
        break

if gamepad:
    print(f"\nUsing gamepad: {gamepad.name}")
    print("Press buttons or move axes to see their codes (Ctrl+C to exit)")

    try:
        for event in gamepad.read_loop():
            if event.type == evdev.ecodes.EV_KEY:
                print(f"Button: {evdev.categorize(event)}")
            elif event.type == evdev.ecodes.EV_ABS:
                print(f"Axis: {evdev.ecodes.ABS[event.code]} Value: {event.value}")
    except KeyboardInterrupt:
        print("\nExiting...")
else:
    print("No gamepad found!")