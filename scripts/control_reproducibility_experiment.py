"""
Interactive control script for reproducibility experiment.

This script allows you to control the reproducibility experiment
via Socket.IO messages.

Usage:
    1. Start relay server: python relay_server.py 5000
    2. Start main_loop: python main_loop.py robot_1
    3. Run this script: python scripts/control_reproducibility_experiment.py
"""

import socketio
import time
import sys

# Configuration
HOST = "127.0.0.1"
PORT = 5000
ROBOT_IDS = ["robot_1", "robot_2"]  # Change to robot_2 if needed

def send_command(sio, action, extra_data=None):
    """Send control command to reproducibility experiment."""
    for ROBOT_ID in ROBOT_IDS:
        data = {
            'robot_ID': ROBOT_ID,
            'action': action
        }
        if extra_data:
            data.update(extra_data)
        
        message = {
            'topic': 'Neuronavigation to Robot: Control reproducibility experiment',
            'data': data
        }
        sio.emit('from_neuronavigation', message)
        print(f"✓ Command '{action}' sent\n")

def main():
    print("=" * 60)
    print("REPRODUCIBILITY EXPERIMENT CONTROL")
    print("=" * 60)
    print(f"Robot ID: {ROBOT_IDS}")
    print(f"Server: {HOST}:{PORT}\n")
    
    # Create Socket.IO client
    sio = socketio.Client()
    
    @sio.event
    def connect():
        print(f"✓ Connected to relay server\n")
    
    @sio.event
    def disconnect():
        print("✗ Disconnected from relay server")
    
    # Connect to server
    try:
        url = f'http://{HOST}:{PORT}'
        sio.connect(url)
        time.sleep(0.5)
    except Exception as e:
        print(f"✗ Error connecting to server: {e}")
        print("\nMake sure:")
        print(f"  1. Relay server is running: python relay_server.py {PORT}")
        print(f"  2. Main loop is running: python main_loop.py {ROBOT_IDS}")
        return

    try:
        while True:
            print("\n" + "=" * 60)
            print("MENU")
            print("=" * 60)
            print("1. START recording   (begin a new trial)")
            print("2. STOP recording    (stop and save current trial)")
            print("3. Check STATUS      (view current state)")
            print("4. CANCEL trial      (abort without saving)")
            print("5. EXPORT summary    (save all trials to JSON)")
            print("6. Help              (show workflow)")
            print("7. Quit")
            print("=" * 60)
            
            choice = input("\nChoice: ").strip()
            
            if choice == "1":
                print("\n--- START RECORDING ---")
                send_command(sio, "start")
                print("Recording started. Send 'stop' when done.")
            
            elif choice == "2":
                print("\n--- STOP RECORDING ---")
                send_command(sio, "stop")
                print("Recording stopped. Data saved to CSV.")
                
            elif choice == "3":
                print("\n--- STATUS ---")
                send_command(sio, "status")
                print("(Check main_loop console for detailed status)")
                
            elif choice == "4":
                print("\n--- CANCEL TRIAL ---")
                confirm = input("Are you sure? (y/n): ").strip().lower()
                if confirm == 'y':
                    send_command(sio, "cancel")
                else:
                    print("Cancelled")
                
            elif choice == "5":
                print("\n--- EXPORT SUMMARY ---")
                filename = input("Base filename (Enter for default): ").strip()
                extra = {"filename": filename} if filename else {}
                send_command(sio, "export", extra)
                
            elif choice == "6":
                print("\n" + "=" * 60)
                print("WORKFLOW")
                print("=" * 60)
                print("\nFor each trial:")
                print("  1. Send 'go' from neuronavigation if desired")
                print("  2. Select '1. START recording' to begin")
                print("  3. When done, select '2. STOP recording'")
                print("  4. Repeat as many times as needed")
                print("\nWhen finished:")
                print("  - Select '5. EXPORT summary' to generate JSON report")
                print("  - Provide a base filename (each trial gets an index)")
                print("=" * 60)
                input("\nPress Enter to continue...")
                
            elif choice == "7":
                print("\nExiting...")
                break

            else:
                print("\n✗ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    
    finally:
        if sio.connected:
            sio.disconnect()
        print("\n✓ Disconnected")


if __name__ == "__main__":
    main()
