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
ROBOT_ID = "robot_1"  # Change to robot_2 if needed

def send_command(sio, action):
    """Send control command to reproducibility experiment."""
    message = {
        'topic': 'Neuronavigation to Robot: Control reproducibility experiment',
        'data': {
            'robot_ID': ROBOT_ID,
            'action': action
        }
    }
    sio.emit('from_neuronavigation', message)
    print(f"✓ Command '{action}' sent\n")

def main():
    print("=" * 60)
    print("REPRODUCIBILITY EXPERIMENT CONTROL")
    print("=" * 60)
    print(f"Robot ID: {ROBOT_ID}")
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
        print(f"  2. Main loop is running: python main_loop.py {ROBOT_ID}")
        return
    
    try:
        while True:
            print("\n" + "=" * 60)
            print("MENU")
            print("=" * 60)
            print("1. ARM next trial   (prepare for recording)")
            print("2. Check STATUS     (view current state)")
            print("3. CANCEL trial     (abort current trial)")
            print("4. EXPORT summary   (save results to JSON)")
            print("5. Help             (show workflow)")
            print("6. Quit")
            print("=" * 60)
            
            choice = input("\nChoice: ").strip()
            
            if choice == "1":
                print("\n--- ARM TRIAL ---")
                send_command(sio, "arm")
                print("Next steps:")
                print("  1. Position robot at starting location")
                print("  2. Move robot until it reaches target")
                print("  3. Recording will start automatically for 90 seconds")
                
            elif choice == "2":
                print("\n--- STATUS ---")
                send_command(sio, "status")
                print("(Check main_loop console for detailed status)")
                
            elif choice == "3":
                print("\n--- CANCEL TRIAL ---")
                confirm = input("Are you sure? (y/n): ").strip().lower()
                if confirm == 'y':
                    send_command(sio, "cancel")
                else:
                    print("Cancelled")
                
            elif choice == "4":
                print("\n--- EXPORT SUMMARY ---")
                send_command(sio, "export")
                print(f"Summary will be saved as: data/reproducibility/{ROBOT_ID}_reproducibility_summary.json")
                
            elif choice == "5":
                print("\n" + "=" * 60)
                print("WORKFLOW")
                print("=" * 60)
                print("\nFor each trial (20 total):")
                print("  1. Select '1. ARM next trial'")
                print("  2. Position robot at desired starting location")
                print("  3. Move robot until 'coil at target' is reached")
                print("  4. System automatically records for 90 seconds:")
                print("     - Target position (robot coordinates)")
                print("     - Coil position (robot coordinates)")
                print("  5. Data saved automatically")
                print("  6. Repeat from step 1\n")
                print("After 20 trials:")
                print("  - Select '4. EXPORT summary' to generate JSON report")
                print("  - Data files: data/reproducibility/{ROBOT_ID}_trial_XX_*.csv")
                print("=" * 60)
                input("\nPress Enter to continue...")
                
            elif choice == "6":
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
