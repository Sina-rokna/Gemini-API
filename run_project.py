import subprocess
import sys
import os
import time

def install_dependencies():
    print("Checking and installing required libraries...")

    if not os.path.exists('requirements.txt'):
        print("Error: 'requirements.txt' not found!")
        print("Make sure this file is in the same folder as requirements.txt")
        sys.exit(1)

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n Libraries installed successfully.\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n Failed to install libraries. Error: {e}")
        sys.exit(1)

def run_extractor():
    script_name = "card_extractor.py"
    
    if not os.path.exists(script_name):
        print(f"Error: '{script_name}' not found.")
        sys.exit(1)
        
    print(f" Starting {script_name}...\n" + "="*30 + "\n")
    
    try:
        result = subprocess.run([sys.executable, script_name])
        
        if result.returncode == 0:
            print("\n" + "="*30 + "\n Process finished successfully.")
        else:
            print("\n The script encountered an error.")
            
    except KeyboardInterrupt:
        print("\nProcess stopped by user.")

if __name__ == "__main__":
    print("--- Auto-Setup Runner ---\n")
    install_dependencies()
    time.sleep(1)
    run_extractor()
    
    if os.name == 'nt':
        input("\nPress Enter to close this window...")