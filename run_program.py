#!/usr/bin/env python
"""
AI Gym Trainer Launcher Script

This script is used to set up and run the AI Gym Trainer application.
It ensures all necessary directories and files exist before launching
the main GUI application.
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if required Python packages are installed"""
    try:
        import cv2
        import mediapipe
        import numpy
        import PIL
        import pygame
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install all required packages using: pip install -r requirements.txt")
        return False

def ensure_directories():
    """Ensure that all required directories exist"""
    directories = ['sounds', 'reference_images']
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Creating {directory} directory...")
            os.makedirs(directory, exist_ok=True)
    print("✓ All required directories exist")

def check_download_resources():
    """Check if sound and reference image resources exist, download if missing"""
    # Check sounds
    sound_files_exist = all(os.path.exists(f"sounds/{sound}.mp3") 
                           for sound in ['good_rep', 'bad_form', 'good_form'])
    
    # Check reference images
    ref_images_exist = all(os.path.exists(f"reference_images/{exercise}_{position}.jpg") 
                          for exercise in ['squat', 'bicep_curl', 'push_up'] 
                          for position in ['up', 'down'])
    
    # Download missing resources
    if not sound_files_exist:
        print("Downloading sound files...")
        try:
            subprocess.run([sys.executable, "download_sounds.py"], check=True)
            print("✓ Sound files downloaded successfully")
        except subprocess.SubprocessError:
            print("✗ Failed to download sound files. The program will still run but without audio.")
    else:
        print("✓ Sound files already exist")
    
    if not ref_images_exist:
        print("Downloading reference images...")
        try:
            subprocess.run([sys.executable, "download_reference_images.py"], check=True)
            print("✓ Reference images downloaded successfully")
        except subprocess.SubprocessError:
            print("✗ Failed to download reference images. The program will still run with placeholders.")
    else:
        print("✓ Reference images already exist")

def run_application():
    """Run the main AI Gym Trainer GUI application"""
    print("\nStarting AI Gym Trainer...\n")
    try:
        subprocess.run([sys.executable, "ai_gym_trainer_gui.py"], check=True)
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"\n✗ Error running the application: {e}")

def main():
    """Main function to set up and run the AI Gym Trainer"""
    print("\n===== AI Gym Trainer Setup =====\n")
    
    # Check dependencies
    if not check_dependencies():
        input("Press Enter to exit...")
        return
    
    # Ensure directories exist
    ensure_directories()
    
    # Check and download resources
    check_download_resources()
    
    # Give the user a moment to read the setup info
    print("\nSetup complete. Starting application in 2 seconds...")
    time.sleep(2)
    
    # Run the application
    run_application()

if __name__ == "__main__":
    main() 