#!/usr/bin/env python3
"""
Data collection helper - capture frames from camera for labeling.
This script helps you collect training data by capturing frames from your camera.
"""

import cv2
import os
from pathlib import Path
import time
import msvcrt
import sys
import threading

def collect_training_data_auto():
    """Automatically capture frames at intervals (headless mode)"""
    
    project_root = Path(__file__).parent
    images_dir = project_root / "training_data" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Automatic Data Collection Mode (Headless)")
    print("Controls:")
    print("  Press 's' + Enter - Save current frame")
    print("  Press 'a' + Enter - Start auto-capture (every 2 seconds)")
    print("  Press 'q' + Enter - Quit")
    print(f"Images will be saved to: {images_dir}")
    print("="*50)
    
    frame_count = 0
    auto_capture = False
    last_capture_time = 0
    
    def get_input():
        """Non-blocking input function"""
        while True:
            try:
                if msvcrt.kbhit():
                    return msvcrt.getch().decode('utf-8').lower()
            except:
                pass
            time.sleep(0.1)
        return None
    
    print("Waiting for commands... (Type 's' and press Enter to save frame)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Check for keyboard input
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8').lower()
            
            if key == 'q':
                break
            elif key == 's':
                # Save current frame
                timestamp = int(time.time() * 1000)
                filename = f"frame_{timestamp:013d}.jpg"
                filepath = images_dir / filename
                
                cv2.imwrite(str(filepath), frame)
                frame_count += 1
                print(f"Saved: {filename} (Total: {frame_count})")
                
            elif key == 'a':
                auto_capture = not auto_capture
                if auto_capture:
                    print("Auto-capture enabled (every 2 seconds)")
                else:
                    print("Auto-capture disabled")
        
        # Auto capture every 2 seconds if enabled
        current_time = time.time()
        if auto_capture and (current_time - last_capture_time) >= 2.0:
            timestamp = int(current_time * 1000)
            filename = f"auto_frame_{timestamp:013d}.jpg"
            filepath = images_dir / filename
            
            cv2.imwrite(str(filepath), frame)
            frame_count += 1
            last_capture_time = current_time
            print(f"Auto-saved: {filename} (Total: {frame_count})")
        
        time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    
    cap.release()
    
    print(f"\nData collection complete!")
    print(f"Captured {frame_count} frames")
    print(f"Next step: Run 'python start_labelme.py' to annotate the images")

def collect_training_data_manual():
    """Manual capture with console input"""
    
    project_root = Path(__file__).parent
    images_dir = project_root / "training_data" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Manual Data Collection Mode")
    print("Commands:")
    print("  'capture' or 'c' - Save current frame")
    print("  'quit' or 'q' - Exit")
    print(f"Images will be saved to: {images_dir}")
    print("="*50)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        print(f"\nCamera active. Frames captured: {frame_count}")
        command = input("Enter command (c/capture, q/quit): ").lower().strip()
        
        if command in ['q', 'quit']:
            break
        elif command in ['c', 'capture']:
            timestamp = int(time.time() * 1000)
            filename = f"frame_{timestamp:013d}.jpg"
            filepath = images_dir / filename
            
            cv2.imwrite(str(filepath), frame)
            frame_count += 1
            print(f"Saved: {filename}")
        else:
            print("Invalid command. Use 'c' to capture or 'q' to quit.")
    
    cap.release()
    
    print(f"\nData collection complete!")
    print(f"Captured {frame_count} frames")
    print(f"Next step: Run 'python start_labelme.py' to annotate the images")

if __name__ == "__main__":
    print("Choose data collection mode:")
    print("1. Auto mode (headless with keyboard shortcuts)")
    print("2. Manual mode (console input)")
    
    while True:
        try:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == '1':
                collect_training_data_auto()
                break
            elif choice == '2':
                collect_training_data_manual()
                break
            else:
                print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
