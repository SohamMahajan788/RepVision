# AI Gym Trainer

A Python-based AI Gym Trainer that uses computer vision to detect and correct human workout postures in real-time.

## Features

- Real-time workout posture tracking and analysis
- Detection and counting for multiple exercises:
  - Squats
  - Bicep curls
  - Push-ups
- Visual feedback on posture correctness
- Rep counting that only increments with correct form
- Exercise switching with GUI buttons
- Interactive tutorials for each exercise
- Audio feedback for rep counting and form corrections
- Advanced feedback visualization:
  - Colored skeleton overlay to highlight problem areas
  - Side-by-side comparison with ideal form
  - Form scoring system (0-100%)

## Requirements

- Python 3.7+
- Webcam
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository or download the source code

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Download sound files (optional, placeholders will be created if missing):
```
python download_sounds.py
```

4. Download reference images for side-by-side comparison (optional, placeholders will be created if missing):
```
python download_reference_images.py
```

## Usage

### Command Line Version
```
python ai_gym_trainer.py
```
- Press 'e' to change exercise type
- Press 'q' to quit the application

### GUI Version (Recommended)
```
python ai_gym_trainer_gui.py
```
- Use buttons to switch between exercises
- Toggle audio feedback on/off
- View exercise tutorials
- Toggle side-by-side comparison view
- Reset rep counter
- Click Quit button to exit

## Exercise Detection

The application detects the following exercises:

### Squats
- Tracks knee angle and posture
- Provides feedback on depth and knee position
- Highlights improper form in the skeleton visualization

### Bicep Curls
- Monitors elbow angle and upper arm stability
- Ensures full range of motion and controlled movement
- Detects and alerts when using momentum instead of proper form

### Push-ups
- Evaluates body alignment and elbow angle
- Checks for proper depth and arm extension
- Ensures straight body position throughout the movement

## Advanced Feedback Features

### Colored Skeleton Overlay
- Real-time visual feedback with color-coded skeleton
- Red highlights for serious form issues
- Yellow for minor corrections
- Green for correct form

### Side-by-Side Comparison
- Shows your form next to an ideal reference image
- Synchronized with your current position (up/down)
- Can be toggled on/off in the interface

### Form Scoring
- Provides a percentage score (0-100%) of your form quality
- Deducts points for specific form issues
- Helps track improvement over time

## How It Works

1. Captures video feed from the webcam
2. Uses MediaPipe Pose to detect body landmarks
3. Analyzes joint angles and body positions
4. Provides real-time feedback and counts reps
5. Gives audio cues for correct/incorrect form (GUI version)
6. Displays visual feedback with colored skeleton and reference images

## Technical Details

- Built with OpenCV for video capture and display
- Uses MediaPipe Pose for human pose estimation
- Implements angle calculations between joints for posture analysis
- Provides color-coded visual feedback
- GUI built with Tkinter
- Audio feedback using Pygame 

## Customizing Reference Images

You can add your own reference images for the side-by-side comparison:

1. Place .jpg images in the `reference_images` directory
2. Name the files using the format: `[exercise]_[position].jpg`
   - For example: `squat_up.jpg`, `squat_down.jpg`, etc.
3. Restart the application to use your custom reference images 