import cv2
import mediapipe as mp
import numpy as np
import time
from exercises import SquatDetector, BicepCurlDetector, PushUpDetector

class AIGymTrainer:
    def __init__(self):
        # Start webcam stream
        self.cap = cv2.VideoCapture(0)
        
        # Set up MediaPipe Pose model
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load all exercise detectors
        self.exercise_detectors = {
            'squat': SquatDetector(),
            'bicep_curl': BicepCurlDetector(),
            'push_up': PushUpDetector()
        }
        
        # Default exercise selection
        self.current_exercise = 'squat'
        
        # Tracking exercise progress
        self.rep_count = 0
        self.status = "Ready"
        self.feedback = ""
        
    def calculate_angle(self, a, b, c):
        """
        Compute the angle between three body landmarks.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        # Normalize angle to stay within 0–180 degrees
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def process_frame(self):
        """
        Capture and process a single video frame.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Mirror the frame for a natural camera view
        frame = cv2.flip(frame, 1)
        
        # MediaPipe expects RGB format
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract pose landmarks
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Draw skeleton overlay
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Use the relevant detector for the selected exercise
            detector = self.exercise_detectors[self.current_exercise]
            rep_count, status, feedback, color = detector.process(results.pose_landmarks, self.mp_pose)
            
            self.rep_count = rep_count
            self.status = status
            self.feedback = feedback
            
            # Display information on screen
            cv2.putText(frame, f'Exercise: {self.current_exercise.replace("_", " ").title()}', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Reps: {self.rep_count}', 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Status: {self.status}', 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            if self.feedback:
                cv2.putText(frame, f'Feedback: {self.feedback}', 
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
        return frame
    
    def change_exercise(self):
        """
        Switch to the next exercise in the list.
        """
        exercises = list(self.exercise_detectors.keys())
        current_idx = exercises.index(self.current_exercise)
        next_idx = (current_idx + 1) % len(exercises)
        
        self.current_exercise = exercises[next_idx]
        
        # Reset counters and messages when exercise changes
        self.rep_count = 0
        self.status = "Ready"
        self.feedback = ""
        
        # Reset specific detector states
        self.exercise_detectors[self.current_exercise].reset()
    
    def run(self):
        """
        Main loop of the application — handles video, UI, and controls.
        """
        while self.cap.isOpened():
            frame = self.process_frame()
            if frame is None:
                break
            
            cv2.imshow('AI Gym Trainer', frame)
            
            # Keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                self.change_exercise()
        
        # Cleanup on exit
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting AI Gym Trainer...")
    print("Controls:")
    print("- Press 'e' to switch exercises")
    print("- Press 'q' to exit")
    
    trainer = AIGymTrainer()
    trainer.run()
