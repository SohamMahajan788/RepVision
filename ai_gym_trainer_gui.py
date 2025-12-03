import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import pygame
import os
from exercises import SquatDetector, BicepCurlDetector, PushUpDetector

class AIGymTrainerGUI:
    def __init__(self, root):
        # Main window setup
        self.root = root
        self.root.title("AI Gym Trainer")
        self.root.geometry("1280x720")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Camera options (0 = integrated, 1 = external typically)
        self.available_cameras = {
            "Integrated Camera": 0,
            "External Camera": 1
        }
        self.current_camera_index = 1  # default to external camera
        
        # Open the selected camera and set resolution
        self.cap = cv2.VideoCapture(self.current_camera_index)
        self.setup_camera_resolution()
        
        # MediaPipe Pose initialization
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Exercise detectors registry
        self.exercise_detectors = {
            'squat': SquatDetector(),
            'bicep_curl': BicepCurlDetector(),
            'push_up': PushUpDetector()
        }
        
        # Default exercise and tracking state
        self.current_exercise = 'squat'
        self.rep_count = 0
        self.status = "Ready"
        self.feedback = ""
        self.form_score = 100
        self.problem_areas = {}
        
        # Reference images for visual comparison
        self.reference_images = {
            'squat': {'up': None, 'down': None},
            'bicep_curl': {'up': None, 'down': None},
            'push_up': {'up': None, 'down': None}
        }
        self.load_reference_images()
        self.show_comparison = True
        
        # Layout: video on left, controls on right
        self.video_frame = ttk.Frame(self.root)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        
        # Canvas for displaying camera feed
        self.video_canvas = tk.Canvas(self.video_frame, width=self.width//2, height=self.height//2)
        self.video_canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Stats panel
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Stats")
        self.stats_frame.pack(padx=10, pady=10, fill=tk.X)
        
        self.exercise_label = ttk.Label(self.stats_frame, text="Exercise: Squat", font=("Arial", 12))
        self.exercise_label.pack(padx=5, pady=5, anchor=tk.W)
        
        self.reps_label = ttk.Label(self.stats_frame, text="Reps: 0", font=("Arial", 12))
        self.reps_label.pack(padx=5, pady=5, anchor=tk.W)
        
        self.status_label = ttk.Label(self.stats_frame, text="Status: Ready", font=("Arial", 12))
        self.status_label.pack(padx=5, pady=5, anchor=tk.W)
        
        self.feedback_label = ttk.Label(self.stats_frame, text="Feedback: ", font=("Arial", 12))
        self.feedback_label.pack(padx=5, pady=5, anchor=tk.W)
        
        self.score_label = ttk.Label(self.stats_frame, text="Form Score: 100%", font=("Arial", 12))
        self.score_label.pack(padx=5, pady=5, anchor=tk.W)
        
        # Advanced analytics section
        self.analytics_frame = ttk.LabelFrame(self.control_frame, text="Advanced Analytics")
        self.analytics_frame.pack(padx=10, pady=10, fill=tk.X)
        
        self.speed_label = ttk.Label(self.analytics_frame, text="Movement Speed: 0 deg/sec", font=("Arial", 10))
        self.speed_label.pack(padx=5, pady=2, anchor=tk.W)
        
        self.rom_label = ttk.Label(self.analytics_frame, text="Range of Motion: 0°", font=("Arial", 10))
        self.rom_label.pack(padx=5, pady=2, anchor=tk.W)
        
        self.stability_label = ttk.Label(self.analytics_frame, text="Stability: 100%", font=("Arial", 10))
        self.stability_label.pack(padx=5, pady=2, anchor=tk.W)
        
        self.calories_label = ttk.Label(self.analytics_frame, text="Calories Burned: 0 cal", font=("Arial", 10))
        self.calories_label.pack(padx=5, pady=2, anchor=tk.W)
        
        self.detail_button = ttk.Button(self.analytics_frame, text="Show Detailed Feedback", 
                                       command=self.show_detailed_feedback)
        self.detail_button.pack(padx=5, pady=5, fill=tk.X)
        
        # Exercise selection buttons
        self.buttons_frame = ttk.LabelFrame(self.control_frame, text="Exercises")
        self.buttons_frame.pack(padx=10, pady=10, fill=tk.X)
        
        self.squat_button = ttk.Button(self.buttons_frame, text="Squat", command=lambda: self.change_exercise('squat'))
        self.squat_button.pack(padx=5, pady=5, fill=tk.X)
        
        self.bicep_curl_button = ttk.Button(self.buttons_frame, text="Bicep Curl", command=lambda: self.change_exercise('bicep_curl'))
        self.bicep_curl_button.pack(padx=5, pady=5, fill=tk.X)
        
        self.push_up_button = ttk.Button(self.buttons_frame, text="Push Up", command=lambda: self.change_exercise('push_up'))
        self.push_up_button.pack(padx=5, pady=5, fill=tk.X)
        
        # Tutorial button
        self.tutorial_button = ttk.Button(self.control_frame, text="Show Tutorial", command=self.show_tutorial)
        self.tutorial_button.pack(padx=10, pady=10, fill=tk.X)
        
        # Reference image toggle
        self.comparison_frame = ttk.LabelFrame(self.control_frame, text="Reference Image")
        self.comparison_frame.pack(padx=10, pady=10, fill=tk.X)
        
        self.comparison_var = tk.BooleanVar(value=True)
        self.comparison_check = ttk.Checkbutton(
            self.comparison_frame, 
            text="Show Reference Image", 
            variable=self.comparison_var,
            command=self.toggle_comparison
        )
        self.comparison_check.pack(padx=5, pady=5, anchor=tk.W)
        
        # Audio controls
        self.audio_frame = ttk.LabelFrame(self.control_frame, text="Audio Settings")
        self.audio_frame.pack(padx=10, pady=10, fill=tk.X)
        
        self.audio_var = tk.BooleanVar(value=True)
        self.audio_check = ttk.Checkbutton(self.audio_frame, text="Audio Feedback", variable=self.audio_var)
        self.audio_check.pack(padx=5, pady=5, anchor=tk.W)
        
        # Initialize audio mixer
        pygame.mixer.init()
        
        # Expected audio file paths
        self.audio_files = {
            'good_rep': 'sounds/good_rep.mp3',
            'bad_form': 'sounds/bad_form.mp3',
            'good_form': 'sounds/good_form.mp3'
        }
        os.makedirs('sounds', exist_ok=True)
        
        # Create simple placeholders if audio files are missing
        for key, filepath in self.audio_files.items():
            if not os.path.exists(filepath):
                with open(filepath, 'w') as f:
                    f.write("")
                print(f"Created placeholder for {filepath}")
        
        # Display settings and camera selection
        self.resolution_frame = ttk.LabelFrame(self.control_frame, text="Display Settings")
        self.resolution_frame.pack(padx=10, pady=10, fill=tk.X)
        
        self.camera_label = ttk.Label(self.resolution_frame, text="Camera:")
        self.camera_label.pack(padx=5, pady=5, anchor=tk.W)
        
        self.camera_var = tk.StringVar(value="External Camera")
        self.camera_dropdown = ttk.Combobox(
            self.resolution_frame,
            textvariable=self.camera_var,
            values=list(self.available_cameras.keys()),
            state="readonly"
        )
        self.camera_dropdown.pack(padx=5, pady=5, fill=tk.X)
        self.camera_dropdown.bind("<<ComboboxSelected>>", self.change_camera)
        
        # Scale control for display size
        self.scale_var = tk.DoubleVar(value=0.5)
        self.scale_label = ttk.Label(self.resolution_frame, text="Display Scale: 50%")
        self.scale_label.pack(padx=5, pady=5, anchor=tk.W)
        
        self.scale_slider = ttk.Scale(
            self.resolution_frame, 
            from_=0.1, 
            to=1.0, 
            orient=tk.HORIZONTAL, 
            variable=self.scale_var,
            command=self.update_scale
        )
        self.scale_slider.pack(padx=5, pady=5, fill=tk.X)
        
        # Reset and quit controls
        self.reset_button = ttk.Button(self.control_frame, text="Reset Counter", command=self.reset_counter)
        self.reset_button.pack(padx=10, pady=10, fill=tk.X)
        
        self.quit_button = ttk.Button(self.control_frame, text="Quit", command=self.on_closing)
        self.quit_button.pack(padx=10, pady=10, fill=tk.X)
        
        # Prevent audio spam and keep image refs alive
        self.last_audio_time = time.time()
        self.photo = None
        
        # Start the video update loop (uses Tk's after to avoid flicker)
        self.update_frame()
    
    def setup_camera_resolution(self):
        """Try common resolutions and pick the highest one the camera supports."""
        resolutions = [
            (3840, 2160),
            (2560, 1440),
            (1920, 1080),
            (1280, 720),
            (640, 480)
        ]
        
        original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        max_width, max_height = original_width, original_height
        
        print("Checking available camera resolutions...")
        for width, height in resolutions:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Tried: {width}x{height}, Got: {actual_width}x{actual_height}")
            
            if (actual_width >= max_width and actual_height >= max_height and actual_width > 0 and actual_height > 0):
                max_width, max_height = actual_width, actual_height
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Using camera resolution: {self.width}x{self.height}")
    
    def load_reference_images(self):
        """Load or create reference images used for side-by-side comparison."""
        os.makedirs('reference_images', exist_ok=True)
        
        for exercise in self.reference_images:
            for position in self.reference_images[exercise]:
                file_path = f'reference_images/{exercise}_{position}.jpg'
                
                if os.path.exists(file_path):
                    try:
                        img = cv2.imread(file_path)
                        if img is not None:
                            self.reference_images[exercise][position] = img
                            print(f"Loaded reference image: {file_path}")
                        else:
                            print(f"Failed to load reference image: {file_path}")
                    except Exception as e:
                        print(f"Error loading reference image {file_path}: {e}")
                else:
                    print(f"Reference image not found: {file_path}")
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    text = f"{exercise.replace('_', ' ').title()} - {position.title()} Position"
                    cv2.putText(placeholder, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(placeholder, "Reference image not found", (50, 280), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    try:
                        cv2.imwrite(file_path, placeholder)
                        self.reference_images[exercise][position] = placeholder
                        print(f"Created placeholder for reference image: {file_path}")
                    except Exception as e:
                        print(f"Error creating placeholder for {file_path}: {e}")
    
    def toggle_comparison(self):
        """Enable or disable the reference image overlay."""
        self.show_comparison = self.comparison_var.get()
        comparison_text = "Show Reference Image" if self.show_comparison else "Show Reference Image"
        self.comparison_check.config(text=comparison_text)
    
    def update_scale(self, value):
        """Adjust displayed video scale label when slider moves."""
        scale = float(value)
        self.scale_label.config(text=f"Display Scale: {int(scale*100)}%")
    
    def update_frame(self):
        """Main frame loop triggered by Tk's after method."""
        frame = self.process_frame()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            scale = self.scale_var.get()
            display_width = int(self.width * scale)
            display_height = int(self.height * scale)
            
            if display_width > 0 and display_height > 0:
                frame_resized = cv2.resize(frame_rgb, (display_width, display_height))
                img = Image.fromarray(frame_resized)
                self.photo = ImageTk.PhotoImage(image=img)
                
                self.video_canvas.config(width=display_width, height=display_height)
                self.video_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.update_stats()
            
        self.root.after(10, self.update_frame)
    
    def draw_colored_skeleton(self, frame, landmarks, mp_pose):
        """Draw the pose skeleton and highlight any problem areas in color."""
        h, w, c = frame.shape
        overlay = np.zeros((h, w, c), dtype=np.uint8)
        
        # Default (good) connection color - green in BGR
        default_color = (0, 255, 0)
        
        # Draw base landmark connections with default styling
        self.mp_drawing.draw_landmarks(
            overlay, 
            landmarks, 
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=default_color[::-1], thickness=3, circle_radius=2)
        )
        
        # Body part pairs to highlight (indices from MediaPipe landmark enum)
        body_parts = {
            'right_arm': [
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
            ],
            'left_arm': [
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST)
            ],
            'right_leg': [
                (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
            ],
            'left_leg': [
                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE)
            ],
            'torso': [
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP)
            ]
        }
        
        # Draw problem-area lines thicker and in the specified color
        for area, color in self.problem_areas.items():
            if area in body_parts:
                for start_idx, end_idx in body_parts[area]:
                    start_point = landmarks.landmark[start_idx]
                    end_point = landmarks.landmark[end_idx]
                    
                    start_x, start_y = int(start_point.x * w), int(start_point.y * h)
                    end_x, end_y = int(end_point.x * w), int(end_point.y * h)
                    
                    cv2.line(
                        overlay,
                        (start_x, start_y),
                        (end_x, end_y),
                        color[::-1],
                        thickness=5
                    )
                    
                    cv2.circle(overlay, (start_x, start_y), 4, color[::-1], -1)
                    cv2.circle(overlay, (end_x, end_y), 4, color[::-1], -1)
        
        # Blend overlay with the frame at moderate transparency
        alpha = 0.5
        result_frame = frame.copy()
        cv2.addWeighted(overlay, alpha, result_frame, 1.0, 0, result_frame)
        
        return result_frame
    
    def create_side_by_side_view(self, frame, stage):
        """Attach a reference image in the top-right corner for visual comparison."""
        if not self.show_comparison or stage is None:
            return frame
        
        reference_img = self.reference_images[self.current_exercise][stage]
        if reference_img is None:
            return frame
        
        h, w, _ = frame.shape
        ref_h, ref_w, _ = reference_img.shape
        ref_height = int(h * 0.25)
        ref_width = int((ref_height / ref_h) * ref_w)
        reference_resized = cv2.resize(reference_img, (ref_width, ref_height))
        
        result_frame = frame.copy()
        x_offset = w - ref_width - 10
        y_offset = 10
        
        # Dark background rectangle for clarity
        cv2.rectangle(result_frame, (x_offset-5, y_offset-5), 
                     (x_offset+ref_width+5, y_offset+ref_height+5), 
                     (0, 0, 0), -1)
        
        result_frame[y_offset:y_offset+ref_height, x_offset:x_offset+ref_width] = reference_resized
        cv2.rectangle(result_frame, (x_offset-2, y_offset-2), 
                     (x_offset+ref_width+2, y_offset+ref_height+2), 
                     (255, 255, 255), 2)
        
        cv2.putText(result_frame, "Ideal Form", (x_offset+5, y_offset-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return result_frame
    
    def process_frame(self):
        """Grab a frame, run pose detection, update detectors, and render overlays."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        original_frame = frame.copy()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            detector = self.exercise_detectors[self.current_exercise]
            old_rep_count = detector.rep_count
            rep_count, status, feedback, color = detector.process(results.pose_landmarks, self.mp_pose)
            
            # Extract diagnostics from detector
            self.problem_areas = detector.get_problem_areas()
            self.form_score = detector.get_form_score()
            
            # Draw skeleton with problem highlights
            frame = self.draw_colored_skeleton(original_frame, results.pose_landmarks, self.mp_pose)
            
            # Play context-aware audio without spamming
            if self.audio_var.get() and time.time() - self.last_audio_time > 1.5:
                if rep_count > old_rep_count:
                    self.play_sound('good_rep')
                    self.last_audio_time = time.time()
                elif feedback != "Good form" and feedback != "Good job!" and self.feedback != feedback:
                    self.play_sound('bad_form')
                    self.last_audio_time = time.time()
                elif (feedback == "Good form" or feedback == "Good job!") and self.feedback != feedback:
                    self.play_sound('good_form')
                    self.last_audio_time = time.time()
            
            self.rep_count = rep_count
            self.status = status
            self.feedback = feedback
            
            # Overlay text info
            cv2.putText(frame, f'Exercise: {self.current_exercise.replace("_", " ").title()}', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Reps: {self.rep_count}', 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Status: {self.status}', 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            if self.feedback:
                cv2.putText(frame, f'Feedback: {self.feedback}', 
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            cv2.putText(frame, f'Form Score: {self.form_score}%', 
                        (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            
            # Map status text to an 'up' or 'down' reference stage for comparison
            current_stage = None
            if 'Up' in self.status or self.status == 'Standing' or self.status == 'Arm Extended':
                current_stage = 'up'
            elif 'Down' in self.status or self.status == 'Squatting' or self.status == 'Arm Curled':
                current_stage = 'down'
            
            if self.show_comparison and current_stage:
                frame = self.create_side_by_side_view(frame, current_stage)
        else:
            frame = original_frame
        
        return frame
    
    def update_stats(self):
        """Refresh the GUI stats labels from the latest detector state."""
        self.exercise_label.config(text=f"Exercise: {self.current_exercise.replace('_', ' ').title()}")
        self.reps_label.config(text=f"Reps: {self.rep_count}")
        self.status_label.config(text=f"Status: {self.status}")
        self.feedback_label.config(text=f"Feedback: {self.feedback}")
        self.score_label.config(text=f"Form Score: {self.form_score}%")
        self.update_analytics_display()
    
    def update_analytics_display(self):
        """Pull analytics from the detector and display them in the panel."""
        detector = self.exercise_detectors[self.current_exercise]
        analytics = detector.get_detailed_analytics()
        
        speed = round(analytics['movement_speed'], 1)
        self.speed_label.config(text=f"Movement Speed: {speed} deg/sec")
        
        rom = round(analytics['range_of_motion'], 1)
        self.rom_label.config(text=f"Range of Motion: {rom}°")
        
        stability = round(analytics['stability_score'])
        self.stability_label.config(text=f"Stability: {stability}%")
        
        calories = round(analytics['calories_burned'], 2)
        self.calories_label.config(text=f"Calories Burned: {calories} cal")
    
    def show_detailed_feedback(self):
        """Open a scrollable window showing per-metric feedback and performance stats."""
        detector = self.exercise_detectors[self.current_exercise]
        analytics = detector.get_detailed_analytics()
        detailed_feedback = analytics.get('detailed_feedback', {})
        
        feedback_window = tk.Toplevel(self.root)
        feedback_window.title(f"Detailed Feedback - {self.current_exercise.replace('_', ' ').title()}")
        feedback_window.geometry("600x500")
        
        main_frame = ttk.Frame(feedback_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        title_label = ttk.Label(scrollable_frame, 
                               text=f"{self.current_exercise.replace('_', ' ').title()} Analysis", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        score_label = ttk.Label(scrollable_frame, 
                               text=f"Overall Form Score: {self.form_score}%", 
                               font=("Arial", 12))
        score_label.pack(pady=5)
        
        avg_rep_duration = analytics.get('avg_rep_duration', 0)
        
        reps_frame = ttk.Frame(scrollable_frame)
        reps_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(reps_frame, text=f"Total Reps: {self.rep_count}", 
                 font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        
        ttk.Label(reps_frame, text=f"Avg Rep Duration: {round(avg_rep_duration, 1)}s", 
                 font=("Arial", 10)).pack(side=tk.RIGHT, padx=10)
        
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        if detailed_feedback:
            feedback_label = ttk.Label(scrollable_frame, text="Detailed Form Analysis", 
                                      font=("Arial", 12, "bold"))
            feedback_label.pack(pady=5, anchor=tk.W)
            
            for metric, data in detailed_feedback.items():
                if isinstance(data, dict) and 'status' in data:
                    metric_frame = ttk.LabelFrame(scrollable_frame, 
                                                 text=metric.replace('_', ' ').title())
                    metric_frame.pack(fill=tk.X, pady=5, padx=5)
                    
                    status_color = "green"
                    if data['status'] == 'warning':
                        status_color = "orange"
                    elif data['status'] == 'bad':
                        status_color = "red"
                    
                    status_label = ttk.Label(metric_frame, text=f"Status: {data['status'].title()}")
                    status_label.pack(anchor=tk.W, padx=5, pady=2)
                    
                    value_text = f"Value: {round(data['value'], 2)}"
                    if 'ideal_range' in data:
                        value_text += f" (Ideal: {data['ideal_range'][0]}-{data['ideal_range'][1]})"
                    
                    value_label = ttk.Label(metric_frame, text=value_text)
                    value_label.pack(anchor=tk.W, padx=5, pady=2)
                    
                    if data['message']:
                        msg_label = ttk.Label(metric_frame, text=f"Feedback: {data['message']}")
                        msg_label.pack(anchor=tk.W, padx=5, pady=2)
        
        perf_label = ttk.Label(scrollable_frame, text="Performance Metrics", 
                              font=("Arial", 12, "bold"))
        perf_label.pack(pady=10, anchor=tk.W)
        
        speed_frame = ttk.Frame(scrollable_frame)
        speed_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(speed_frame, text="Movement Speed:", width=20, 
                 anchor=tk.W).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(speed_frame, 
                 text=f"{round(analytics['movement_speed'], 1)} deg/sec").pack(side=tk.LEFT)
        
        rom_frame = ttk.Frame(scrollable_frame)
        rom_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(rom_frame, text="Range of Motion:", width=20, 
                 anchor=tk.W).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(rom_frame, 
                 text=f"{round(analytics['max_range_of_motion'], 1)}°").pack(side=tk.LEFT)
        
        stability_frame = ttk.Frame(scrollable_frame)
        stability_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(stability_frame, text="Stability Score:", width=20, 
                 anchor=tk.W).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(stability_frame, 
                 text=f"{round(analytics['stability_score'])}%").pack(side=tk.LEFT)
        
        calories_frame = ttk.Frame(scrollable_frame)
        calories_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(calories_frame, text="Calories Burned:", width=20, 
                 anchor=tk.W).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(calories_frame, 
                 text=f"{round(analytics['calories_burned'], 2)} cal").pack(side=tk.LEFT)
        
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        settings_label = ttk.Label(scrollable_frame, text="User Settings", 
                                  font=("Arial", 12, "bold"))
        settings_label.pack(pady=5, anchor=tk.W)
        
        user_frame = ttk.Frame(scrollable_frame)
        user_frame.pack(fill=tk.X, pady=5)
        
        # Prefill weight/age/gender from detector defaults
        weight_var = tk.StringVar(value=str(detector.user_weight_kg))
        ttk.Label(user_frame, text="Weight (kg):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        weight_entry = ttk.Entry(user_frame, textvariable=weight_var, width=10)
        weight_entry.grid(row=0, column=1, padx=5, pady=5)
        
        age_var = tk.StringVar(value=str(detector.user_age))
        ttk.Label(user_frame, text="Age:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        age_entry = ttk.Entry(user_frame, textvariable=age_var, width=10)
        age_entry.grid(row=1, column=1, padx=5, pady=5)
        
        gender_var = tk.StringVar(value=detector.user_gender.title())
        ttk.Label(user_frame, text="Gender:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        gender_combo = ttk.Combobox(user_frame, textvariable=gender_var, 
                                   values=["Male", "Female"], width=10, state="readonly")
        gender_combo.grid(row=2, column=1, padx=5, pady=5)
        
        def update_user_data():
            """Validate and push user settings to all detectors."""
            try:
                weight = float(weight_var.get())
                age = int(age_var.get())
                gender = gender_var.get().lower()
                
                for detector in self.exercise_detectors.values():
                    detector.set_user_data(weight_kg=weight, age=age, gender=gender)
                
                messagebox.showinfo("Success", "User data updated successfully!")
                self.update_analytics_display()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers for weight and age.")
        
        update_button = ttk.Button(user_frame, text="Update", command=update_user_data)
        update_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10, sticky=tk.E)
        
        close_button = ttk.Button(scrollable_frame, text="Close", command=feedback_window.destroy)
        close_button.pack(pady=10)
    
    def change_exercise(self, exercise):
        """Switch active exercise and reset related state."""
        if exercise != self.current_exercise:
            self.current_exercise = exercise
            self.rep_count = 0
            self.status = "Ready"
            self.feedback = ""
            self.form_score = 100
            self.problem_areas = {}
            self.exercise_detectors[self.current_exercise].reset()
            self.update_stats()
    
    def reset_counter(self):
        """Zero the rep counter and reset form tracking for the current exercise."""
        self.rep_count = 0
        self.status = "Ready"
        self.feedback = ""
        self.form_score = 100
        self.problem_areas = {}
        self.exercise_detectors[self.current_exercise].reset()
        self.update_stats()
    
    def show_tutorial(self):
        """Open a small window with step-by-step instructions for the selected exercise."""
        tutorials = {
            'squat': """
            SQUAT TUTORIAL:
            
            1. Stand with feet shoulder-width apart
            2. Keep your back straight
            3. Bend your knees and lower your body as if sitting in a chair
            4. Keep knees behind toes
            5. Lower until thighs are parallel to ground
            6. Push through heels to return to standing
            
            Common mistakes:
            - Letting knees go too far forward
            - Not squatting deep enough
            - Rounding your back
            """,
            
            'bicep_curl': """
            BICEP CURL TUTORIAL:
            
            1. Stand with feet shoulder-width apart
            2. Hold weights with arms fully extended
            3. Keep elbows close to your sides
            4. Curl weights up while keeping upper arms stationary
            5. Squeeze biceps at top of movement
            6. Lower weights slowly back to starting position
            
            Common mistakes:
            - Swinging the weights (using momentum)
            - Moving your elbows forward
            - Not fully extending arms at bottom
            """,
            
            'push_up': """
            PUSH-UP TUTORIAL:
            
            1. Start in plank position with hands slightly wider than shoulders
            2. Keep your body in a straight line from head to heels
            3. Lower your body until chest nearly touches the floor
            4. Keep elbows at about 45-degree angle to your body
            5. Push back up to starting position
            
            Common mistakes:
            - Sagging or arching your lower back
            - Not lowering far enough
            - Flaring elbows too far out
            """
        }
        
        tutorial_window = tk.Toplevel(self.root)
        tutorial_window.title(f"{self.current_exercise.replace('_', ' ').title()} Tutorial")
        tutorial_window.geometry("500x500")
        
        tutorial_text = tk.Text(tutorial_window, wrap=tk.WORD, padx=10, pady=10)
        tutorial_text.pack(fill=tk.BOTH, expand=True)
        tutorial_text.insert(tk.END, tutorials[self.current_exercise])
        tutorial_text.config(state=tk.DISABLED)
        
        close_button = ttk.Button(tutorial_window, text="Close", command=tutorial_window.destroy)
        close_button.pack(pady=10)
    
    def play_sound(self, sound_type):
        """Attempt to play an audio cue if the file exists and is non-empty."""
        try:
            if os.path.getsize(self.audio_files[sound_type]) > 0:
                pygame.mixer.music.load(self.audio_files[sound_type])
                pygame.mixer.music.play()
        except:
            # Ignore audio failures silently (no crash from missing/corrupt files)
            pass
    
    def on_closing(self):
        """Release camera and destroy main window on exit."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def change_camera(self, event):
        """Switch the video source when user picks a different camera from the dropdown."""
        selected_camera = self.camera_var.get()
        self.current_camera_index = self.available_cameras[selected_camera]
        
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.current_camera_index)
        self.setup_camera_resolution()
        self.update_frame()

if __name__ == "__main__":
    print("Starting AI Gym Trainer GUI...")
    root = tk.Tk()
    app = AIGymTrainerGUI(root)
    root.mainloop()
