import numpy as np
import time

class ExerciseDetector:
    """Base class for exercise detection"""
    
    def __init__(self):
        self.rep_count = 0
        self.status = "Ready"
        self.feedback = ""
        # Colors for feedback (BGR format)
        self.color_correct = (0, 255, 0)    # Bright Green
        self.color_incorrect = (0, 0, 255)  # Bright Red
        self.color_warning = (0, 255, 255)  # Bright Yellow
        self.color_moderate = (0, 165, 255) # Bright Orange
        self.current_color = (0, 0, 0)      # Black
        
        # Form scoring
        self.form_score = 100
        
        # Problem areas for color coding
        self.problem_areas = {}
        
        # Advanced analytics
        self.analytics = {
            'rep_durations': [],            # Duration of each rep in seconds
            'range_of_motion': 0,           # Current ROM in degrees
            'max_range_of_motion': 0,       # Maximum ROM achieved in degrees
            'stability_score': 100,         # Stability score (0-100)
            'movement_speed': 0,            # Current speed in degrees/second
            'avg_movement_speed': 0,        # Average speed in degrees/second
            'calories_burned': 0,           # Estimated calories burned
            'detailed_feedback': {}         # Detailed feedback on specific issues
        }
        
        # For speed and time tracking
        self.last_angle = 0
        self.last_time = time.time()
        self.rep_start_time = 0
        
        # User data for calorie calculations (default values)
        self.user_weight_kg = 70  # Default 70kg
        self.user_age = 30        # Default 30 years
        self.user_gender = 'male' # Default male
        
        # MET values for different exercises (metabolic equivalent of task)
        self.met_values = {
            'squat': 5.0,        # Moderate squats
            'bicep_curl': 3.5,   # Light weightlifting  
            'push_up': 4.0       # Moderate calisthenics
        }
    
    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def update_analytics(self, angle, exercise_type, rep_completed=False):
        """Update analytics based on current movement"""
        current_time = time.time()
        time_diff = current_time - self.last_time
        
        # Track range of motion
        if self.analytics['max_range_of_motion'] == 0:
            # Initialize with first angle
            self.analytics['max_range_of_motion'] = angle
        else:
            # Track the range of motion for this rep
            rom = abs(self.last_angle - angle)
            self.analytics['range_of_motion'] = rom
            self.analytics['max_range_of_motion'] = max(self.analytics['max_range_of_motion'], rom)
        
        # Calculate movement speed (degrees per second)
        if time_diff > 0:
            angle_change = abs(angle - self.last_angle)
            current_speed = angle_change / time_diff
            self.analytics['movement_speed'] = current_speed
            
            # Update average speed
            if len(self.analytics['rep_durations']) > 0:
                self.analytics['avg_movement_speed'] = (
                    self.analytics['avg_movement_speed'] * len(self.analytics['rep_durations']) + current_speed
                ) / (len(self.analytics['rep_durations']) + 1)
            else:
                self.analytics['avg_movement_speed'] = current_speed
        
        # Start tracking rep time
        if self.rep_start_time == 0:
            self.rep_start_time = current_time
        
        # When a rep is completed
        if rep_completed:
            rep_duration = current_time - self.rep_start_time
            self.analytics['rep_durations'].append(rep_duration)
            
            # Reset rep start time
            self.rep_start_time = current_time
            
            # Calculate calories burned
            # Formula: MET value * weight in kg * duration in hours
            exercise_met = self.met_values.get(exercise_type, 4.0)  # Default to 4.0 if not found
            hours = rep_duration / 3600  # Convert seconds to hours
            calories_this_rep = exercise_met * self.user_weight_kg * hours
            
            # Add to total calories
            self.analytics['calories_burned'] += calories_this_rep
        
        # Update last values for next calculation
        self.last_angle = angle
        self.last_time = current_time
        
        return self.analytics
    
    def calculate_stability(self, landmarks, key_points):
        """Calculate stability score based on movement of non-primary joints"""
        stability_score = 100
        
        # Check for excess movement in landmarks that should remain stable
        for point in key_points:
            # Calculate movement between frames
            if hasattr(self, f'last_{point.name}_pos'):
                last_pos = getattr(self, f'last_{point.name}_pos')
                current_pos = np.array([landmarks.landmark[point].x, landmarks.landmark[point].y])
                
                # Calculate Euclidean distance between last and current position
                distance = np.linalg.norm(current_pos - last_pos)
                
                # Penalize stability score for excess movement
                # Scale factor can be adjusted based on sensitivity
                stability_penalty = min(distance * 1000, 20)  # Cap penalty at 20 points
                stability_score -= stability_penalty
                
                # Store current position
                setattr(self, f'last_{point.name}_pos', current_pos)
            else:
                # First frame, initialize position
                setattr(self, f'last_{point.name}_pos', 
                        np.array([landmarks.landmark[point].x, landmarks.landmark[point].y]))
        
        # Ensure stability score stays within 0-100 range
        stability_score = max(0, min(100, stability_score))
        
        # Update stability score in analytics
        self.analytics['stability_score'] = stability_score
        
        return stability_score
    
    def get_detailed_analytics(self):
        """Get the detailed analytics data"""
        # Calculate averages for time-based metrics
        if len(self.analytics['rep_durations']) > 0:
            self.analytics['avg_rep_duration'] = sum(self.analytics['rep_durations']) / len(self.analytics['rep_durations'])
        else:
            self.analytics['avg_rep_duration'] = 0
            
        return self.analytics
    
    def set_user_data(self, weight_kg=None, age=None, gender=None):
        """Set user data for more accurate calorie calculations"""
        if weight_kg:
            self.user_weight_kg = weight_kg
        if age:
            self.user_age = age
        if gender:
            self.user_gender = gender.lower()
    
    def reset(self):
        """Reset the detector state"""
        self.rep_count = 0
        self.status = "Ready"
        self.feedback = ""
        self.current_color = (0, 0, 0)
        self.form_score = 100
        self.problem_areas = {}
        
        # Reset analytics
        self.analytics = {
            'rep_durations': [],
            'range_of_motion': 0,
            'max_range_of_motion': 0,
            'stability_score': 100,
            'movement_speed': 0,
            'avg_movement_speed': 0,
            'calories_burned': 0,
            'detailed_feedback': {}
        }
        
        self.last_angle = 0
        self.last_time = time.time()
        self.rep_start_time = 0
    
    def get_form_score(self):
        """Get the current form score (0-100)"""
        return self.form_score
    
    def get_problem_areas(self):
        """Get the current problem areas with their associated colors"""
        return self.problem_areas
    
    def process(self, landmarks, mp_pose):
        """Process landmarks and detect exercise (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement this method")


class SquatDetector(ExerciseDetector):
    """Detector for squat exercise"""
    
    def __init__(self):
        super().__init__()
        self.stage = None  # None, "down", or "up"
        self.min_angle = 180  # Track minimum angle during squat
        self.deductions = {
            'knee_alignment': 0,  # Knees too far forward
            'squat_depth': 0,     # Not squatting deep enough
            'hip_alignment': 0    # Hips not aligned
        }
        
        # Detailed feedback dictionary
        self.detailed_feedback = {
            'knee_alignment': {
                'status': 'good',  # 'good', 'warning', 'bad'
                'message': '',
                'value': 0.0,      # Distance measure for knees
                'ideal_range': (0.0, 0.05)  # Ideal range for knee forward movement
            },
            'squat_depth': {
                'status': 'good',
                'message': '',
                'value': 0.0,      # Current depth (angle)
                'ideal_range': (80, 100)  # Ideal range for squat depth angle
            },
            'hip_alignment': {
                'status': 'good',
                'message': '',
                'value': 0.0,      # Hip alignment measure
                'ideal_range': (0.0, 0.02)  # Ideal range for hip level difference
            },
            'movement_speed': {
                'status': 'good',
                'message': '',
                'value': 0.0,      # Current speed (degrees/second)
                'ideal_range': (30, 60)  # Ideal range for squat speed
            },
            'stability': {
                'status': 'good',
                'message': '',
                'value': 100,      # Stability score
                'ideal_range': (85, 100)  # Ideal range for stability
            }
        }
    
    def process(self, landmarks, mp_pose):
        # Get coordinates for right and left side
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        # Calculate knee angles
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        
        # Use the average of both knee angles
        knee_angle = (right_knee_angle + left_knee_angle) / 2
        
        # Calculate back angle (for posture analysis)
        right_back_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
        left_back_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        back_angle = (right_back_angle + left_back_angle) / 2
        
        # Update analytics
        self.update_analytics(knee_angle, 'squat')
        
        # Calculate stability with key points that should remain stable during squat
        stable_points = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP
        ]
        stability_score = self.calculate_stability(landmarks, stable_points)
        
        # Reset deductions for this frame
        self.deductions = {
            'knee_alignment': 0,
            'squat_depth': 0,
            'hip_alignment': 0
        }
        
        # Problem areas dict is cleared each frame
        self.problem_areas = {}
        
        # Detect squat stages
        if knee_angle > 160:
            if self.stage == "down":
                self.rep_count += 1
                # Mark rep completion for analytics
                self.update_analytics(knee_angle, 'squat', rep_completed=True)
                self.feedback = "Good job!"
                self.current_color = self.color_correct
            self.stage = "up"
            self.status = "Standing"
            # Reset minimum angle for next rep
            self.min_angle = 180
        elif knee_angle < 100:
            self.stage = "down"
            self.status = "Squatting"
            # Update minimum angle
            if knee_angle < self.min_angle:
                self.min_angle = knee_angle
        
        # Provide feedback based on form
        if self.stage == "down" or (self.stage == "up" and knee_angle <= 160):
            # Check if knees are properly aligned (not going too far forward)
            right_knee_x = right_knee.x
            right_ankle_x = right_ankle.x
            left_knee_x = left_knee.x
            left_ankle_x = left_ankle.x
            
            # Calculate knee forward movement measures
            right_knee_forward = max(0, right_knee_x - right_ankle_x)
            left_knee_forward = max(0, left_knee_x - left_ankle_x)
            avg_knee_forward = (right_knee_forward + left_knee_forward) / 2
            
            # Update detailed feedback for knee alignment
            self.detailed_feedback['knee_alignment']['value'] = avg_knee_forward
            
            # Check if knees are going too far forward (past toes)
            if right_knee_x > right_ankle_x + 0.1 or left_knee_x > left_ankle_x + 0.1:
                self.feedback = "Keep knees behind toes"
                self.current_color = self.color_incorrect
                self.deductions['knee_alignment'] = 30
                # Mark both legs as problem areas (red for serious issues)
                self.problem_areas['right_leg'] = self.color_incorrect
                self.problem_areas['left_leg'] = self.color_incorrect
                
                # Update detailed feedback for knee alignment
                self.detailed_feedback['knee_alignment']['status'] = 'bad'
                self.detailed_feedback['knee_alignment']['message'] = "Knees are too far forward"
            elif right_knee_x > right_ankle_x + 0.05 or left_knee_x > left_ankle_x + 0.05:
                # Minor knee alignment issue (yellow warning)
                if not self.problem_areas:
                    self.feedback = "Keep knees more aligned with ankles"
                    self.current_color = self.color_warning
                self.deductions['knee_alignment'] = 15
                self.problem_areas['right_leg'] = self.color_warning
                self.problem_areas['left_leg'] = self.color_warning
                
                # Update detailed feedback
                self.detailed_feedback['knee_alignment']['status'] = 'warning'
                self.detailed_feedback['knee_alignment']['message'] = "Knees slightly forward"
            else:
                # Good knee alignment
                self.detailed_feedback['knee_alignment']['status'] = 'good'
                self.detailed_feedback['knee_alignment']['message'] = "Good knee alignment"
            
            # Check if not squatting deep enough
            # Update detailed feedback for squat depth
            self.detailed_feedback['squat_depth']['value'] = self.min_angle
            
            if self.min_angle > 110 and self.stage == "down":
                self.feedback = "Try to squat deeper"
                self.current_color = self.color_warning
                self.deductions['squat_depth'] = 20
                # Mark both legs as problem areas (yellow for moderate issues)
                self.problem_areas['right_leg'] = self.color_warning
                self.problem_areas['left_leg'] = self.color_warning
                
                # Update detailed feedback
                self.detailed_feedback['squat_depth']['status'] = 'warning'
                self.detailed_feedback['squat_depth']['message'] = "Squat depth is insufficient"
            elif self.min_angle > 90 and self.stage == "down":
                # Decent but not optimal depth
                self.detailed_feedback['squat_depth']['status'] = 'good'
                self.detailed_feedback['squat_depth']['message'] = "Good squat depth"
            else:
                # Excellent depth
                self.detailed_feedback['squat_depth']['status'] = 'good'
                self.detailed_feedback['squat_depth']['message'] = "Excellent squat depth"
            
            # Check hip alignment (should be parallel to ground at bottom)
            hip_alignment_diff = abs(right_hip.y - left_hip.y)
            self.detailed_feedback['hip_alignment']['value'] = hip_alignment_diff
            
            if hip_alignment_diff > 0.05:
                if not self.problem_areas:  # Only if no other problems detected
                    self.feedback = "Keep hips level"
                    self.current_color = self.color_moderate
                self.deductions['hip_alignment'] = 15
                # Mark torso as a problem area (orange for minor issues)
                self.problem_areas['torso'] = self.color_moderate
                
                # Update detailed feedback
                self.detailed_feedback['hip_alignment']['status'] = 'warning'
                self.detailed_feedback['hip_alignment']['message'] = "Hips are not level"
            else:
                # Good hip alignment
                self.detailed_feedback['hip_alignment']['status'] = 'good'
                self.detailed_feedback['hip_alignment']['message'] = "Good hip alignment"
            
            # Check back posture
            if back_angle < 150 and self.stage == "down":
                feedback_msg = "Keep your back straighter"
                if not self.problem_areas:
                    self.feedback = feedback_msg
                    self.current_color = self.color_warning
                self.problem_areas['torso'] = self.color_warning
            
            # Check movement speed
            speed = self.analytics['movement_speed']
            self.detailed_feedback['movement_speed']['value'] = speed
            
            if speed > 70:  # Too fast
                self.detailed_feedback['movement_speed']['status'] = 'warning'
                self.detailed_feedback['movement_speed']['message'] = "Movement too fast"
                if not self.problem_areas:
                    self.feedback = "Slow down, control the movement"
                    self.current_color = self.color_warning
            elif speed < 20 and speed > 0:  # Too slow
                self.detailed_feedback['movement_speed']['status'] = 'warning'
                self.detailed_feedback['movement_speed']['message'] = "Movement too slow"
            else:  # Good speed
                self.detailed_feedback['movement_speed']['status'] = 'good'
                self.detailed_feedback['movement_speed']['message'] = "Good movement speed"
            
            # Check stability
            self.detailed_feedback['stability']['value'] = stability_score
            
            if stability_score < 70:
                self.detailed_feedback['stability']['status'] = 'bad'
                self.detailed_feedback['stability']['message'] = "Excessive movement, improve stability"
                if not self.problem_areas:
                    self.feedback = "Stay more stable"
                    self.current_color = self.color_warning
            elif stability_score < 85:
                self.detailed_feedback['stability']['status'] = 'warning'
                self.detailed_feedback['stability']['message'] = "Moderate stability, try to reduce movement"
            else:
                self.detailed_feedback['stability']['status'] = 'good'
                self.detailed_feedback['stability']['message'] = "Good stability"
            
            # If no issues detected, provide positive feedback
            if not self.problem_areas:
                self.feedback = "Good form"
                self.current_color = self.color_correct
                # Mark legs as good (green)
                self.problem_areas['right_leg'] = self.color_correct
                self.problem_areas['left_leg'] = self.color_correct
                self.problem_areas['torso'] = self.color_correct
        
        # Calculate form score based on deductions
        total_deduction = sum(self.deductions.values())
        self.form_score = max(0, 100 - total_deduction)
        
        # Update analytics with detailed feedback
        self.analytics['detailed_feedback'] = self.detailed_feedback
        
        return self.rep_count, self.status, self.feedback, self.current_color
    
    def reset(self):
        super().reset()
        self.stage = None
        self.min_angle = 180
        self.deductions = {
            'knee_alignment': 0,
            'squat_depth': 0,
            'hip_alignment': 0
        }
        
        # Reset detailed feedback
        self.detailed_feedback = {
            'knee_alignment': {
                'status': 'good',
                'message': '',
                'value': 0.0,
                'ideal_range': (0.0, 0.05)
            },
            'squat_depth': {
                'status': 'good',
                'message': '',
                'value': 0.0,
                'ideal_range': (80, 100)
            },
            'hip_alignment': {
                'status': 'good',
                'message': '',
                'value': 0.0,
                'ideal_range': (0.0, 0.02)
            },
            'movement_speed': {
                'status': 'good',
                'message': '',
                'value': 0.0,
                'ideal_range': (30, 60)
            },
            'stability': {
                'status': 'good',
                'message': '',
                'value': 100,
                'ideal_range': (85, 100)
            }
        }


class BicepCurlDetector(ExerciseDetector):
    """Detector for bicep curl exercise"""
    
    def __init__(self):
        super().__init__()
        self.stage = None  # None, "down", or "up"
        self.last_elbow_angle = 0
        self.deductions = {
            'arm_swing': 0,      # Swinging the arm
            'full_extension': 0,  # Not fully extending
            'curl_speed': 0       # Too fast/jerky movement
        }
        
        # Detailed feedback dictionary
        self.detailed_feedback = {
            'arm_swing': {
                'status': 'good',  # 'good', 'warning', 'bad'
                'message': '',
                'value': 0.0,      # Shoulder deviation measure
                'ideal_range': (0.0, 0.05)  # Ideal range for shoulder deviation
            },
            'full_extension': {
                'status': 'good',
                'message': '',
                'value': 0.0,      # Curl angle at top
                'ideal_range': (20, 40)  # Ideal range for full curl
            },
            'curl_speed': {
                'status': 'good',
                'message': '',
                'value': 0.0,      # Speed measure (angle change)
                'ideal_range': (0, 10)  # Ideal range for controlled movement
            },
            'movement_speed': {
                'status': 'good',
                'message': '',
                'value': 0.0,      # Current speed (degrees/second)
                'ideal_range': (30, 60)  # Ideal range for bicep curl speed
            },
            'stability': {
                'status': 'good',
                'message': '',
                'value': 100,      # Stability score
                'ideal_range': (85, 100)  # Ideal range for stability
            }
        }
    
    def process(self, landmarks, mp_pose):
        # Get coordinates for right arm
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Get coordinates for left arm
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        
        # Calculate elbow angles
        right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Use the smaller angle (the arm that's curling)
        elbow_angle = min(right_elbow_angle, left_elbow_angle)
        is_right_arm = right_elbow_angle <= left_elbow_angle
        
        # Update analytics
        self.update_analytics(elbow_angle, 'bicep_curl')
        
        # Calculate stability with key points that should remain stable during bicep curl
        stable_points = [
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_SHOULDER if not is_right_arm else mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]
        stability_score = self.calculate_stability(landmarks, stable_points)
        
        # Update detailed feedback for stability
        self.detailed_feedback['stability']['value'] = stability_score
        if stability_score < 70:
            self.detailed_feedback['stability']['status'] = 'bad'
            self.detailed_feedback['stability']['message'] = "Excessive movement, improve stability"
        elif stability_score < 85:
            self.detailed_feedback['stability']['status'] = 'warning'
            self.detailed_feedback['stability']['message'] = "Moderate stability, try to reduce movement"
        else:
            self.detailed_feedback['stability']['status'] = 'good'
            self.detailed_feedback['stability']['message'] = "Good stability"
        
        # Reset deductions for this frame
        self.deductions = {
            'arm_swing': 0,
            'full_extension': 0,
            'curl_speed': 0
        }
        
        # Problem areas dict is cleared each frame
        self.problem_areas = {}
        
        # Detect curl stages
        if elbow_angle > 160:
            self.stage = "down"
            self.status = "Arm Extended"
        elif elbow_angle < 30 and self.stage == "down":
            self.stage = "up"
            self.status = "Arm Curled"
            self.rep_count += 1
            # Mark rep completion for analytics
            self.update_analytics(elbow_angle, 'bicep_curl', rep_completed=True)
            self.feedback = "Good job!"
            self.current_color = self.color_correct
        
        # Provide feedback based on form
        if self.stage == "down" or self.stage == "up":
            # Determine which arm is being used
            active_arm = 'right_arm' if is_right_arm else 'left_arm'
            inactive_arm = 'left_arm' if is_right_arm else 'right_arm'
            active_shoulder = right_shoulder if is_right_arm else left_shoulder
            active_elbow = right_elbow if is_right_arm else left_elbow
            
            # Check if arm is swinging (shoulder moving too much)
            shoulder_deviation = abs(active_shoulder.x - active_elbow.x)
            # Update detailed feedback for arm swing
            self.detailed_feedback['arm_swing']['value'] = shoulder_deviation
            
            if shoulder_deviation > 0.1:
                self.feedback = "Keep upper arm still"
                self.current_color = self.color_incorrect
                self.deductions['arm_swing'] = 30
                self.problem_areas[active_arm] = self.color_incorrect
                self.problem_areas['torso'] = self.color_warning  # Yellow for torso
                
                # Update detailed feedback
                self.detailed_feedback['arm_swing']['status'] = 'bad'
                self.detailed_feedback['arm_swing']['message'] = "Upper arm moving too much"
            
            # Check if not curling fully
            elif self.stage == "up" and elbow_angle > 45:
                self.feedback = "Try to curl higher"
                self.current_color = self.color_warning
                self.deductions['full_extension'] = 20
                self.problem_areas[active_arm] = self.color_warning  # Yellow
                
                # Update detailed feedback
                self.detailed_feedback['full_extension']['status'] = 'warning'
                self.detailed_feedback['full_extension']['value'] = elbow_angle
                self.detailed_feedback['full_extension']['message'] = "Not curling high enough"
            
            # Check if going too fast (using momentum)
            angle_change = abs(elbow_angle - self.last_elbow_angle)
            # Update detailed feedback for curl speed
            self.detailed_feedback['curl_speed']['value'] = angle_change
            
            if angle_change > 15:
                self.feedback = "Slow down, control the movement"
                self.current_color = self.color_moderate
                self.deductions['curl_speed'] = 25
                self.problem_areas[active_arm] = self.color_moderate  # Orange
                
                # Update detailed feedback
                self.detailed_feedback['curl_speed']['status'] = 'warning'
                self.detailed_feedback['curl_speed']['message'] = "Movement too fast, use more control"
            
            else:
                self.feedback = "Good form"
                self.current_color = self.color_correct
                # Mark the active arm as good (green)
                self.problem_areas[active_arm] = self.color_correct
                self.problem_areas['torso'] = self.color_correct
                
                # The inactive arm should also be marked
                self.problem_areas[inactive_arm] = self.color_correct
        
        # Store current angle for next frame
        self.last_elbow_angle = elbow_angle
        
        # Calculate form score based on deductions
        total_deduction = sum(self.deductions.values())
        self.form_score = max(0, 100 - total_deduction)
        
        # Update analytics with detailed feedback
        self.analytics['detailed_feedback'] = self.detailed_feedback
        
        return self.rep_count, self.status, self.feedback, self.current_color
    
    def reset(self):
        super().reset()
        self.stage = None
        self.last_elbow_angle = 0
        self.deductions = {
            'arm_swing': 0,
            'full_extension': 0,
            'curl_speed': 0
        }
        
        # Reset detailed feedback
        self.detailed_feedback = {
            'arm_swing': {
                'status': 'good',
                'message': '',
                'value': 0.0,
                'ideal_range': (0.0, 0.05)
            },
            'full_extension': {
                'status': 'good',
                'message': '',
                'value': 0.0,
                'ideal_range': (20, 40)
            },
            'curl_speed': {
                'status': 'good',
                'message': '',
                'value': 0.0,
                'ideal_range': (0, 10)
            },
            'movement_speed': {
                'status': 'good',
                'message': '',
                'value': 0.0,
                'ideal_range': (30, 60)
            },
            'stability': {
                'status': 'good',
                'message': '',
                'value': 100,
                'ideal_range': (85, 100)
            }
        }


class PushUpDetector(ExerciseDetector):
    """Detector for push-up exercise"""
    
    def __init__(self):
        super().__init__()
        self.stage = None  # None, "down", or "up"
        self.body_aligned = True
        self.deductions = {
            'body_alignment': 0,  # Body not straight
            'elbow_angle': 0,     # Not going low enough
            'arm_extension': 0    # Not fully extending
        }
        
        # Detailed feedback dictionary
        self.detailed_feedback = {
            'body_alignment': {
                'status': 'good',  # 'good', 'warning', 'bad'
                'message': '',
                'value': 0.0,      # Body alignment measure (average angle deviation)
                'ideal_range': (160, 180)  # Ideal range for body angle
            },
            'elbow_angle': {
                'status': 'good',
                'message': '',
                'value': 90.0,     # Elbow angle at bottom
                'ideal_range': (70, 90)  # Ideal range for push-up depth
            },
            'arm_extension': {
                'status': 'good',
                'message': '',
                'value': 180.0,    # Elbow extension at top
                'ideal_range': (160, 180)  # Ideal range for full extension
            },
            'movement_speed': {
                'status': 'good',
                'message': '',
                'value': 0.0,      # Current speed (degrees/second)
                'ideal_range': (30, 60)  # Ideal range for push-up speed
            },
            'stability': {
                'status': 'good',
                'message': '',
                'value': 100,      # Stability score
                'ideal_range': (85, 100)  # Ideal range for stability
            }
        }
    
    def process(self, landmarks, mp_pose):
        # Get relevant landmarks
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Calculate elbow angles
        left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Use average elbow angle
        elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        
        # Update analytics
        self.update_analytics(elbow_angle, 'push_up')
        
        # Calculate stability with key points that should remain stable during push-up
        stable_points = [
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE
        ]
        stability_score = self.calculate_stability(landmarks, stable_points)
        
        # Update detailed feedback for stability
        self.detailed_feedback['stability']['value'] = stability_score
        if stability_score < 70:
            self.detailed_feedback['stability']['status'] = 'bad'
            self.detailed_feedback['stability']['message'] = "Excessive movement, improve stability"
        elif stability_score < 85:
            self.detailed_feedback['stability']['status'] = 'warning'
            self.detailed_feedback['stability']['message'] = "Moderate stability, try to reduce movement"
        else:
            self.detailed_feedback['stability']['status'] = 'good'
            self.detailed_feedback['stability']['message'] = "Good stability"
        
        # Calculate angles to check body alignment
        left_shoulder_hip_ankle_angle = self.calculate_angle(left_shoulder, left_hip, left_ankle)
        left_hip_knee_ankle_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        
        right_shoulder_hip_ankle_angle = self.calculate_angle(right_shoulder, right_hip, right_ankle)
        right_hip_knee_ankle_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        # Calculate body alignment values for detailed feedback
        alignment_angles = [
            left_shoulder_hip_ankle_angle,
            left_hip_knee_ankle_angle,
            right_shoulder_hip_ankle_angle,
            right_hip_knee_ankle_angle
        ]
        avg_alignment_deviation = sum([abs(angle - 180) for angle in alignment_angles]) / 4
        self.detailed_feedback['body_alignment']['value'] = avg_alignment_deviation

        # Check body alignment (should be straight like a plank)
        self.body_aligned = (160 < left_shoulder_hip_ankle_angle < 200 and 
                            160 < left_hip_knee_ankle_angle < 200 and
                            160 < right_shoulder_hip_ankle_angle < 200 and
                            160 < right_hip_knee_ankle_angle < 200)
        
        # Reset deductions for this frame
        self.deductions = {
            'body_alignment': 0,
            'elbow_angle': 0,
            'arm_extension': 0
        }
        
        # Problem areas dict is cleared each frame
        self.problem_areas = {}
        
        # Detect push-up stages
        if elbow_angle > 150:
            if self.stage == "down":
                self.rep_count += 1
                # Mark rep completion for analytics
                self.update_analytics(elbow_angle, 'push_up', rep_completed=True)
                if self.body_aligned:
                    self.feedback = "Good job!"
                    self.current_color = self.color_correct
            self.stage = "up"
            self.status = "Up Position"
        elif elbow_angle < 90:
            self.stage = "down"
            self.status = "Down Position"
        
        # Provide feedback based on form
        if not self.body_aligned:
            self.feedback = "Keep body straight"
            self.current_color = self.color_incorrect
            self.deductions['body_alignment'] = 35
            # Red for serious alignment issues
            self.problem_areas['torso'] = self.color_incorrect
            self.problem_areas['right_leg'] = self.color_incorrect
            self.problem_areas['left_leg'] = self.color_incorrect
            
            # Update detailed feedback
            self.detailed_feedback['body_alignment']['status'] = 'bad'
            self.detailed_feedback['body_alignment']['message'] = "Body not aligned properly"
        elif self.stage == "down" and elbow_angle > 110:
            self.feedback = "Lower your body more"
            self.current_color = self.color_warning
            self.deductions['elbow_angle'] = 25
            # Yellow for depth issues
            self.problem_areas['right_arm'] = self.color_warning
            self.problem_areas['left_arm'] = self.color_warning
            
            # Update detailed feedback
            self.detailed_feedback['elbow_angle']['status'] = 'warning'
            self.detailed_feedback['elbow_angle']['message'] = "Not lowering deep enough"
        elif self.stage == "up" and elbow_angle < 150:
            self.feedback = "Extend arms fully"
            self.current_color = self.color_moderate
            self.deductions['arm_extension'] = 15
            # Orange for extension issues
            self.problem_areas['right_arm'] = self.color_moderate
            self.problem_areas['left_arm'] = self.color_moderate
            
            # Update detailed feedback
            self.detailed_feedback['arm_extension']['status'] = 'warning'
            self.detailed_feedback['arm_extension']['message'] = "Arms not fully extended"
        else:
            self.feedback = "Good form"
            self.current_color = self.color_correct
            # Green for good form
            self.problem_areas['right_arm'] = self.color_correct
            self.problem_areas['left_arm'] = self.color_correct
            self.problem_areas['torso'] = self.color_correct
            self.problem_areas['right_leg'] = self.color_correct
            self.problem_areas['left_leg'] = self.color_correct
        
        # Update detailed feedback for elbow angle at bottom position
        if self.stage == "down":
            self.detailed_feedback['elbow_angle']['value'] = elbow_angle

        # Update detailed feedback for arm extension at top position
        if self.stage == "up":
            self.detailed_feedback['arm_extension']['value'] = elbow_angle

        # Check movement speed
        speed = self.analytics['movement_speed']
        self.detailed_feedback['movement_speed']['value'] = speed

        if speed > 70:  # Too fast
            self.detailed_feedback['movement_speed']['status'] = 'warning'
            self.detailed_feedback['movement_speed']['message'] = "Movement too fast"
        elif speed < 20 and speed > 0:  # Too slow
            self.detailed_feedback['movement_speed']['status'] = 'warning'
            self.detailed_feedback['movement_speed']['message'] = "Movement too slow"
        else:  # Good speed
            self.detailed_feedback['movement_speed']['status'] = 'good'
            self.detailed_feedback['movement_speed']['message'] = "Good movement speed"
        
        # Calculate form score based on deductions
        total_deduction = sum(self.deductions.values())
        self.form_score = max(0, 100 - total_deduction)
        
        # Update analytics with detailed feedback
        self.analytics['detailed_feedback'] = self.detailed_feedback
        
        return self.rep_count, self.status, self.feedback, self.current_color
    
    def reset(self):
        super().reset()
        self.stage = None
        self.body_aligned = True
        self.deductions = {
            'body_alignment': 0,
            'elbow_angle': 0,
            'arm_extension': 0
        }
        
        # Reset detailed feedback
        self.detailed_feedback = {
            'body_alignment': {
                'status': 'good',
                'message': '',
                'value': 0.0,
                'ideal_range': (160, 180)
            },
            'elbow_angle': {
                'status': 'good',
                'message': '',
                'value': 90.0,
                'ideal_range': (70, 90)
            },
            'arm_extension': {
                'status': 'good',
                'message': '',
                'value': 180.0,
                'ideal_range': (160, 180)
            },
            'movement_speed': {
                'status': 'good',
                'message': '',
                'value': 0.0,
                'ideal_range': (30, 60)
            },
            'stability': {
                'status': 'good',
                'message': '',
                'value': 100,
                'ideal_range': (85, 100)
            }
        } 