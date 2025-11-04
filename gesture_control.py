import cv2
import mediapipe as mp
import threading
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class GestureController(QThread):
    """
    Gesture control using MediaPipe for face detection.
    FIXED: Uses shared frame buffer - completely local processing
    """
    
    camera_control_signal = pyqtSignal(bool)
    status_update_signal = pyqtSignal(str)
    
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.running = False
        self.face_detection_enabled = True
        
        # MediaPipe setup
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        # Camera control variables
        self.last_face_detected = False
        self.face_absent_counter = 0
        self.face_present_counter = 0
        self.face_absent_threshold = 30
        self.face_present_threshold = 10
        
        # Visual feedback variables (LOCAL DISPLAY ONLY)
        self.show_detection_boxes = True
        self.detection_results = None
        
        # Shared frame buffer for detection (separate from transmission)
        self.last_detection_frame = None
        self.frame_lock = threading.Lock()
        
        # Connect signals
        self.camera_control_signal.connect(self.control_camera)
        self.status_update_signal.connect(self.update_status)
    
    def run(self):
        """Main thread loop - processes frames from shared buffer"""
        self.running = True
        self.status_update_signal.emit("Gesture Control: Started - Face detection active")
        
        camera = self.main_window.client.camera
        if not camera:
            self.status_update_signal.emit("Gesture Control: Error - No camera available")
            return
        
        while self.running:
            try:
                # Get frame from shared buffer
                with self.frame_lock:
                    if self.last_detection_frame is None:
                        time.sleep(0.05)
                        continue
                    
                    # Work on a copy to avoid threading issues
                    frame = self.last_detection_frame.copy()
                
                # Convert to RGB if needed (MediaPipe expects RGB)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    rgb_frame = frame  # Already RGB from camera
                else:
                    rgb_frame = frame
                
                # Process frame for face detection
                results = self.face_detection.process(rgb_frame)
                
                # Store detection results for LOCAL drawing only
                self.detection_results = results
                
                # Check if face is detected
                face_detected = results.detections is not None and len(results.detections) > 0
                
                # Update face detection counters
                self.update_face_counters(face_detected)
                
                # Control camera based on face detection (LOCAL STATE ONLY)
                self.handle_camera_control(face_detected)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.status_update_signal.emit(f"Gesture Control: Error - {str(e)}")
                time.sleep(1)
        
        self.status_update_signal.emit("Gesture Control: Stopped")
    
    def update_frame_for_detection(self, frame):
        """
        Update the frame buffer for gesture detection
        Called by Camera.get_frame() to provide frames for detection
        This is LOCAL ONLY - doesn't affect transmission
        """
        with self.frame_lock:
            self.last_detection_frame = frame.copy()
    
    def draw_detection_boxes(self, frame):
        """
        Draw detection boxes on frame for LOCAL DISPLAY ONLY
        This is NEVER transmitted to other clients
        """
        if not self.show_detection_boxes or self.detection_results is None:
            return frame
        
        # Work on a copy
        display_frame = frame.copy()
        
        # Convert to BGR for OpenCV drawing
        if len(display_frame.shape) == 3 and display_frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = display_frame.copy()
        
        height, width = frame_bgr.shape[:2]
        
        # Draw face detection boxes
        if self.detection_results.detections:
            for detection in self.detection_results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert normalized coordinates to pixel coordinates
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Draw bounding box
                color = (0, 255, 0)  # Green for face detection
                thickness = 2
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, thickness)
                
                # Draw confidence score
                confidence = detection.score[0] if detection.score else 0
                label = f"Face: {confidence:.2f}"
                
                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                
                # Draw text background
                cv2.rectangle(
                    frame_bgr, 
                    (x, y - text_height - baseline - 5), 
                    (x + text_width, y), 
                    color, 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    frame_bgr, 
                    label, 
                    (x, y - baseline - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    1
                )
                
                # Draw key facial landmarks if available
                if hasattr(detection, 'location_data') and hasattr(detection.location_data, 'relative_keypoints'):
                    for keypoint in detection.location_data.relative_keypoints:
                        kp_x = int(keypoint.x * width)
                        kp_y = int(keypoint.y * height)
                        cv2.circle(frame_bgr, (kp_x, kp_y), 3, (255, 0, 0), -1)
        
        # Add status text
        status_text = f"Faces: {len(self.detection_results.detections) if self.detection_results.detections else 0}"
        camera_status = "ON" if self.main_window.client.camera_enabled else "OFF"
        status_text += f" | Camera: {camera_status}"
        
        # Draw status background
        (text_width, text_height), baseline = cv2.getTextSize(
            status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        cv2.rectangle(
            frame_bgr, 
            (10, 10), 
            (20 + text_width, 20 + text_height + baseline), 
            (0, 0, 0), 
            -1
        )
        
        # Draw status text
        cv2.putText(
            frame_bgr, 
            status_text, 
            (15, 15 + text_height), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Convert back to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            return frame_bgr
    
    def update_face_counters(self, face_detected):
        """Update counters for face detection stability"""
        if face_detected:
            self.face_present_counter += 1
            self.face_absent_counter = 0
        else:
            self.face_absent_counter += 1
            self.face_present_counter = 0
    
    def handle_camera_control(self, face_detected):
        """
        Handle camera on/off based on face detection
        This only affects LOCAL state - transmitted automatically
        """
        current_camera_state = self.main_window.client.camera_enabled
        
        # Turn off camera if no face detected for threshold frames
        if (not face_detected and 
            self.face_absent_counter >= self.face_absent_threshold and 
            current_camera_state):
            
            self.camera_control_signal.emit(False)
            self.status_update_signal.emit("Gesture Control: No face detected - Camera turned OFF")
        
        # Turn on camera if face detected for threshold frames
        elif (face_detected and 
              self.face_present_counter >= self.face_present_threshold and 
              not current_camera_state):
            
            self.camera_control_signal.emit(True)
            self.status_update_signal.emit("Gesture Control: Face detected - Camera turned ON")
    
    def control_camera(self, enable):
        """Control LOCAL camera state - changes are transmitted automatically"""
        if enable != self.main_window.client.camera_enabled:
            self.main_window.toggle_camera()
    
    def update_status(self, message):
        """Update status in chat widget"""
        self.main_window.chat_widget.add_msg("System", "You", message)
    
    def stop_gesture_control(self):
        """Stop the gesture control thread"""
        self.running = False
        if self.isRunning():
            self.wait(3000)
    
    def toggle_detection_boxes(self, show=None):
        """Toggle visibility of detection boxes (LOCAL DISPLAY ONLY)"""
        if show is None:
            self.show_detection_boxes = not self.show_detection_boxes
        else:
            self.show_detection_boxes = show
        
        status = "enabled" if self.show_detection_boxes else "disabled"
        self.status_update_signal.emit(f"Gesture Control: Detection boxes {status}")
    
    def set_face_detection_sensitivity(self, sensitivity):
        """Adjust face detection sensitivity"""
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=sensitivity
        )
        self.status_update_signal.emit(f"Gesture Control: Sensitivity set to {sensitivity}")
    
    def set_thresholds(self, absent_threshold=30, present_threshold=10):
        """Set custom thresholds for camera control"""
        self.face_absent_threshold = absent_threshold
        self.face_present_threshold = present_threshold
        self.status_update_signal.emit(
            f"Gesture Control: Thresholds updated - "
            f"Absent: {absent_threshold}, Present: {present_threshold}"
        )


class AdvancedGestureController(GestureController):
    """Extended gesture controller with hand gestures - ALL LOCAL"""
    
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.last_gesture = None
        self.gesture_counter = 0
        self.gesture_threshold = 5
        self.hand_results = None
    
    def run(self):
        """Enhanced run method with hand detection - ALL LOCAL"""
        self.running = True
        self.status_update_signal.emit("Advanced Gesture Control: Started - Face + Hand detection active")
        
        camera = self.main_window.client.camera
        if not camera:
            self.status_update_signal.emit("Gesture Control: Error - No camera available")
            return
        
        while self.running:
            try:
                # Get frame from shared buffer
                with self.frame_lock:
                    if self.last_detection_frame is None:
                        time.sleep(0.05)
                        continue
                    
                    frame = self.last_detection_frame.copy()
                
                rgb_frame = frame
                
                # Face detection
                face_results = self.face_detection.process(rgb_frame)
                face_detected = face_results.detections is not None and len(face_results.detections) > 0
                
                self.detection_results = face_results
                
                self.update_face_counters(face_detected)
                self.handle_camera_control(face_detected)
                
                # Hand gesture detection (LOCAL ONLY)
                hand_results = self.hands.process(rgb_frame)
                self.hand_results = hand_results
                
                if hand_results.multi_hand_landmarks:
                    self.process_hand_gestures(hand_results.multi_hand_landmarks)
                
                time.sleep(0.1)
                
            except Exception as e:
                self.status_update_signal.emit(f"Advanced Gesture Control: Error - {str(e)}")
                time.sleep(1)
        
        self.status_update_signal.emit("Advanced Gesture Control: Stopped")
    
    def draw_detection_boxes(self, frame):
        """Enhanced drawing with hand landmarks - LOCAL DISPLAY ONLY"""
        # First draw face detection boxes
        frame = super().draw_detection_boxes(frame)
        
        if not self.show_detection_boxes or self.hand_results is None:
            return frame
        
        # Work on a copy
        display_frame = frame.copy()
        
        # Convert to BGR for OpenCV
        if len(display_frame.shape) == 3 and display_frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = display_frame.copy()
        
        height, width = frame_bgr.shape[:2]
        
        # Draw hand landmarks
        if self.hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(self.hand_results.multi_hand_landmarks):
                # Draw hand skeleton
                self.mp_drawing.draw_landmarks(
                    frame_bgr, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
                
                # Get hand classification
                hand_label = "Hand"
                if self.hand_results.multi_handedness:
                    if hand_idx < len(self.hand_results.multi_handedness):
                        hand_label = self.hand_results.multi_handedness[hand_idx].classification[0].label
                
                # Detect current gesture
                current_gesture = self.detect_gesture(hand_landmarks)
                if current_gesture:
                    hand_label += f" - {current_gesture}"
                
                # Get hand bounding box
                landmarks = hand_landmarks.landmark
                x_coords = [lm.x * width for lm in landmarks]
                y_coords = [lm.y * height for lm in landmarks]
                
                x_min, x_max = int(min(x_coords) - 20), int(max(x_coords) + 20)
                y_min, y_max = int(min(y_coords) - 20), int(max(y_coords) + 20)
                
                # Draw bounding box
                cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
                
                # Draw hand label
                (text_width, text_height), baseline = cv2.getTextSize(
                    hand_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                
                cv2.rectangle(
                    frame_bgr, 
                    (x_min, y_min - text_height - baseline - 5), 
                    (x_min + text_width, y_min), 
                    (255, 0, 255), 
                    -1
                )
                
                cv2.putText(
                    frame_bgr, 
                    hand_label, 
                    (x_min, y_min - baseline - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    1
                )
        
        # Convert back to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            return frame_bgr
    
    def process_hand_gestures(self, hand_landmarks_list):
        """Process hand gestures for additional LOCAL controls"""
        for hand_landmarks in hand_landmarks_list:
            gesture = self.detect_gesture(hand_landmarks)
            
            if gesture == self.last_gesture:
                self.gesture_counter += 1
            else:
                self.gesture_counter = 0
                self.last_gesture = gesture
            
            # Execute gesture command if stable (affects LOCAL state only)
            if self.gesture_counter >= self.gesture_threshold:
                self.execute_gesture_command(gesture)
                self.gesture_counter = 0
    
    def detect_gesture(self, hand_landmarks):
        """Simple gesture detection based on landmark positions"""
        landmarks = hand_landmarks.landmark
        
        # Get key landmark positions
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        
        # Thumbs up gesture (microphone control)
        if (thumb_tip.y < thumb_ip.y and 
            index_tip.y > index_pip.y and 
            middle_tip.y > middle_pip.y):
            return "thumbs_up"
        
        # Peace sign
        if (index_tip.y < index_pip.y and 
            middle_tip.y < middle_pip.y and 
            thumb_tip.y > thumb_ip.y):
            return "peace_sign"
        
        # Open palm
        fingers_extended = sum([
            thumb_tip.y < thumb_ip.y,
            index_tip.y < index_pip.y,
            middle_tip.y < middle_pip.y,
            landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < 
            landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP].y,
            landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y < 
            landmarks[self.mp_hands.HandLandmark.PINKY_PIP].y
        ])
        
        if fingers_extended >= 4:
            return "open_palm"
        
        return None
    
    def execute_gesture_command(self, gesture):
        """Execute command based on detected gesture (affects LOCAL state only)"""
        if gesture == "thumbs_up":
            # Toggle LOCAL microphone
            current_mic_state = self.main_window.client.microphone_enabled
            self.main_window.toggle_microphone()
            status = "ON" if not current_mic_state else "OFF"
            self.status_update_signal.emit(f"Gesture Control: Thumbs up detected - Microphone turned {status}")
        
        elif gesture == "peace_sign":
            self.status_update_signal.emit("Gesture Control: Peace sign detected - Feature not implemented")
        
        elif gesture == "open_palm":
            self.status_update_signal.emit("Gesture Control: Open palm detected - Wave hello!")


def integrate_gesture_control(main_window):
    """Helper function to integrate gesture control with the main window"""
    try:
        import mediapipe
    except ImportError:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(
            main_window, 
            "Missing Dependency", 
            "MediaPipe library is required for gesture control.\n"
            "Please install it using: pip install mediapipe"
        )
        return None
        
    # Create gesture controller (use Advanced for hand gestures)
    gesture_controller = AdvancedGestureController(main_window)
        
    return gesture_controller