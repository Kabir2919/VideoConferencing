# import cv2
# import mediapipe as mp
# import threading
# import time
# import numpy as np
# from PyQt6.QtCore import QThread, pyqtSignal


# class GestureController(QThread):
#     """
#     Gesture control using MediaPipe for face detection.
#     FIXED: Proper signal handling and camera state control
#     """

#     status_update_signal = pyqtSignal(str)
#     initialization_complete_signal = pyqtSignal(bool)
#     # FIXED: Use bool signal to directly set camera state
#     set_camera_state_signal = pyqtSignal(bool)

#     def __init__(self, main_window, control_transmission=False, parent=None):
#         super().__init__(parent)
#         self.main_window = main_window
#         self.running = False
#         self.initialized = False
#         self.face_detection_enabled = True
        
#         # Control whether to affect camera transmission or just local preview
#         self.control_transmission = control_transmission

#         # MediaPipe setup - will be initialized in background
#         self.mp_face_detection = None
#         self.mp_drawing = None
#         self.face_detection = None

#         # Face-state counters
#         self.last_face_detected = False
#         self.face_absent_counter = 0
#         self.face_present_counter = 0
#         self.face_absent_threshold = 30
#         self.face_present_threshold = 10

#         # Visual feedback (LOCAL preview only)
#         self.show_detection_boxes = True
#         self.detection_results = None

#         # Shared frame buffer for detection (separate from transmission)
#         self.last_detection_frame = None
#         self.frame_lock = threading.Lock()

#         # Local preview hide flag
#         self.local_hide_camera = False

#         # Connect signals
#         self.status_update_signal.connect(self.update_status)
#         self.initialization_complete_signal.connect(self.on_initialization_complete)
        
#         # FIXED: Connect camera control signal properly
#         if self.control_transmission:
#             self.set_camera_state_signal.connect(self._set_camera_state_slot)

#     def _set_camera_state_slot(self, state):
#         """Slot to handle camera state changes - runs in main thread"""
#         try:
#             self.main_window.client.camera_enabled = state
#             self.main_window.update_camera_ui_state()
#         except Exception as e:
#             print(f"[GESTURE] Error setting camera state: {e}")

#     def run(self):
#         """Main thread loop with async initialization"""
#         self.running = True
#         self.status_update_signal.emit("Gesture Control: Initializing...")

#         camera = self.main_window.client.camera
#         if not camera:
#             self.status_update_signal.emit("Gesture Control: Error - No camera available")
#             return

#         # Initialize MediaPipe in background (non-blocking)
#         try:
#             self.status_update_signal.emit("Gesture Control: Loading MediaPipe models...")
#             self.mp_face_detection = mp.solutions.face_detection
#             self.mp_drawing = mp.solutions.drawing_utils
            
#             # Lower thread priority if possible
#             import os
#             if hasattr(os, 'nice'):
#                 try:
#                     os.nice(5)
#                 except:
#                     pass
            
#             self.face_detection = self.mp_face_detection.FaceDetection(
#                 model_selection=0,
#                 min_detection_confidence=0.5
#             )
            
#             self.initialized = True
#             self.initialization_complete_signal.emit(True)
#             mode = "with camera control" if self.control_transmission else "preview only"
#             self.status_update_signal.emit(f"Gesture Control: Started ({mode})")
            
#         except Exception as e:
#             self.initialized = False
#             self.initialization_complete_signal.emit(False)
#             self.status_update_signal.emit(f"Gesture Control: Initialization failed - {str(e)}")
#             return

#         # Main processing loop - only runs after successful initialization
#         while self.running and self.initialized:
#             try:
#                 # Get frame from shared buffer
#                 with self.frame_lock:
#                     if self.last_detection_frame is None:
#                         time.sleep(0.05)
#                         continue
#                     frame = self.last_detection_frame.copy()

#                 # MediaPipe expects RGB
#                 rgb_frame = frame

#                 # Face detection
#                 results = self.face_detection.process(rgb_frame)
#                 self.detection_results = results

#                 # Face present?
#                 face_detected = results.detections is not None and len(results.detections) > 0

#                 # Update counters & apply behavior
#                 self.update_face_counters(face_detected)
                
#                 # Choose behavior based on control mode
#                 if self.control_transmission:
#                     self.handle_camera_transmission(face_detected)
#                 else:
#                     self.handle_local_hide(face_detected)

#                 time.sleep(0.1)  # limit CPU usage

#             except Exception as e:
#                 self.status_update_signal.emit(f"Gesture Control: Error - {str(e)}")
#                 time.sleep(1)

#         self.status_update_signal.emit("Gesture Control: Stopped")

#     def on_initialization_complete(self, success):
#         """Called when initialization completes"""
#         if not success:
#             self.running = False

#     def update_frame_for_detection(self, frame):
#         """
#         Update the frame buffer for gesture detection.
#         Called by Camera.get_frame() to provide frames for detection.
#         """
#         try:
#             with self.frame_lock:
#                 self.last_detection_frame = frame.copy()
#         except Exception as e:
#             print(f"[GestureControl] Warning: failed to update frame for detection: {e}")

#     def draw_detection_boxes(self, frame):
#         """
#         Draw detection boxes on frame for LOCAL DISPLAY ONLY.
#         This is NEVER transmitted to other clients.
#         """
#         # Don't draw if not initialized yet
#         if not self.initialized:
#             return frame
            
#         # Local preview hidden → draw placeholder
#         if self.local_hide_camera:
#             blank = np.zeros_like(frame)
#             text = "Face not detected - local preview hidden"
#             try:
#                 (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#                 x = max(10, (blank.shape[1] - tw) // 2)
#                 y = max(40, (blank.shape[0] // 2))
#                 cv2.putText(blank, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#             except:
#                 pass
#             return blank

#         if not self.show_detection_boxes or self.detection_results is None:
#             return frame

#         display_frame = frame.copy()

#         # Convert to BGR for OpenCV drawing
#         try:
#             if len(display_frame.shape) == 3 and display_frame.shape[2] == 3:
#                 frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
#             else:
#                 frame_bgr = display_frame.copy()
#         except Exception as e:
#             print(f"[GESTURE] Color conversion error: {e}")
#             return frame

#         height, width = frame_bgr.shape[:2]

#         # Draw face detection boxes
#         if self.detection_results.detections:
#             for detection in self.detection_results.detections:
#                 try:
#                     # Bounding box (normalized → pixels)
#                     bbox = detection.location_data.relative_bounding_box
#                     x = int(bbox.xmin * width)
#                     y = int(bbox.ymin * height)
#                     w = int(bbox.width * width)
#                     h = int(bbox.height * height)

#                     # Clamp to frame
#                     x = max(0, x); y = max(0, y)
#                     w = max(1, min(w, width - x))
#                     h = max(1, min(h, height - y))

#                     # Box + label
#                     cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     confidence = detection.score[0] if detection.score else 0
#                     label = f"Face: {confidence:.2f}"

#                     (text_width, text_height), baseline = cv2.getTextSize(
#                         label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
#                     )
#                     cv2.rectangle(
#                         frame_bgr,
#                         (x, y - text_height - baseline - 5),
#                         (x + text_width, y),
#                         (0, 255, 0),
#                         -1
#                     )
#                     cv2.putText(
#                         frame_bgr,
#                         label,
#                         (x, y - baseline - 2),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         (255, 255, 255),
#                         1
#                     )

#                     # Optional facial keypoints
#                     if (hasattr(detection, 'location_data')
#                             and hasattr(detection.location_data, 'relative_keypoints')
#                             and detection.location_data.relative_keypoints):
#                         for kp in detection.location_data.relative_keypoints:
#                             kp_x = int(kp.x * width)
#                             kp_y = int(kp.y * height)
#                             cv2.circle(frame_bgr, (kp_x, kp_y), 3, (255, 0, 0), -1)
#                 except Exception as e:
#                     print(f"[GESTURE] Drawing error: {e}")
#                     continue

#         # Status text
#         try:
#             status_text = f"Faces: {len(self.detection_results.detections) if self.detection_results.detections else 0}"
#             camera_status = "ON" if self.main_window.client.camera_enabled else "OFF"
#             status_text += f" | Camera: {camera_status}"

#             (text_width, text_height), baseline = cv2.getTextSize(
#                 status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
#             )
#             cv2.rectangle(
#                 frame_bgr,
#                 (10, 10),
#                 (20 + text_width, 20 + text_height + baseline),
#                 (0, 0, 0),
#                 -1
#             )
#             cv2.putText(
#                 frame_bgr,
#                 status_text,
#                 (15, 15 + text_height),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 255, 255),
#                 2
#             )
#         except Exception as e:
#             print(f"[GESTURE] Status text error: {e}")

#         # Back to RGB for Qt
#         try:
#             if len(frame.shape) == 3 and frame.shape[2] == 3:
#                 return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#             else:
#                 return frame_bgr
#         except Exception as e:
#             print(f"[GESTURE] Final conversion error: {e}")
#             return frame

#     def update_face_counters(self, face_detected):
#         """Update counters for face detection stability"""
#         if face_detected:
#             self.face_present_counter += 1
#             self.face_absent_counter = 0
#         else:
#             self.face_absent_counter += 1
#             self.face_present_counter = 0

#     def handle_camera_transmission(self, face_detected):
#         """
#         Control actual camera transmission based on face presence.
#         This will turn the camera on/off for ALL clients.
#         """
#         current_camera_state = self.main_window.client.camera_enabled
        
#         # Turn off camera after sustained absence
#         if (not face_detected
#                 and self.face_absent_counter >= self.face_absent_threshold
#                 and current_camera_state):
#             # Emit signal to turn camera OFF
#             self.set_camera_state_signal.emit(False)
#             self.status_update_signal.emit("Gesture: Face lost → Camera transmission OFF")

#         # Turn on camera after sustained presence
#         if (face_detected
#                 and self.face_present_counter >= self.face_present_threshold
#                 and not current_camera_state):
#             # Emit signal to turn camera ON
#             self.set_camera_state_signal.emit(True)
#             self.status_update_signal.emit("Gesture: Face detected → Camera transmission ON")

#     def handle_local_hide(self, face_detected):
#         """
#         Only hide/show LOCAL preview based on face presence.
#         Never toggles camera transmission.
#         """
#         # Hide local preview after sustained absence
#         if (not face_detected
#                 and self.face_absent_counter >= self.face_absent_threshold
#                 and not self.local_hide_camera):
#             self.local_hide_camera = True
#             self.status_update_signal.emit("Gesture: Face lost → local preview hidden")

#         # Restore local preview after sustained presence
#         if (face_detected
#                 and self.face_present_counter >= self.face_present_threshold
#                 and self.local_hide_camera):
#             self.local_hide_camera = False
#             self.status_update_signal.emit("Gesture: Face detected → local preview restored")

#     def update_status(self, message):
#         """Update status in chat widget"""
#         try:
#             self.main_window.chat_widget.add_msg("System", "You", message)
#         except Exception as e:
#             print(f"[GESTURE] Status update error: {e}")

#     def stop_gesture_control(self):
#         """Stop the gesture control thread"""
#         self.running = False
#         # Give it time to exit cleanly
#         if self.isRunning():
#             self.wait(3000)
        
#         # Clean up signal connections
#         try:
#             if self.control_transmission:
#                 self.set_camera_state_signal.disconnect()
#         except:
#             pass

#     def toggle_detection_boxes(self, show=None):
#         """Toggle visibility of detection boxes (LOCAL DISPLAY ONLY)"""
#         if show is None:
#             self.show_detection_boxes = not self.show_detection_boxes
#         else:
#             self.show_detection_boxes = show    

#         status = "enabled" if self.show_detection_boxes else "disabled"
#         self.status_update_signal.emit(f"Gesture: Detection boxes {status}")

#     def set_face_detection_sensitivity(self, sensitivity):
#         """Adjust face detection sensitivity"""
#         if not self.initialized:
#             return
#         try:
#             self.face_detection = self.mp_face_detection.FaceDetection(
#                 model_selection=0,
#                 min_detection_confidence=sensitivity
#             )
#             self.status_update_signal.emit(f"Gesture: Sensitivity set to {sensitivity}")
#         except Exception as e:
#             self.status_update_signal.emit(f"Gesture: Failed to set sensitivity - {str(e)}")

#     def set_thresholds(self, absent_threshold=30, present_threshold=10):
#         """Set custom thresholds for behavior"""
#         self.face_absent_threshold = absent_threshold
#         self.face_present_threshold = present_threshold
#         self.status_update_signal.emit(
#             f"Gesture: Thresholds updated - Absent: {absent_threshold}, Present: {present_threshold}"
#         )


# class AdvancedGestureController(GestureController):
#     """Extended gesture controller with hand gestures"""

#     def __init__(self, main_window, control_transmission=False, parent=None):
#         super().__init__(main_window, control_transmission, parent)

#         self.mp_hands = None
#         self.hands = None

#         self.last_gesture = None
#         self.gesture_counter = 0
#         self.gesture_threshold = 5
#         self.hand_results = None

#     def run(self):
#         """Enhanced run method with hand detection"""
#         self.running = True
#         self.status_update_signal.emit("Advanced Gesture Control: Initializing...")

#         camera = self.main_window.client.camera
#         if not camera:
#             self.status_update_signal.emit("Gesture Control: Error - No camera available")
#             return

#         # Initialize MediaPipe in background
#         try:
#             self.status_update_signal.emit("Advanced Gesture Control: Loading models...")
            
#             # Lower thread priority
#             import os
#             if hasattr(os, 'nice'):
#                 try:
#                     os.nice(5)
#                 except:
#                     pass
            
#             # Face detection
#             self.mp_face_detection = mp.solutions.face_detection
#             self.mp_drawing = mp.solutions.drawing_utils
#             self.face_detection = self.mp_face_detection.FaceDetection(
#                 model_selection=0,
#                 min_detection_confidence=0.5
#             )
            
#             # Hand detection
#             self.mp_hands = mp.solutions.hands
#             self.hands = self.mp_hands.Hands(
#                 static_image_mode=False,
#                 max_num_hands=2,
#                 min_detection_confidence=0.5,
#                 min_tracking_confidence=0.5
#             )
            
#             self.initialized = True
#             self.initialization_complete_signal.emit(True)
#             mode = "with camera control" if self.control_transmission else "preview only"
#             self.status_update_signal.emit(f"Advanced Gesture Control: Started ({mode})")
            
#         except Exception as e:
#             self.initialized = False
#             self.initialization_complete_signal.emit(False)
#             self.status_update_signal.emit(f"Advanced Gesture Control: Initialization failed - {str(e)}")
#             return

#         while self.running and self.initialized:
#             try:
#                 with self.frame_lock:
#                     if self.last_detection_frame is None:
#                         time.sleep(0.05)
#                         continue
#                     frame = self.last_detection_frame.copy()

#                 rgb_frame = frame

#                 # Face detection
#                 face_results = self.face_detection.process(rgb_frame)
#                 face_detected = face_results.detections is not None and len(face_results.detections) > 0

#                 self.detection_results = face_results

#                 self.update_face_counters(face_detected)
                
#                 # Apply behavior based on mode
#                 if self.control_transmission:
#                     self.handle_camera_transmission(face_detected)
#                 else:
#                     self.handle_local_hide(face_detected)

#                 # Hand gesture detection
#                 hand_results = self.hands.process(rgb_frame)
#                 self.hand_results = hand_results

#                 if hand_results.multi_hand_landmarks:
#                     self.process_hand_gestures(hand_results.multi_hand_landmarks)

#                 time.sleep(0.1)

#             except Exception as e:
#                 self.status_update_signal.emit(f"Advanced Gesture Control: Error - {str(e)}")
#                 time.sleep(1)

#         self.status_update_signal.emit("Advanced Gesture Control: Stopped")

#     def draw_detection_boxes(self, frame):
#         """Enhanced drawing with hand landmarks"""
#         if not self.initialized:
#             return frame
            
#         # Respect local preview hide first
#         frame = super().draw_detection_boxes(frame)
#         if self.local_hide_camera:
#             return frame

#         if not self.show_detection_boxes or self.hand_results is None:
#             return frame

#         display_frame = frame.copy()

#         try:
#             if len(display_frame.shape) == 3 and display_frame.shape[2] == 3:
#                 frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
#             else:
#                 frame_bgr = display_frame.copy()

#             height, width = frame_bgr.shape[:2]

#             # Draw hand landmarks
#             if self.hand_results.multi_hand_landmarks:
#                 for hand_idx, hand_landmarks in enumerate(self.hand_results.multi_hand_landmarks):
#                     self.mp_drawing.draw_landmarks(
#                         frame_bgr,
#                         hand_landmarks,
#                         self.mp_hands.HAND_CONNECTIONS,
#                         self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
#                         self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
#                     )

#                     hand_label = "Hand"
#                     if self.hand_results.multi_handedness and hand_idx < len(self.hand_results.multi_handedness):
#                         hand_label = self.hand_results.multi_handedness[hand_idx].classification[0].label

#                     current_gesture = self.detect_gesture(hand_landmarks)
#                     if current_gesture:
#                         hand_label += f" - {current_gesture}"

#                     landmarks = hand_landmarks.landmark
#                     x_coords = [lm.x * width for lm in landmarks]
#                     y_coords = [lm.y * height for lm in landmarks]
#                     x_min, x_max = int(min(x_coords) - 20), int(max(x_coords) + 20)
#                     y_min, y_max = int(min(y_coords) - 20), int(max(y_coords) + 20)
#                     x_min = max(0, x_min); y_min = max(0, y_min)
#                     x_max = min(width - 1, x_max); y_max = min(height - 1, y_max)

#                     cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

#                     (text_width, text_height), baseline = cv2.getTextSize(
#                         hand_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
#                     )
#                     cv2.rectangle(
#                         frame_bgr,
#                         (x_min, y_min - text_height - baseline - 5),
#                         (x_min + text_width, y_min),
#                         (255, 0, 255),
#                         -1
#                     )
#                     cv2.putText(
#                         frame_bgr,
#                         hand_label,
#                         (x_min, y_min - baseline - 2),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         (255, 255, 255),
#                         1
#                     )

#             if len(frame.shape) == 3 and frame.shape[2] == 3:
#                 return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#             else:
#                 return frame_bgr
#         except Exception as e:
#             print(f"[GESTURE] Hand drawing error: {e}")
#             return frame

#     def process_hand_gestures(self, hand_landmarks_list):
#         """Process hand gestures"""
#         for hand_landmarks in hand_landmarks_list:
#             gesture = self.detect_gesture(hand_landmarks)

#             if gesture == self.last_gesture:
#                 self.gesture_counter += 1
#             else:
#                 self.gesture_counter = 0
#                 self.last_gesture = gesture

#             if self.gesture_counter >= self.gesture_threshold:
#                 self.execute_gesture_command(gesture)
#                 self.gesture_counter = 0

#     def detect_gesture(self, hand_landmarks):
#         """Enhanced gesture detection with more gestures"""
#         try:
#             landmarks = hand_landmarks.landmark

#             thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
#             thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
#             index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
#             index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
#             middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
#             middle_pip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
#             ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
#             ring_pip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP]
#             pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
#             pinky_pip = landmarks[self.mp_hands.HandLandmark.PINKY_PIP]

#             # Count extended fingers
#             fingers_extended = [
#                 thumb_tip.x < thumb_ip.x if thumb_tip.x < landmarks[self.mp_hands.HandLandmark.WRIST].x else thumb_tip.x > thumb_ip.x,
#                 index_tip.y < index_pip.y,
#                 middle_tip.y < middle_pip.y,
#                 ring_tip.y < ring_pip.y,
#                 pinky_tip.y < pinky_pip.y
#             ]
            
#             num_extended = sum(fingers_extended)

#             # Open palm - all 5 fingers extended
#             if num_extended >= 4:
#                 return "open_palm"
            
#             # Pointing up - only index finger extended
#             if (fingers_extended[1] and 
#                 not fingers_extended[2] and 
#                 not fingers_extended[3] and 
#                 not fingers_extended[4]):
#                 return "pointing_up"

#             # Peace sign - index and middle extended
#             if (fingers_extended[1] and 
#                 fingers_extended[2] and 
#                 not fingers_extended[3] and 
#                 not fingers_extended[4]):
#                 return "peace_sign"

#             # Thumbs up
#             if (fingers_extended[0] and
#                 not fingers_extended[1] and
#                 not fingers_extended[2] and
#                 index_tip.y > index_pip.y and
#                 middle_tip.y > middle_pip.y):
#                 return "thumbs_up"
            
#             # Closed fist - no fingers extended
#             if num_extended == 0:
#                 return "fist"

#             return None
            
#         except Exception as e:
#             print(f"[GESTURE] Gesture detection error: {e}")
#             return None

#     def execute_gesture_command(self, gesture):
#         """Execute command based on detected gesture"""
#         try:
#             if gesture == "thumbs_up":
#                 current_mic_state = self.main_window.client.microphone_enabled
#                 self.main_window.toggle_microphone()
#                 status = "ON" if not current_mic_state else "OFF"
#                 self.status_update_signal.emit(f"Gesture: Thumbs up → Microphone {status}")

#             elif gesture == "peace_sign":
#                 # Toggle camera using gesture
#                 current_camera_state = self.main_window.client.camera_enabled
#                 self.main_window.toggle_camera()
#                 status = "ON" if not current_camera_state else "OFF"
#                 self.status_update_signal.emit(f"Gesture: Peace sign → Camera {status}")

#             elif gesture == "open_palm":
#                 # Send "Hello 👋" message to all connected clients
#                 try:
#                     from constants import Message, POST, TEXT
                    
#                     # Get all connected clients (excluding self)
#                     recipients = []
#                     from client import all_clients
#                     for name in all_clients.keys():
#                         if name != self.main_window.client.name:
#                             recipients.append(name)
                    
#                     if len(recipients) > 0:
#                         # Send hello message
#                         hello_msg = "Hello 👋"
#                         msg = Message(
#                             self.main_window.client.name, 
#                             POST, 
#                             TEXT, 
#                             data=hello_msg, 
#                             to_names=tuple(recipients)
#                         )
#                         self.main_window.server_conn.send_msg(
#                             self.main_window.server_conn.main_socket, 
#                             msg
#                         )
                        
#                         # Show in local chat
#                         self.main_window.chat_widget.add_msg(
#                             "You", 
#                             ", ".join(recipients), 
#                             hello_msg
#                         )
#                         self.status_update_signal.emit("Gesture: Open palm → Sent Hello 👋")
#                     else:
#                         self.status_update_signal.emit("Gesture: Open palm → No recipients to send Hello")
#                 except Exception as e:
#                     print(f"[GESTURE] Error sending hello: {e}")
#                     self.status_update_signal.emit(f"Gesture: Error sending hello - {str(e)}")
                    
#             elif gesture == "fist":
#                 # Mute both camera and microphone (privacy mode)
#                 if self.main_window.client.camera_enabled or self.main_window.client.microphone_enabled:
#                     if self.main_window.client.camera_enabled:
#                         self.main_window.toggle_camera()
#                     if self.main_window.client.microphone_enabled:
#                         self.main_window.toggle_microphone()
#                     self.status_update_signal.emit("Gesture: Fist → Privacy Mode (Muted all)")
#                 else:
#                     # Unmute both
#                     if not self.main_window.client.camera_enabled:
#                         self.main_window.toggle_camera()
#                     if not self.main_window.client.microphone_enabled:
#                         self.main_window.toggle_microphone()
#                     self.status_update_signal.emit("Gesture: Fist → Unmuted all")
                    
#             elif gesture == "pointing_up":
#                 # Raise hand / Request attention
#                 try:
#                     from constants import Message, POST, TEXT
#                     from client import all_clients
                    
#                     recipients = [name for name in all_clients.keys() if name != self.main_window.client.name]
                    
#                     if len(recipients) > 0:
#                         attention_msg = "✋ [Raised Hand - Requesting Attention]"
#                         msg = Message(
#                             self.main_window.client.name,
#                             POST,
#                             TEXT,
#                             data=attention_msg,
#                             to_names=tuple(recipients)
#                         )
#                         self.main_window.server_conn.send_msg(
#                             self.main_window.server_conn.main_socket,
#                             msg
#                         )
#                         self.main_window.chat_widget.add_msg("You", ", ".join(recipients), attention_msg)
#                         self.status_update_signal.emit("Gesture: Pointing up → Raised hand")
#                 except Exception as e:
#                     print(f"[GESTURE] Error raising hand: {e}")
                    
#         except Exception as e:
#             print(f"[GESTURE] Command execution error: {e}")

# def integrate_gesture_control(main_window, control_transmission=False):
#     """
#     Helper to integrate gesture control with the main window
    
#     Args:
#         main_window: MainWindow instance
#         control_transmission: If True, control actual camera transmission.
#                             If False, only affect local preview (default)
#     """
#     try:
#         import mediapipe  # noqa: F401
#     except ImportError:
#         from PyQt6.QtWidgets import QMessageBox
#         QMessageBox.warning(
#             main_window,
#             "Missing Dependency",
#             "MediaPipe library is required for gesture control.\n"
#             "Install with: pip install mediapipe"
#         )
#         return None

#     # Use Advanced controller (face + hands)
#     gesture_controller = AdvancedGestureController(main_window, control_transmission)
#     return gesture_controller
import cv2
import mediapipe as mp
import threading
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class GestureController(QThread):
    """
    Gesture control using MediaPipe for face detection.
    FIXED: Proper signal handling and camera state control
    """

    status_update_signal = pyqtSignal(str)
    initialization_complete_signal = pyqtSignal(bool)
    set_camera_state_signal = pyqtSignal(bool)

    def __init__(self, main_window, control_transmission=False, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.running = False
        self.initialized = False
        self.face_detection_enabled = True
        
        self.control_transmission = control_transmission

        self.mp_face_detection = None
        self.mp_drawing = None
        self.face_detection = None

        self.last_face_detected = False
        self.face_absent_counter = 0
        self.face_present_counter = 0
        self.face_absent_threshold = 30
        self.face_present_threshold = 10

        self.show_detection_boxes = True
        self.detection_results = None

        self.last_detection_frame = None
        self.frame_lock = threading.Lock()

        self.local_hide_camera = False

        self.status_update_signal.connect(self.update_status)
        self.initialization_complete_signal.connect(self.on_initialization_complete)
        
        if self.control_transmission:
            self.set_camera_state_signal.connect(self._set_camera_state_slot)

    def _set_camera_state_slot(self, state):
        """Slot to handle camera state changes - runs in main thread"""
        try:
            self.main_window.client.camera_enabled = state
            self.main_window.update_camera_ui_state()
        except Exception as e:
            print(f"[GESTURE] Error setting camera state: {e}")

    def run(self):
        """Main thread loop with async initialization"""
        self.running = True
        self.status_update_signal.emit("Gesture Control: Initializing...")

        camera = self.main_window.client.camera
        if not camera:
            self.status_update_signal.emit("Gesture Control: Error - No camera available")
            return

        try:
            self.status_update_signal.emit("Gesture Control: Loading MediaPipe models...")
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            
            import os
            if hasattr(os, 'nice'):
                try:
                    os.nice(5)
                except:
                    pass
            
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            
            self.initialized = True
            self.initialization_complete_signal.emit(True)
            mode = "with camera control" if self.control_transmission else "preview only"
            self.status_update_signal.emit(f"Gesture Control: Started ({mode})")
            
        except Exception as e:
            self.initialized = False
            self.initialization_complete_signal.emit(False)
            self.status_update_signal.emit(f"Gesture Control: Initialization failed - {str(e)}")
            return

        while self.running and self.initialized:
            try:
                with self.frame_lock:
                    if self.last_detection_frame is None:
                        time.sleep(0.05)
                        continue
                    frame = self.last_detection_frame.copy()

                rgb_frame = frame

                results = self.face_detection.process(rgb_frame)
                self.detection_results = results

                face_detected = results.detections is not None and len(results.detections) > 0

                self.update_face_counters(face_detected)
                
                if self.control_transmission:
                    self.handle_camera_transmission(face_detected)
                else:
                    self.handle_local_hide(face_detected)

                time.sleep(0.1)

            except Exception as e:
                self.status_update_signal.emit(f"Gesture Control: Error - {str(e)}")
                time.sleep(1)

        self.status_update_signal.emit("Gesture Control: Stopped")

    def on_initialization_complete(self, success):
        """Called when initialization completes"""
        if not success:
            self.running = False

    def update_frame_for_detection(self, frame):
        """
        Update the frame buffer for gesture detection.
        Called by Camera.get_frame() to provide frames for detection.
        """
        try:
            with self.frame_lock:
                self.last_detection_frame = frame.copy()
        except Exception as e:
            print(f"[GestureControl] Warning: failed to update frame for detection: {e}")

    def draw_detection_boxes(self, frame):
        """
        Draw detection boxes on frame for LOCAL DISPLAY ONLY.
        This is NEVER transmitted to other clients.
        """
        if not self.initialized:
            return frame
            
        if self.local_hide_camera:
            blank = np.zeros_like(frame)
            text = "Face not detected - local preview hidden"
            try:
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                x = max(10, (blank.shape[1] - tw) // 2)
                y = max(40, (blank.shape[0] // 2))
                cv2.putText(blank, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except:
                pass
            return blank

        if not self.show_detection_boxes or self.detection_results is None:
            return frame

        display_frame = frame.copy()

        try:
            if len(display_frame.shape) == 3 and display_frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = display_frame.copy()
        except Exception as e:
            print(f"[GESTURE] Color conversion error: {e}")
            return frame

        height, width = frame_bgr.shape[:2]

        if self.detection_results.detections:
            for detection in self.detection_results.detections:
                try:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    x = max(0, x); y = max(0, y)
                    w = max(1, min(w, width - x))
                    h = max(1, min(h, height - y))

                    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    confidence = detection.score[0] if detection.score else 0
                    label = f"Face: {confidence:.2f}"

                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                    )
                    cv2.rectangle(
                        frame_bgr,
                        (x, y - text_height - baseline - 5),
                        (x + text_width, y),
                        (0, 255, 0),
                        -1
                    )
                    cv2.putText(
                        frame_bgr,
                        label,
                        (x, y - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1
                    )

                    if (hasattr(detection, 'location_data')
                            and hasattr(detection.location_data, 'relative_keypoints')
                            and detection.location_data.relative_keypoints):
                        for kp in detection.location_data.relative_keypoints:
                            kp_x = int(kp.x * width)
                            kp_y = int(kp.y * height)
                            cv2.circle(frame_bgr, (kp_x, kp_y), 3, (255, 0, 0), -1)
                except Exception as e:
                    print(f"[GESTURE] Drawing error: {e}")
                    continue

        try:
            status_text = f"Faces: {len(self.detection_results.detections) if self.detection_results.detections else 0}"
            camera_status = "ON" if self.main_window.client.camera_enabled else "OFF"
            status_text += f" | Camera: {camera_status}"

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
            cv2.putText(
                frame_bgr,
                status_text,
                (15, 15 + text_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        except Exception as e:
            print(f"[GESTURE] Status text error: {e}")

        try:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                return frame_bgr
        except Exception as e:
            print(f"[GESTURE] Final conversion error: {e}")
            return frame

    def update_face_counters(self, face_detected):
        """Update counters for face detection stability"""
        if face_detected:
            self.face_present_counter += 1
            self.face_absent_counter = 0
        else:
            self.face_absent_counter += 1
            self.face_present_counter = 0

    def handle_camera_transmission(self, face_detected):
        """
        Control actual camera transmission based on face presence.
        This will turn the camera on/off for ALL clients.
        """
        current_camera_state = self.main_window.client.camera_enabled
        
        if (not face_detected
                and self.face_absent_counter >= self.face_absent_threshold
                and current_camera_state):
            self.set_camera_state_signal.emit(False)
            self.status_update_signal.emit("Gesture: Face lost → Camera transmission OFF")

        if (face_detected
                and self.face_present_counter >= self.face_present_threshold
                and not current_camera_state):
            self.set_camera_state_signal.emit(True)
            self.status_update_signal.emit("Gesture: Face detected → Camera transmission ON")

    def handle_local_hide(self, face_detected):
        """
        Only hide/show LOCAL preview based on face presence.
        Never toggles camera transmission.
        """
        if (not face_detected
                and self.face_absent_counter >= self.face_absent_threshold
                and not self.local_hide_camera):
            self.local_hide_camera = True
            self.status_update_signal.emit("Gesture: Face lost → local preview hidden")

        if (face_detected
                and self.face_present_counter >= self.face_present_threshold
                and self.local_hide_camera):
            self.local_hide_camera = False
            self.status_update_signal.emit("Gesture: Face detected → local preview restored")

    def update_status(self, message):
        """Update status in chat widget"""
        try:
            self.main_window.chat_widget.add_msg("System", "You", message)
        except Exception as e:
            print(f"[GESTURE] Status update error: {e}")

    def stop_gesture_control(self):
        """Stop the gesture control thread"""
        self.running = False
        if self.isRunning():
            self.wait(3000)
        
        try:
            if self.control_transmission:
                self.set_camera_state_signal.disconnect()
        except:
            pass

    def toggle_detection_boxes(self, show=None):
        """Toggle visibility of detection boxes (LOCAL DISPLAY ONLY)"""
        if show is None:
            self.show_detection_boxes = not self.show_detection_boxes
        else:
            self.show_detection_boxes = show    

        status = "enabled" if self.show_detection_boxes else "disabled"
        self.status_update_signal.emit(f"Gesture: Detection boxes {status}")

    def set_face_detection_sensitivity(self, sensitivity):
        """Adjust face detection sensitivity"""
        if not self.initialized:
            return
        try:
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=sensitivity
            )
            self.status_update_signal.emit(f"Gesture: Sensitivity set to {sensitivity}")
        except Exception as e:
            self.status_update_signal.emit(f"Gesture: Failed to set sensitivity - {str(e)}")

    def set_thresholds(self, absent_threshold=30, present_threshold=10):
        """Set custom thresholds for behavior"""
        self.face_absent_threshold = absent_threshold
        self.face_present_threshold = present_threshold
        self.status_update_signal.emit(
            f"Gesture: Thresholds updated - Absent: {absent_threshold}, Present: {present_threshold}"
        )


class AdvancedGestureController(GestureController):
    """Extended gesture controller with hand gestures"""

    def __init__(self, main_window, control_transmission=False, parent=None):
        super().__init__(main_window, control_transmission, parent)

        self.mp_hands = None
        self.hands = None

        self.last_gesture = None
        self.gesture_counter = 0
        self.gesture_threshold = 5
        self.hand_results = None
        
        # FIXED: Add cooldown mechanism to prevent rapid toggling
        self.last_gesture_time = {}
        self.gesture_cooldown = 3.0

    def run(self):
        """Enhanced run method with hand detection"""
        self.running = True
        self.status_update_signal.emit("Advanced Gesture Control: Initializing...")

        camera = self.main_window.client.camera
        if not camera:
            self.status_update_signal.emit("Gesture Control: Error - No camera available")
            return

        try:
            self.status_update_signal.emit("Advanced Gesture Control: Loading models...")
            
            import os
            if hasattr(os, 'nice'):
                try:
                    os.nice(5)
                except:
                    pass
            
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.initialized = True
            self.initialization_complete_signal.emit(True)
            mode = "with camera control" if self.control_transmission else "preview only"
            self.status_update_signal.emit(f"Advanced Gesture Control: Started ({mode})")
            
        except Exception as e:
            self.initialized = False
            self.initialization_complete_signal.emit(False)
            self.status_update_signal.emit(f"Advanced Gesture Control: Initialization failed - {str(e)}")
            return

        while self.running and self.initialized:
            try:
                with self.frame_lock:
                    if self.last_detection_frame is None:
                        time.sleep(0.05)
                        continue
                    frame = self.last_detection_frame.copy()

                rgb_frame = frame

                face_results = self.face_detection.process(rgb_frame)
                face_detected = face_results.detections is not None and len(face_results.detections) > 0

                self.detection_results = face_results

                self.update_face_counters(face_detected)
                
                if self.control_transmission:
                    self.handle_camera_transmission(face_detected)
                else:
                    self.handle_local_hide(face_detected)

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
        """Enhanced drawing with hand landmarks"""
        if not self.initialized:
            return frame
            
        frame = super().draw_detection_boxes(frame)
        if self.local_hide_camera:
            return frame

        if not self.show_detection_boxes or self.hand_results is None:
            return frame

        display_frame = frame.copy()

        try:
            if len(display_frame.shape) == 3 and display_frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = display_frame.copy()

            height, width = frame_bgr.shape[:2]

            if self.hand_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(self.hand_results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(
                        frame_bgr,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )

                    hand_label = "Hand"
                    if self.hand_results.multi_handedness and hand_idx < len(self.hand_results.multi_handedness):
                        hand_label = self.hand_results.multi_handedness[hand_idx].classification[0].label

                    current_gesture = self.detect_gesture(hand_landmarks)
                    if current_gesture:
                        hand_label += f" - {current_gesture}"

                    landmarks = hand_landmarks.landmark
                    x_coords = [lm.x * width for lm in landmarks]
                    y_coords = [lm.y * height for lm in landmarks]
                    x_min, x_max = int(min(x_coords) - 20), int(max(x_coords) + 20)
                    y_min, y_max = int(min(y_coords) - 20), int(max(y_coords) + 20)
                    x_min = max(0, x_min); y_min = max(0, y_min)
                    x_max = min(width - 1, x_max); y_max = min(height - 1, y_max)

                    cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

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

            if len(frame.shape) == 3 and frame.shape[2] == 3:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                return frame_bgr
        except Exception as e:
            print(f"[GESTURE] Hand drawing error: {e}")
            return frame

    def process_hand_gestures(self, hand_landmarks_list):
        """Process hand gestures with cooldown mechanism"""
        for hand_landmarks in hand_landmarks_list:
            gesture = self.detect_gesture(hand_landmarks)

            if gesture == self.last_gesture:
                self.gesture_counter += 1
            else:
                self.gesture_counter = 0
                self.last_gesture = gesture

            if self.gesture_counter >= self.gesture_threshold:
                current_time = time.time()
                last_time = self.last_gesture_time.get(gesture, 0)
                
                if current_time - last_time >= self.gesture_cooldown:
                    self.execute_gesture_command(gesture)
                    self.last_gesture_time[gesture] = current_time
                    self.gesture_counter = 0

    def detect_gesture(self, hand_landmarks):
        """FIXED: Enhanced gesture detection with better finger extension logic"""
        try:
            landmarks = hand_landmarks.landmark

            wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
            thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
            thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_CMC]
            index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_mcp = landmarks[self.mp_hands.HandLandmark.RING_FINGER_MCP]
            pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
            pinky_mcp = landmarks[self.mp_hands.HandLandmark.PINKY_MCP]

            def is_finger_extended(tip, mcp, wrist):
                tip_dist = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)**0.5
                mcp_dist = ((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2)**0.5
                return tip_dist > mcp_dist * 1.2
            
            index_extended = is_finger_extended(index_tip, index_mcp, wrist)
            middle_extended = is_finger_extended(middle_tip, middle_mcp, wrist)
            ring_extended = is_finger_extended(ring_tip, ring_mcp, wrist)
            pinky_extended = is_finger_extended(pinky_tip, pinky_mcp, wrist)
            
            thumb_extended = ((thumb_tip.x - thumb_mcp.x)**2 + (thumb_tip.y - thumb_mcp.y)**2)**0.5 > 0.1
            
            fingers_extended = [
                thumb_extended,
                index_extended,
                middle_extended,
                ring_extended,
                pinky_extended
            ]
            
            num_extended = sum(fingers_extended)

            if (index_extended and 
                not middle_extended and 
                not ring_extended and 
                not pinky_extended and
                not thumb_extended):
                return "pointing_up"

            if (index_extended and 
                middle_extended and 
                not ring_extended and 
                not pinky_extended):
                return "peace_sign"

            if num_extended >= 4:
                return "open_palm"

            if (thumb_extended and
                not index_extended and
                not middle_extended and
                not ring_extended and
                not pinky_extended):
                return "thumbs_up"
            
            if num_extended == 0:
                return "fist"

            return None
            
        except Exception as e:
            print(f"[GESTURE] Gesture detection error: {e}")
            return None

    # In gesture_control.py, replace the execute_gesture_command method in AdvancedGestureController:

    def execute_gesture_command(self, gesture):
        """Execute command based on detected gesture"""
        try:
            if gesture == "thumbs_up":
                current_mic_state = self.main_window.client.microphone_enabled
                self.main_window.toggle_microphone()
                status = "ON" if not current_mic_state else "OFF"
                self.status_update_signal.emit(f"Gesture: Thumbs up → Microphone {status}")

            elif gesture == "peace_sign":
                current_camera_state = self.main_window.client.camera_enabled
                self.main_window.toggle_camera()
                status = "ON" if not current_camera_state else "OFF"
                self.status_update_signal.emit(f"Gesture: Peace sign → Camera {status}")

            elif gesture == "open_palm":
                try:
                    from constants import Message, POST, TEXT
                    
                    # FIXED: Get recipients from server_conn's list, not all_clients
                    recipients = []
                    
                    # Get all clients from the video list widget
                    try:
                        video_list = self.main_window.video_list_widget
                        for client_name in video_list.all_items.keys():
                            if client_name != self.main_window.client.name:
                                recipients.append(client_name)
                    except Exception as e:
                        print(f"[GESTURE] Error getting clients from video list: {e}")
                    
                    # Fallback: try all_clients if available
                    if len(recipients) == 0:
                        try:
                            from client import all_clients
                            for name in all_clients.keys():
                                if name != self.main_window.client.name:
                                    recipients.append(name)
                        except Exception as e:
                            print(f"[GESTURE] Error getting all_clients: {e}")
                    
                    if len(recipients) > 0:
                        hello_msg = "Hello 👋"
                        msg = Message(
                            self.main_window.client.name, 
                            POST, 
                            TEXT, 
                            data=hello_msg, 
                            to_names=tuple(recipients)
                        )
                        self.main_window.server_conn.send_msg(
                            self.main_window.server_conn.main_socket, 
                            msg
                        )
                        
                        self.main_window.chat_widget.add_msg(
                            "You", 
                            ", ".join(recipients), 
                            hello_msg
                        )
                        self.status_update_signal.emit(f"Gesture: Open palm → Sent Hello 👋 to {len(recipients)} client(s)")
                    else:
                        self.status_update_signal.emit("Gesture: Open palm → No other clients connected")
                except Exception as e:
                    print(f"[GESTURE] Error sending hello: {e}")
                    import traceback
                    traceback.print_exc()
                    self.status_update_signal.emit(f"Gesture: Error sending hello - {str(e)}")
                    
            elif gesture == "fist":
                if self.main_window.client.camera_enabled or self.main_window.client.microphone_enabled:
                    if self.main_window.client.camera_enabled:
                        self.main_window.toggle_camera()
                    if self.main_window.client.microphone_enabled:
                        self.main_window.toggle_microphone()
                    self.status_update_signal.emit("Gesture: Fist → Privacy Mode (Muted all)")
                else:
                    if not self.main_window.client.camera_enabled:
                        self.main_window.toggle_camera()
                    if not self.main_window.client.microphone_enabled:
                        self.main_window.toggle_microphone()
                    self.status_update_signal.emit("Gesture: Fist → Unmuted all")
                    
            elif gesture == "pointing_up":
                try:
                    from constants import Message, POST, TEXT
                    
                    # FIXED: Get recipients from video list widget
                    recipients = []
                    try:
                        video_list = self.main_window.video_list_widget
                        for client_name in video_list.all_items.keys():
                            if client_name != self.main_window.client.name:
                                recipients.append(client_name)
                    except Exception as e:
                        print(f"[GESTURE] Error getting clients: {e}")
                    
                    # Fallback
                    if len(recipients) == 0:
                        try:
                            from client import all_clients
                            recipients = [name for name in all_clients.keys() if name != self.main_window.client.name]
                        except Exception as e:
                            print(f"[GESTURE] Error with fallback: {e}")
                    
                    if len(recipients) > 0:
                        attention_msg = "✋ [Raised Hand - Requesting Attention]"
                        msg = Message(
                            self.main_window.client.name,
                            POST,
                            TEXT,
                            data=attention_msg,
                            to_names=tuple(recipients)
                        )
                        self.main_window.server_conn.send_msg(
                            self.main_window.server_conn.main_socket,
                            msg
                        )
                        self.main_window.chat_widget.add_msg("You", ", ".join(recipients), attention_msg)
                        self.status_update_signal.emit(f"Gesture: Pointing up → Raised hand to {len(recipients)} client(s)")
                    else:
                        self.status_update_signal.emit("Gesture: Pointing up → No other clients connected")
                except Exception as e:
                    print(f"[GESTURE] Error raising hand: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"[GESTURE] Command execution error: {e}")
            import traceback
            traceback.print_exc()


def integrate_gesture_control(main_window, control_transmission=False):
    """
    Helper to integrate gesture control with the main window
    
    Args:
        main_window: MainWindow instance
        control_transmission: If True, control actual camera transmission.
                            If False, only affect local preview (default)
    """
    try:
        import mediapipe
    except ImportError:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(
            main_window,
            "Missing Dependency",
            "MediaPipe library is required for gesture control.\n"
            "Install with: pip install mediapipe"
        )
        return None

    gesture_controller = AdvancedGestureController(main_window, control_transmission)
    return gesture_controller