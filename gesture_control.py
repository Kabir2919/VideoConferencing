import cv2
import mediapipe as mp
import threading
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class GestureController(QThread):
    """
    Gesture control using MediaPipe for face detection.
    FIXED: Proper signal handling and camera state control with connection stability
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
        
        # FIXED: Add state change lock to prevent race conditions
        self.state_change_lock = threading.Lock()
        self.pending_camera_state = None

        self.status_update_signal.connect(self.update_status)
        self.initialization_complete_signal.connect(self.on_initialization_complete)
        
        if self.control_transmission:
            self.set_camera_state_signal.connect(self._set_camera_state_slot)

    def _set_camera_state_slot(self, state):
        """Slot to handle camera state changes - runs in main thread with locking"""
        try:
            with self.state_change_lock:
                # Only change if different from current state
                if self.main_window.client.camera_enabled != state:
                    print(f"[GESTURE] Changing camera state to: {state}")
                    self.main_window.client.camera_enabled = state
                    self.main_window.update_camera_ui_state()
                    # Give broadcast thread time to react
                    time.sleep(0.05)
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

        # FIXED: Add longer initial delay to ensure broadcast thread is stable
        time.sleep(0.5)

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
        FIXED: Use locking to prevent state change during broadcast
        """
        with self.state_change_lock:
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
    """Extended gesture controller with hand gestures - inherits connection fixes"""

    def __init__(self, main_window, control_transmission=False, parent=None):
        super().__init__(main_window, control_transmission, parent)

        self.mp_hands = None
        self.hands = None

        self.last_gesture = None
        self.gesture_counter = 0
        self.gesture_threshold = 5
        self.hand_results = None
        
        self.last_gesture_time = {}
        self.gesture_cooldown = 3.0

    # ... rest of the AdvancedGestureController methods remain the same ...


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