import os
import cv2
import pyaudio
import numpy as np
import urllib.request
from PyQt6.QtCore import Qt, QThread, QTimer, QSize, QRunnable, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QActionGroup, QIcon, QFont, QPalette, QColor
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout, QDockWidget \
    , QLabel, QWidget, QListWidget, QListWidgetItem, QMessageBox \
    , QComboBox, QTextEdit, QLineEdit, QPushButton, QFileDialog \
    , QDialog, QMenu, QWidgetAction, QCheckBox, QFrame, QSizePolicy

from constants import *

# Camera
CAMERA_RES = '240p'
LAYOUT_RES = '240p'
frame_size = {
    '240p': (352, 240),
    '360p': (480, 360),
    '480p': (640, 480),
    '560p': (800, 560),
    '720p': (1280, 720),
    '900p': (1400, 900),
    '1080p': (1920, 1080)
}

FRAME_WIDTH = frame_size[CAMERA_RES][0]
FRAME_HEIGHT = frame_size[CAMERA_RES][1]

# Image Encoding
ENABLE_ENCODE = True
ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

# Audio
ENABLE_AUDIO = True
SAMPLE_RATE = 44100
BLOCK_SIZE = 512

# Create default frames if image files don't exist
def create_default_frame(width, height, text, color=(64, 64, 64)):
    """Create a default frame with text if image files are missing"""
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_color = (255, 255, 255)
    
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    x = (width - text_width) // 2
    y = (height + text_height) // 2
    
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)
    return frame


def download_image_from_url(url, target_width, target_height):
    """Download and process image from URL"""
    try:
        print(f"[INFO] Downloading image from: {url}")
        
        # Download image
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise Exception("Failed to decode image")
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target dimensions
        img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        print(f"[INFO] Successfully downloaded and processed image")
        return img
        
    except Exception as e:
        print(f"[WARNING] Failed to download image from URL: {e}")
        return None


# Try to download camera off image from URL
NOCAM_IMAGE_URL = "https://www.shutterstock.com/image-vector/video-cam-off-icon-no-260nw-1750968803.jpg"
NOCAM_FRAME = download_image_from_url(NOCAM_IMAGE_URL, FRAME_WIDTH, FRAME_HEIGHT)

# Fallback to default frame if download fails
if NOCAM_FRAME is None:
    print("[WARNING] Using default 'Camera Off' frame")
    NOCAM_FRAME = create_default_frame(FRAME_WIDTH, FRAME_HEIGHT, "Camera Off")

# Microphone off image - using default
NOMIC_FRAME = create_default_frame(100, 50, "Mic Off", (128, 0, 0))

# Audio
pa = pyaudio.PyAudio()

# Modern UI Color Scheme
COLORS = {
    'primary': '#2563eb',
    'primary_hover': '#1d4ed8',
    'secondary': '#64748b',
    'success': '#10b981',
    'success_hover': '#059669',
    'danger': '#ef4444',
    'danger_hover': '#dc2626',
    'warning': '#f59e0b',
    'warning_hover': '#d97706',
    'background': '#f8fafc',
    'surface': '#ffffff',
    'surface_hover': '#f1f5f9',
    'text': '#1e293b',
    'text_muted': '#64748b',
    'border': '#e2e8f0',
    'border_focus': '#3b82f6'
}


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        self.fn(*self.args, **self.kwargs)


class Microphone:
    def __init__(self):
        self.stream = None
        self.microphone_available = False
        
        try:
            self.stream = pa.open(
                rate=SAMPLE_RATE,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=BLOCK_SIZE
            )
            self.microphone_available = True
            print("[INFO] Microphone initialized successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize microphone: {e}")
            self.microphone_available = False

    def get_data(self):
        if not self.microphone_available or self.stream is None:
            return None
            
        try:
            return self.stream.read(BLOCK_SIZE, exception_on_overflow=False)
        except Exception as e:
            print(f"[ERROR] Microphone data capture failed: {e}")
            return None


class AudioThread(QThread):
    def __init__(self, client, parent=None):
        super().__init__(parent)
        self.client = client
        self.stream = pa.open(
            rate=SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            output=True,
            frames_per_buffer=BLOCK_SIZE
        )
        self.connected = True

    def run(self):
        if self.client.microphone is not None:
            return
        while self.connected:
            self.update_audio()

    def update_audio(self):
        data = self.client.get_audio()
        if data is not None:
            self.stream.write(data)


class Camera:
    def __init__(self):
        self.cap = None
        self.camera_available = False
        self.gesture_controller = None
        self.max_frame_size = 28000
        
        try:
            self.cap = cv2.VideoCapture(0)
            
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                self.cap.set(cv2.CAP_PROP_SATURATION, 128)
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
                self.cap.set(cv2.CAP_PROP_CONTRAST, 128)
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
                
                self.camera_available = True
                print("[INFO] Camera initialized successfully")
            else:
                print("[WARNING] No camera available")
                self.camera_available = False
                
        except Exception as e:
            print(f"[ERROR] Failed to initialize camera: {e}")
            self.camera_available = False

    def set_gesture_controller(self, gesture_controller):
        """Set reference to gesture controller for drawing overlays"""
        self.gesture_controller = gesture_controller
    
    def get_frame(self, apply_overlays=False):
        """Get camera frame with AGGRESSIVE compression to meet packet size limits"""
        if not self.camera_available or self.cap is None:
            return None

        try:
            ret, frame = self.cap.read()
            if not ret:
                return None

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if (self.gesture_controller is not None
                and hasattr(self.gesture_controller, 'update_frame_for_detection')
                and self.gesture_controller.running):
                try:
                    self.gesture_controller.update_frame_for_detection(frame.copy())
                except Exception as e:
                    print(f"[CAMERA] Gesture update error (non-fatal): {e}")

            tx_frame = frame.copy()
            
            if ENABLE_ENCODE:
                quality = 50
                max_attempts = 5
                
                for attempt in range(max_attempts):
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                    success, encoded = cv2.imencode('.jpg', cv2.cvtColor(tx_frame, cv2.COLOR_RGB2BGR), encode_param)
                    
                    if not success:
                        print(f"[CAMERA] Encoding failed at quality {quality}")
                        return None
                    
                    estimated_packet_size = len(encoded) + 500
                    
                    if estimated_packet_size <= self.max_frame_size:
                        return encoded
                    
                    quality = max(10, quality - 10)
                    if attempt == max_attempts - 1:
                        print(f"[CAMERA] Warning: Using minimum quality to meet packet size")
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
                        success, encoded = cv2.imencode('.jpg', cv2.cvtColor(tx_frame, cv2.COLOR_RGB2BGR), encode_param)
                        if success and len(encoded) + 500 <= self.max_frame_size:
                            return encoded
                        else:
                            print(f"[CAMERA] ERROR: Cannot encode frame small enough ({len(encoded) + 500} bytes)")
                            return None
                
                return None
            else:
                return tx_frame

        except Exception as e:
            print(f"[ERROR] Camera frame capture failed: {e}")
            return None

class VideoControlButton(QPushButton):
    """Circular control button for video interface"""
    def __init__(self, text="", icon_text="", button_type="primary", size=50, parent=None):
        super().__init__(text, parent)
        self.button_type = button_type
        self.icon_text = icon_text
        self.button_size = size
        self.is_active = True
        self.init_style()
        
    def init_style(self):
        self.setFixedSize(self.button_size, self.button_size)
        
        if self.is_active:
            if self.button_type == "camera":
                bg_color = COLORS['success']
                bg_hover = COLORS['success_hover']
            elif self.button_type == "microphone":
                bg_color = COLORS['primary']
                bg_hover = COLORS['primary_hover']
            else:
                bg_color = COLORS['secondary']
                bg_hover = COLORS['text_muted']
        else:
            bg_color = COLORS['danger']
            bg_hover = COLORS['danger_hover']
            
        style = f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                border-radius: {self.button_size // 2}px;
                font-size: 18px;
                font-weight: bold;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {bg_hover};
            }}
            QPushButton:pressed {{
                background-color: {bg_hover};
            }}
        """
        self.setStyleSheet(style)
        
    def toggle_state(self):
        """Toggle button active/inactive state"""
        self.is_active = not self.is_active
        self.init_style()
        return self.is_active


class VideoControlsOverlay(QWidget):
    """Floating controls overlay for video interface"""
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.init_ui()
        
    def init_ui(self):
        self.setStyleSheet(f"""
            VideoControlsOverlay {{
                background-color: rgba(0, 0, 0, 100);
                border-radius: 25px;
                padding: 10px;
            }}
        """)
        
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(20)
        
        self.camera_btn = VideoControlButton(
            icon_text="📹", 
            button_type="camera", 
            size=60
        )
        self.camera_btn.setText("📹")
        self.camera_btn.setToolTip("Toggle Camera")
        self.camera_btn.clicked.connect(self.toggle_camera)
        
        self.mic_btn = VideoControlButton(
            icon_text="🎤", 
            button_type="microphone", 
            size=60
        )
        self.mic_btn.setText("🎤")
        self.mic_btn.setToolTip("Toggle Microphone")
        self.mic_btn.clicked.connect(self.toggle_microphone)
        
        self.end_call_btn = VideoControlButton(
            icon_text="📞", 
            button_type="danger", 
            size=60
        )
        self.end_call_btn.setText("📞")
        self.end_call_btn.setToolTip("End Call")
        self.end_call_btn.is_active = False
        self.end_call_btn.init_style()
        self.end_call_btn.clicked.connect(self.main_window.close)
        
        self.settings_btn = VideoControlButton(
            icon_text="⚙️", 
            button_type="secondary", 
            size=50
        )
        self.settings_btn.setText("⚙️")
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.clicked.connect(self.show_settings_menu)
        
        layout.addWidget(self.camera_btn)
        layout.addWidget(self.mic_btn)
        layout.addWidget(self.end_call_btn)
        layout.addWidget(self.settings_btn)
        
        self.setLayout(layout)
        
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        
    def toggle_camera(self):
        """Toggle camera on/off - let MainWindow handle the state"""
        self.main_window.toggle_camera()
        
    def toggle_microphone(self):
        """Toggle microphone on/off - let MainWindow handle the state"""  
        self.main_window.toggle_microphone()
            
    def show_settings_menu(self):
        """Show settings menu"""
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {COLORS['surface']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 8px;
                color: {COLORS['text']};
            }}
            QMenu::item {{
                padding: 8px 16px;
                border-radius: 4px;
                color: {COLORS['text']};
            }}
            QMenu::item:selected {{
                background-color: {COLORS['primary']};
                color: white;
            }}
        """)
        
        layout_menu = menu.addMenu("Layout")
        layout_action_group = QActionGroup(self)
        
        for res in frame_size.keys():
            layout_action = layout_action_group.addAction(res)
            layout_action.setCheckable(True)
            layout_action.triggered.connect(
                lambda checked, res=res: self.main_window.video_list_widget.resize_widgets(res)
            )
            if res == LAYOUT_RES:
                layout_action.setChecked(True)
            layout_menu.addAction(layout_action)
        
        menu.addSeparator()
        
        if hasattr(self.main_window, 'gesture_controller') and self.main_window.gesture_controller:
            gesture_action = menu.addAction("Stop Gesture Control")
            gesture_action.triggered.connect(self.main_window.toggle_gesture_control)
        else:
            gesture_action = menu.addAction("Start Gesture Control")
            gesture_action.triggered.connect(self.main_window.toggle_gesture_control)
        
        menu.addSeparator()
        
        chat_action = menu.addAction("Toggle Chat Panel")
        chat_action.triggered.connect(self.toggle_chat_panel)
        
        menu.exec(self.settings_btn.mapToGlobal(self.settings_btn.rect().bottomLeft()))
        
    def toggle_chat_panel(self):
        """Toggle chat panel visibility"""
        if self.main_window.sidebar.isVisible():
            self.main_window.sidebar.hide()
        else:
            self.main_window.sidebar.show()
        
    def update_position(self, parent_geometry):
        """Update overlay position to stay at bottom center of parent"""
        overlay_width = self.sizeHint().width()
        overlay_height = self.sizeHint().height()
        
        x = parent_geometry.center().x() - overlay_width // 2
        y = parent_geometry.bottom() - overlay_height - 30
        
        self.move(x, y)


class VideoWidget(QWidget):
    def __init__(self, client, parent=None):
        super().__init__(parent)
        self.client = client
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)

        self.init_video()

    def init_ui(self):
        self.setStyleSheet(f"""
            VideoWidget {{
                background-color: {COLORS['surface']};
                border: 2px solid {COLORS['border']};
                border-radius: 12px;
                margin: 4px;
            }}
            VideoWidget:hover {{
                border-color: {COLORS['primary']};
            }}
        """)

        self.video_viewer = QLabel()
        self.video_viewer.setStyleSheet(f"""
            QLabel {{
                border-radius: 8px;
                background-color: #000000;
            }}
        """)
        
        if self.client.current_device:
            self.name_label = QLabel(f"You - {self.client.name}")
        else:
            self.name_label = QLabel(self.client.name)
        
        self.name_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text']};
                font-weight: 600;
                font-size: 12px;
                padding: 4px 8px;
                background-color: {COLORS['surface']};
                border-radius: 6px;
                margin: 4px;
            }}
        """)
        
        self.video_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.addWidget(self.video_viewer)
        self.layout.addWidget(self.name_label)
        self.setLayout(self.layout)
    
    def init_video(self):
        self.timer.start(30)
    
    def update_video(self):
        """
        FIXED: Proper handling of camera states and gesture control
        - When camera is disabled via button: show NOCAM_FRAME
        - When gesture control hides local preview: show NOCAM_FRAME
        - Never show generic text overlay
        """
        if self.client.current_device:
            # LOCAL CLIENT
            # Check if camera is disabled
            if not self.client.camera_enabled:
                frame = NOCAM_FRAME.copy()
            else:
                # Camera is enabled, get frame
                frame = self.client.camera.get_frame(apply_overlays=False) if self.client.camera else None
                
                if frame is None:
                    frame = NOCAM_FRAME.copy()
                elif ENABLE_ENCODE and isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and len(frame.shape) == 1:
                    # This is an encoded frame (1D array of bytes)
                    try:
                        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                        if frame is not None:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        else:
                            frame = NOCAM_FRAME.copy()
                    except Exception as e:
                        print(f"[VIDEO] Decode error: {e}")
                        frame = NOCAM_FRAME.copy()
                
                # Apply gesture overlays ONLY if camera is enabled
                camera = self.client.camera
                if (camera and camera.gesture_controller is not None
                    and hasattr(camera.gesture_controller, 'draw_detection_boxes')
                    and camera.gesture_controller.running
                    and not camera.gesture_controller.local_hide_camera):  # Don't draw if locally hidden
                    # Apply gesture detection boxes
                    frame = camera.gesture_controller.draw_detection_boxes(frame)
                elif (camera and camera.gesture_controller is not None
                      and hasattr(camera.gesture_controller, 'local_hide_camera')
                      and camera.gesture_controller.local_hide_camera):
                    # Gesture control is hiding local preview - show camera off image
                    frame = NOCAM_FRAME.copy()
        else:
            # REMOTE CLIENT
            frame = self.client.video_frame  # Get the stored frame
            
            if frame is None:
                # No frame available - camera is off
                frame = NOCAM_FRAME.copy()
            elif isinstance(frame, bytes) and frame == CAMERA_OFF_MARKER:
                # Explicit camera off marker
                frame = NOCAM_FRAME.copy()
            elif ENABLE_ENCODE and isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and len(frame.shape) == 1:
                # This is an encoded frame - decode it
                try:
                    decoded = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    if decoded is not None:
                        frame = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                    else:
                        print(f"[VIDEO] Failed to decode frame from {self.client.name}")
                        frame = NOCAM_FRAME.copy()
                except Exception as e:
                    print(f"[VIDEO] Decode error for {self.client.name}: {e}")
                    frame = NOCAM_FRAME.copy()
            elif not isinstance(frame, np.ndarray):
                # Invalid frame type
                print(f"[VIDEO] Invalid frame type from {self.client.name}: {type(frame)}")
                frame = NOCAM_FRAME.copy()
        
        # Resize to display size
        try:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"[VIDEO] Resize error: {e}")
            frame = NOCAM_FRAME.copy()
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        
        # Add microphone indicator if no audio
        if self.client.audio_data is None:
            try:
                nomic_h, nomic_w, _ = NOMIC_FRAME.shape
                x, y = FRAME_WIDTH//2 - nomic_w//2, FRAME_HEIGHT - 50
                frame[y:y+nomic_h, x:x+nomic_w] = NOMIC_FRAME.copy()
            except Exception as e:
                print(f"[VIDEO] Mic indicator error: {e}")

        # Display the frame
        try:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_viewer.setPixmap(QPixmap.fromImage(q_img))
        except Exception as e:
            print(f"[VIDEO] Display error: {e}")

class VideoListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.all_items = {}
        self.init_ui()

    def init_ui(self):
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS['background']};
                border: none;
                padding: 16px;
            }}
            QListWidget::item {{
                border: none;
                padding: 4px;
            }}
        """)

    def add_client(self, client):
        video_widget = VideoWidget(client)

        item = QListWidgetItem()
        item.setFlags(item.flags() & ~(Qt.ItemFlag.ItemIsSelectable|Qt.ItemFlag.ItemIsEnabled))
        if client.current_device:
            self.insertItem(0, item)
        else:
            self.addItem(item)
        item.setSizeHint(QSize(FRAME_WIDTH + 20, FRAME_HEIGHT + 60))
        self.setItemWidget(item, video_widget)
        self.all_items[client.name] = item
        self.resize_widgets()
    
    def resize_widgets(self, res: str = None):
        global FRAME_WIDTH, FRAME_HEIGHT, LAYOUT_RES
        n = self.count()
        if res is None:
            if n <= 1:
                res = "900p"
            elif n <= 4:
                res = "480p"
            elif n <= 6:
                res = "360p"
            else:
                res = "240p"
        new_size = frame_size[res]
        
        if new_size == (FRAME_WIDTH, FRAME_HEIGHT):
            return
        else:
            FRAME_WIDTH, FRAME_HEIGHT = new_size
            LAYOUT_RES = res
        
        for i in range(n):
            self.item(i).setSizeHint(QSize(FRAME_WIDTH + 20, FRAME_HEIGHT + 60))

    def remove_client(self, name: str):
        self.takeItem(self.row(self.all_items[name]))
        self.all_items.pop(name)
        self.resize_widgets()


class ModernButton(QPushButton):
    def __init__(self, text, button_type="primary", parent=None):
        super().__init__(text, parent)
        self.button_type = button_type
        self.init_style()
    
    def init_style(self):
        base_style = f"""
            QPushButton {{
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 8px;
                border: none;
                min-width: 100px;
            }}
        """
        
        if self.button_type == "primary":
            style = base_style + f"""
                QPushButton {{
                    background-color: {COLORS['primary']};
                    color: white;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['primary_hover']};
                }}
                QPushButton:pressed {{
                    background-color: {COLORS['primary_hover']};
                    transform: translateY(1px);
                }}
            """
        elif self.button_type == "success":
            style = base_style + f"""
                QPushButton {{
                    background-color: {COLORS['success']};
                    color: white;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['success_hover']};
                }}
            """
        elif self.button_type == "danger":
            style = base_style + f"""
                QPushButton {{
                    background-color: {COLORS['danger']};
                    color: white;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['danger_hover']};
                }}
            """
        elif self.button_type == "warning":
            style = base_style + f"""
                QPushButton {{
                    background-color: {COLORS['warning']};
                    color: white;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['warning_hover']};
                }}
            """
        else:
            style = base_style + f"""
                QPushButton {{
                    background-color: {COLORS['surface']};
                    color: {COLORS['text']};
                    border: 2px solid {COLORS['border']};
                }}
                QPushButton:hover {{
                    background-color: {COLORS['surface_hover']};
                    border-color: {COLORS['border_focus']};
                }}
            """
        
        self.setStyleSheet(style)


class ChatWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet(f"""
            ChatWidget {{
                background-color: {COLORS['surface']};
                border-radius: 12px;
            }}
        """)
        
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(16, 16, 16, 16)
        self.layout.setSpacing(12)
        self.setLayout(self.layout)

        self.central_widget = QTextEdit(self)
        self.central_widget.setReadOnly(True)
        self.central_widget.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['background']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 12px;
                font-size: 13px;
                line-height: 1.4;
            }}
        """)
        self.layout.addWidget(self.central_widget)

        self.clients_menu = QMenu("Clients", self)
        self.clients_menu.aboutToShow.connect(self.resize_clients_menu)
        self.clients_menu.setStyleSheet(f"""
            QMenu {{
                background-color: {COLORS['surface']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 8px;
            }}
            QMenu::item {{
                padding: 4px 12px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: {COLORS['primary']};
                color: white;
            }}
        """)
        self.clients_checkboxes = {}
        self.clients_menu_actions = {}

        self.select_all_checkbox, _ = self.add_client("")
        self.clients_menu.addSeparator()

        self.clients_button = ModernButton("Select Recipients", "secondary", self)
        self.clients_button.setMenu(self.clients_menu)
        self.layout.addWidget(self.clients_button)

        buttons_frame = QFrame()
        buttons_layout = QHBoxLayout(buttons_frame)
        buttons_layout.setSpacing(12)
        
        self.file_button = ModernButton("Send File", "secondary", self)
        self.gesture_button = ModernButton("Gesture Control", "warning", self)
        
        buttons_layout.addWidget(self.file_button)
        buttons_layout.addWidget(self.gesture_button)
        
        self.layout.addWidget(buttons_frame)

        message_frame = QFrame()
        message_layout = QVBoxLayout(message_frame)
        message_layout.setSpacing(8)
        
        self.line_edit = QLineEdit(self)
        self.line_edit.setPlaceholderText("Type your message here...")
        self.line_edit.setStyleSheet(f"""
            QLineEdit {{
                padding: 6px 10px;
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                font-size: 14px;
                background-color: {COLORS['surface']};
                min-height: 32px;
            }}
            QLineEdit:focus {{
                border-color: {COLORS['border_focus']};
                outline: none;
            }}
        """)
        
        self.send_button = ModernButton("Send Message", "primary", self)
        
        message_layout.addWidget(self.line_edit)
        message_layout.addWidget(self.send_button)
        
        self.layout.addWidget(message_frame)

        self.layout.addSpacing(20)

        self.end_button = ModernButton("End Call", "danger", self)
        self.layout.addWidget(self.end_button)
    
    def add_client(self, name: str):
        checkbox = QCheckBox(name, self)
        checkbox.setChecked(True)
        checkbox.setStyleSheet(f"""
            QCheckBox {{
                padding: 4px;
                font-size: 13px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 2px solid {COLORS['border']};
                border-radius: 3px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLORS['primary']};
                border-color: {COLORS['primary']};
            }}
        """)
        
        action_widget = QWidgetAction(self)
        action_widget.setDefaultWidget(checkbox)
        self.clients_menu.addAction(action_widget)

        if name == "":
            checkbox.setText("Select All")
            checkbox.stateChanged.connect(
                lambda state: self.on_checkbox_click(state, is_select_all=True)
            )
            return checkbox, action_widget
        
        checkbox.stateChanged.connect(
            lambda state: self.on_checkbox_click(state)
        )
        self.clients_checkboxes[name] = checkbox
        self.clients_menu_actions[name] = action_widget
    
    def remove_client(self, name: str):
        self.clients_menu.removeAction(self.clients_menu_actions[name])
        self.clients_menu_actions.pop(name)
        self.clients_checkboxes.pop(name)

    def resize_clients_menu(self):
        self.clients_menu.setMinimumWidth(self.clients_button.width())
    
    def on_checkbox_click(self, is_checked: bool, is_select_all: bool = False):
        if is_select_all:
            for client_checkbox in self.clients_checkboxes.values():
                client_checkbox.blockSignals(True)
                client_checkbox.setChecked(is_checked)
                client_checkbox.blockSignals(False)
        else:
            if not is_checked:
                self.select_all_checkbox.blockSignals(True)
                self.select_all_checkbox.setChecked(False)
                self.select_all_checkbox.blockSignals(False)
    
    def selected_clients(self):
        selected = []
        for name, checkbox in self.clients_checkboxes.items():
            if checkbox.isChecked():
                selected.append(name)
        return tuple(selected)

    def get_file(self):
        file_path = QFileDialog.getOpenFileName(None, "Select File", options=QFileDialog.Option.DontUseNativeDialog)[0]
        return file_path

    def get_text(self):
        text = self.line_edit.text()
        self.line_edit.clear()
        return text

    def add_msg(self, from_name: str, to_name: str, msg: str):
        """Add message to chat with proper text colors"""
        formatted_msg = f"<div style='margin: 8px 0; padding: 8px; background-color: {COLORS['surface_hover']}; border-radius: 6px;'>"
        formatted_msg += f"<span style='font-weight: 600; color: {COLORS['primary']};'>{from_name}</span>"
        formatted_msg += f" <span style='color: {COLORS['text_muted']}; font-size: 12px;'>→ {to_name}</span><br>"
        # FIXED: Use text color instead of white
        formatted_msg += f"<span style='color: {COLORS['text']};'>{msg}</span></div>"
        self.central_widget.insertHtml(formatted_msg)
        
        scrollbar = self.central_widget.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Join Video Conference")
        self.setFixedSize(400, 200)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['surface']};
            }}
        """)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(30, 30, 30, 30)
        self.layout.setSpacing(20)
        self.setLayout(self.layout)

        title_label = QLabel("Enter Your Username")
        title_label.setStyleSheet(f"""
            QLabel {{
                font-size: 20px;
                font-weight: 700;
                color: {COLORS['text']};
                qproperty-alignment: AlignCenter;
            }}
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(title_label)

        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("Enter your username...")
        self.name_edit.setStyleSheet(f"""
            QLineEdit {{
                padding: 6px 10px;
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                font-size: 14px;
                color: black;
                min-height: 32px;
                background-color: {COLORS['background']};
            }}
            QLineEdit:focus {{
                border-color: {COLORS['border_focus']};
            }}
        """)

        self.layout.addWidget(self.name_edit)

        self.button = ModernButton("Join Conference", "primary", self)
        self.layout.addWidget(self.button)

        self.button.clicked.connect(self.login)
        self.name_edit.returnPressed.connect(self.login)
    
    def get_name(self):
        return self.name_edit.text()
    
    def login(self):
        if self.get_name() == "":
            QMessageBox.critical(self, "Error", "Username cannot be empty")
            return
        if " " in self.get_name():
            QMessageBox.critical(self, "Error", "Username cannot contain spaces")
            return
        self.accept()
    
    def close(self):
        self.reject()


class MainWindow(QMainWindow):
    def __init__(self, client, server_conn):
        super().__init__()
        self.client = client
        self.server_conn = server_conn
        self.audio_threads = {}

        self.server_conn.add_client_signal.connect(self.add_client)
        self.server_conn.remove_client_signal.connect(self.remove_client)
        self.server_conn.add_msg_signal.connect(self.add_msg)

        self.login_dialog = LoginDialog(self)
        if not self.login_dialog.exec():
            exit()
        
        self.server_conn.name = self.login_dialog.get_name()
        self.server_conn.start()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Video Conferencing - Modern Interface")
        self.setGeometry(0, 0, 1920, 1000)
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['background']};
            }}
        """)

        self.video_list_widget = VideoListWidget()
        self.setCentralWidget(self.video_list_widget)

        self.chat_widget = ChatWidget()

        self.sidebar = QDockWidget("Chat & Controls", self)
        self.sidebar.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.sidebar.setStyleSheet(f"""
            QDockWidget {{
                background-color: {COLORS['surface']};
                border: none;
            }}
            QDockWidget::title {{
                background-color: {COLORS['primary']};
                color: white;
                padding: 12px;
                font-weight: 600;
                font-size: 14px;
            }}
        """)
        
        self.sidebar.setWidget(self.chat_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.sidebar)
        
        self.chat_widget.send_button.clicked.connect(lambda: self.send_msg(TEXT))
        self.chat_widget.line_edit.returnPressed.connect(lambda: self.send_msg(TEXT))
        self.chat_widget.file_button.clicked.connect(lambda: self.send_msg(FILE))
        self.chat_widget.gesture_button.clicked.connect(self.toggle_gesture_control)
        self.chat_widget.end_button.clicked.connect(self.close)

        self.setup_menu_bar()
        
        self.controls_overlay = VideoControlsOverlay(self)
        self.controls_overlay.show()
        
        self.update_camera_ui_state()
        self.update_microphone_ui_state()

    def setup_menu_bar(self):
        menubar = self.menuBar()
        menubar.setStyleSheet(f"""
            QMenuBar {{
                background-color: {COLORS['surface']};
                border-bottom: 2px solid {COLORS['border']};
                padding: 8px;
                font-weight: 600;
                color: {COLORS['text']};
            }}
            QMenuBar::item {{
                background-color: transparent;
                color: {COLORS['text']};
                padding: 8px 16px;
                margin: 0 4px;
                border-radius: 6px;
            }}
            QMenuBar::item:selected {{
                background-color: {COLORS['primary']};
                color: white;
            }}
            QMenuBar::item:hover {{
                background-color: {COLORS['surface_hover']};
                color: {COLORS['text']};
            }}
            QMenu {{
                background-color: {COLORS['surface']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 8px;
                color: {COLORS['text']};
            }}
            QMenu::item {{
                padding: 8px 16px;
                border-radius: 4px;
                color: {COLORS['text']};
            }}
            QMenu::item:selected {{
                background-color: {COLORS['primary']};
                color: white;
            }}
            QMenu::item:hover {{
                background-color: {COLORS['surface_hover']};
                color: {COLORS['text']};
            }}
        """)

        self.camera_menu = menubar.addMenu("📹 Camera")
        self.camera_menu.addAction("Disable", self.toggle_camera)
        
        self.microphone_menu = menubar.addMenu("🎤 Microphone")
        self.microphone_menu.addAction("Disable", self.toggle_microphone)
        
        self.layout_menu = menubar.addMenu("📐 Layout")
        self.layout_actions = {}
        layout_action_group = QActionGroup(self)
        
        for res in frame_size.keys():
            layout_action = layout_action_group.addAction(res)
            layout_action.setCheckable(True)
            layout_action.triggered.connect(lambda checked, res=res: self.video_list_widget.resize_widgets(res))
            if res == LAYOUT_RES:
                layout_action.setChecked(True)
            self.layout_menu.addAction(layout_action)
            self.layout_actions[res] = layout_action
    
    def add_client(self, client):
        self.video_list_widget.add_client(client)
        self.layout_actions[LAYOUT_RES].setChecked(True)
        if ENABLE_AUDIO:
            self.audio_threads[client.name] = AudioThread(client, self)
            self.audio_threads[client.name].start()
        if not client.current_device:
            self.chat_widget.add_client(client.name)
    
    def remove_client(self, name: str):
        self.video_list_widget.remove_client(name)
        self.layout_actions[LAYOUT_RES].setChecked(True)
        if ENABLE_AUDIO:
            self.audio_threads[name].connected = False
            self.audio_threads[name].wait()
            self.audio_threads.pop(name)
            print(f"Audio Thread for {name} terminated")
        print(f"removing {name} chat...")
        self.chat_widget.remove_client(name)
        print(f"{name} removed")

    def send_msg(self, data_type: str = TEXT):
        selected = self.chat_widget.selected_clients()
        if len(selected) == 0:
            QMessageBox.critical(self, "Error", "Select at least one recipient")
            return
        
        if data_type == TEXT:
            msg_text = self.chat_widget.get_text()
        elif data_type == FILE:
            filepath = self.chat_widget.get_file()
            if not filepath:
                return
            msg_text = os.path.basename(filepath)
        else:
            print(f"{data_type} data_type not supported")
            return
        
        if msg_text == "":
            QMessageBox.critical(self, "Error", f"{data_type} cannot be empty")
            return
        
        msg = Message(self.client.name, POST, data_type, data=msg_text, to_names=selected)
        self.server_conn.send_msg(self.server_conn.main_socket, msg)
        
        if data_type == FILE:
            send_file_thread = Worker(self.server_conn.send_file, filepath, selected)
            self.server_conn.threadpool.start(send_file_thread)
            msg_text = f"Sending {msg_text}..."

        self.chat_widget.add_msg("You", ", ".join(selected), msg_text)
    
    def add_msg(self, from_name: str, msg: str):
        self.chat_widget.add_msg(from_name, "You", msg)
    
    def update_camera_ui_state(self):
        """Update all camera-related UI elements to match current state"""
        if self.client.camera_enabled:
            self.camera_menu.actions()[0].setText("Disable")
            if hasattr(self, 'controls_overlay'):
                self.controls_overlay.camera_btn.is_active = True
                self.controls_overlay.camera_btn.init_style()
                self.controls_overlay.camera_btn.setText("📹")
                self.controls_overlay.camera_btn.setToolTip("Camera On - Click to turn off")
        else:
            self.camera_menu.actions()[0].setText("Enable")
            if hasattr(self, 'controls_overlay'):
                self.controls_overlay.camera_btn.is_active = False
                self.controls_overlay.camera_btn.init_style()
                self.controls_overlay.camera_btn.setText("📹")
                self.controls_overlay.camera_btn.setToolTip("Camera Off - Click to turn on")

    def update_microphone_ui_state(self):
        """Update all microphone-related UI elements to match current state"""
        if self.client.microphone_enabled:
            self.microphone_menu.actions()[0].setText("Disable")
            if hasattr(self, 'controls_overlay'):
                self.controls_overlay.mic_btn.is_active = True
                self.controls_overlay.mic_btn.init_style()
                self.controls_overlay.mic_btn.setText("🎤")
                self.controls_overlay.mic_btn.setToolTip("Microphone On - Click to turn off")
        else:
            self.microphone_menu.actions()[0].setText("Enable")
            if hasattr(self, 'controls_overlay'):
                self.controls_overlay.mic_btn.is_active = False
                self.controls_overlay.mic_btn.init_style()
                self.controls_overlay.mic_btn.setText("🔇")
                self.controls_overlay.mic_btn.setToolTip("Microphone Off - Click to turn on")

    def toggle_camera(self):
        """Toggle camera state and update all UI elements"""
        self.client.camera_enabled = not self.client.camera_enabled
        self.update_camera_ui_state()

    def toggle_microphone(self):
        """Toggle microphone state and update all UI elements"""
        self.client.microphone_enabled = not self.client.microphone_enabled
        self.update_microphone_ui_state()
    
    def toggle_gesture_control(self):
        """Handle gesture control button click with proper mode selection and cleanup"""
        from PyQt6.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QLabel, QRadioButton, QDialogButtonBox
        from PyQt6.QtCore import QTimer
        
        if not hasattr(self, 'gesture_controller') or self.gesture_controller is None:
            try:
                dialog = QDialog(self)
                dialog.setWindowTitle("Gesture Control Mode")
                dialog.setFixedSize(450, 220)
                
                layout = QVBoxLayout()
                layout.setSpacing(12)
                
                label = QLabel("Choose gesture control mode:")
                label.setStyleSheet(f"font-size: 14px; font-weight: 600; color: {COLORS['text']};")
                layout.addWidget(label)
                
                preview_mode = QRadioButton("Preview Only (local display changes only)")
                preview_mode.setChecked(True)
                preview_mode.setStyleSheet(f"font-size: 13px; color: {COLORS['text']}; padding: 8px;")
                
                camera_mode = QRadioButton("Camera Control (affects transmission to all clients)")
                camera_mode.setStyleSheet(f"font-size: 13px; color: {COLORS['text']}; padding: 8px;")
                
                layout.addWidget(preview_mode)
                layout.addWidget(camera_mode)
                layout.addSpacing(10)
                
                button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
                button_box.accepted.connect(dialog.accept)
                button_box.rejected.connect(dialog.reject)
                layout.addWidget(button_box)
                
                dialog.setLayout(layout)
                dialog.setStyleSheet(f"""
                    QDialog {{
                        background-color: {COLORS['surface']};
                    }}
                    QRadioButton::indicator {{
                        width: 18px;
                        height: 18px;
                    }}
                    QRadioButton::indicator:checked {{
                        background-color: {COLORS['primary']};
                        border: 2px solid {COLORS['primary']};
                        border-radius: 9px;
                    }}
                """)
                
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                
                control_transmission = camera_mode.isChecked()
                
                from gesture_control import integrate_gesture_control
                self.gesture_controller = integrate_gesture_control(self, control_transmission)
                if self.gesture_controller is None:
                    return

                if self.client and getattr(self.client, "camera", None):
                    self.client.camera.set_gesture_controller(self.gesture_controller)

                try:
                    self.chat_widget.gesture_button.setText("Starting...")
                    self.chat_widget.gesture_button.button_type = "secondary"
                    self.chat_widget.gesture_button.init_style()
                except Exception:
                    pass

                def _start_controller():
                    try:
                        self.gesture_controller.start()
                        try:
                            mode_text = "Camera Control" if control_transmission else "Preview Only"
                            self.chat_widget.gesture_button.setText(f"Stop Gesture ({mode_text})")
                            self.chat_widget.gesture_button.button_type = "danger"
                            self.chat_widget.gesture_button.init_style()
                        except Exception:
                            pass
                    except Exception as e:
                        try:
                            QMessageBox.critical(self, "Error", f"Failed to start gesture control: {str(e)}")
                        except Exception:
                            print(f"Failed to start gesture control: {e}")

                        try:
                            if self.client and getattr(self.client, "camera", None):
                                self.client.camera.set_gesture_controller(None)
                        except Exception:
                            pass
                        self.gesture_controller = None
                        
                        try:
                            self.chat_widget.gesture_button.setText("Gesture Control")
                            self.chat_widget.gesture_button.button_type = "warning"
                            self.chat_widget.gesture_button.init_style()
                        except Exception:
                            pass

                QTimer.singleShot(250, _start_controller)

            except Exception as e:
                try:
                    QMessageBox.critical(self, "Error", f"Failed to initialize gesture control: {str(e)}")
                except Exception:
                    print(f"Failed to initialize gesture control: {e}")
                
                try:
                    if getattr(self, "gesture_controller", None):
                        if self.client and getattr(self.client, "camera", None):
                            self.client.camera.set_gesture_controller(None)
                        self.gesture_controller = None
                except Exception:
                    pass

        else:
            try:
                try:
                    self.chat_widget.gesture_button.setText("Stopping...")
                    self.chat_widget.gesture_button.button_type = "secondary"
                    self.chat_widget.gesture_button.init_style()
                except Exception:
                    pass
                
                try:
                    if self.client and getattr(self.client, "camera", None):
                        self.client.camera.set_gesture_controller(None)
                except Exception as e:
                    print(f"Error disconnecting camera from gesture controller: {e}")
                
                try:
                    if hasattr(self.gesture_controller, "stop_gesture_control"):
                        self.gesture_controller.stop_gesture_control()
                    elif hasattr(self.gesture_controller, "stop"):
                        self.gesture_controller.running = False
                        self.gesture_controller.wait(3000)
                except Exception as e:
                    print(f"Error while stopping gesture controller: {e}")

                try:
                    if hasattr(self.gesture_controller, "isRunning") and self.gesture_controller.isRunning():
                        self.gesture_controller.wait(3000)
                except Exception as e:
                    print(f"Error waiting for gesture thread: {e}")

                self.gesture_controller = None

                try:
                    self.chat_widget.gesture_button.setText("Gesture Control")
                    self.chat_widget.gesture_button.button_type = "warning"
                    self.chat_widget.gesture_button.init_style()
                except Exception:
                    pass
                    
            except Exception as e:
                print(f"Error during gesture control cleanup: {e}")
                self.gesture_controller = None
                try:
                    self.chat_widget.gesture_button.setText("Gesture Control")
                    self.chat_widget.gesture_button.button_type = "warning"
                    self.chat_widget.gesture_button.init_style()
                except Exception:
                    pass

    def resizeEvent(self, event):
        """Handle window resize to update overlay position"""
        super().resizeEvent(event)
        if hasattr(self, 'controls_overlay'):
            self.controls_overlay.update_position(self.geometry())

    def closeEvent(self, event):
        """Handle window close event to properly cleanup gesture control"""
        if hasattr(self, 'gesture_controller') and self.gesture_controller:
            self.gesture_controller.stop_gesture_control()
        if hasattr(self, 'controls_overlay'):
            self.controls_overlay.close()
        event.accept()