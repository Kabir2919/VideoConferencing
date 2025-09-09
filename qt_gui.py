import os
import cv2
import pyaudio
from PyQt6.QtCore import Qt, QThread, QTimer, QSize, QRunnable, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QActionGroup, QIcon, QFont, QPalette, QColor
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout, QDockWidget \
    , QLabel, QWidget, QListWidget, QListWidgetItem, QMessageBox \
    , QComboBox, QTextEdit, QLineEdit, QPushButton, QFileDialog \
    , QDialog, QMenu, QWidgetAction, QCheckBox, QFrame, QSizePolicy

from constants import *

# Camera
CAMERA_RES = '1080p'   # Changed from '240p' → better clarity
LAYOUT_RES = '900p'
frame_size = {
    '240p': (352, 240),
    '360p': (480, 360),
    '480p': (640, 480),
    '560p': (800, 560),
    '720p': (1280, 720),   # corrected 720p resolution
    '900p': (1400, 900),
    '1080p': (1920, 1080)
}

FRAME_WIDTH = frame_size[CAMERA_RES][0]
FRAME_HEIGHT = frame_size[CAMERA_RES][1]

# Image Encoding
ENABLE_ENCODE = False
ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 95]  # sharper frames


# Replace the image loading section in qt_gui.py with this:

import numpy as np

# Create default frames if image files don't exist
def create_default_frame(width, height, text, color=(64, 64, 64)):
    """Create a default frame with text if image files are missing"""
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    
    # Add text to frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_color = (255, 255, 255)
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Center the text
    x = (width - text_width) // 2
    y = (height + text_height) // 2
    
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)
    return frame

# Try to load images, create defaults if they don't exist
try:
    NOCAM_FRAME = cv2.imread("C:/Users/Kabir/Desktop/Video/img/nocam.jpeg")
    if NOCAM_FRAME is None:
        raise FileNotFoundError("nocam.jpeg not found")
    
    # Crop center part of the nocam frame
    nocam_h, nocam_w = NOCAM_FRAME.shape[:2]
    x, y = (nocam_w - FRAME_WIDTH)//2, (nocam_h - FRAME_HEIGHT)//2
    NOCAM_FRAME = NOCAM_FRAME[y:y+FRAME_HEIGHT, x:x+FRAME_WIDTH]
    
except (FileNotFoundError, AttributeError):
    print("[WARNING] nocam.jpeg not found, using default frame")
    NOCAM_FRAME = create_default_frame(FRAME_WIDTH, FRAME_HEIGHT, "Camera Off")

try:
    NOMIC_FRAME = cv2.imread("C:/Users/Kabir/Desktop/Video/img/nomic.jpeg")
    if NOMIC_FRAME is None:
        raise FileNotFoundError("nomic.jpeg not found")
except (FileNotFoundError, AttributeError):
    print("[WARNING] nomic.jpeg not found, using default frame")
    NOMIC_FRAME = create_default_frame(100, 50, "Mic Off", (128, 0, 0))
# Audio
ENABLE_AUDIO = True
SAMPLE_RATE = 48000
BLOCK_SIZE = 2048
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
        # Store constructor arguments (re-used for processing)
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
        # if this is the current client, then don't play audio
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
        
        # Try to initialize camera
        try:
            self.cap = cv2.VideoCapture(2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            
            if self.cap.isOpened():
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
    
    def get_frame(self):
        if not self.camera_available or self.cap is None:
            return None
            
        try:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, frame_size[CAMERA_RES], interpolation=cv2.INTER_AREA)
                
                # Apply gesture detection overlay if gesture controller is active
                if (self.gesture_controller is not None and 
                    hasattr(self.gesture_controller, 'draw_detection_boxes') and
                    self.gesture_controller.running):
                    frame = self.gesture_controller.draw_detection_boxes(frame)
                
                if ENABLE_ENCODE:
                    _, frame = cv2.imencode('.jpg', frame, ENCODE_PARAM)
                return frame
            else:
                return None
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
                transform: scale(1.05);
            }}
            QPushButton:pressed {{
                background-color: {bg_hover};
                transform: scale(0.95);
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
        
        # Make the overlay semi-transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Layout for control buttons
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(20)
        
        # Camera control button
        self.camera_btn = VideoControlButton(
            icon_text="📹", 
            button_type="camera", 
            size=60
        )
        self.camera_btn.setText("📹")
        self.camera_btn.setToolTip("Toggle Camera")
        self.camera_btn.clicked.connect(self.toggle_camera)
        
        # Microphone control button
        self.mic_btn = VideoControlButton(
            icon_text="🎤", 
            button_type="microphone", 
            size=60
        )
        self.mic_btn.setText("🎤")
        self.mic_btn.setToolTip("Toggle Microphone")
        self.mic_btn.clicked.connect(self.toggle_microphone)
        
        # End call button
        self.end_call_btn = VideoControlButton(
            icon_text="📞", 
            button_type="danger", 
            size=60
        )
        self.end_call_btn.setText("📞")
        self.end_call_btn.setToolTip("End Call")
        self.end_call_btn.is_active = False  # Always red
        self.end_call_btn.init_style()
        self.end_call_btn.clicked.connect(self.main_window.close)
        
        # Settings/More button
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
        
        # Make sure the overlay stays on top and is positioned correctly
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
        
        # Layout options
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
        
        # Gesture control toggle
        if hasattr(self.main_window, 'gesture_controller') and self.main_window.gesture_controller:
            gesture_action = menu.addAction("Stop Gesture Control")
            gesture_action.triggered.connect(self.main_window.toggle_gesture_control)
        else:
            gesture_action = menu.addAction("Start Gesture Control")
            gesture_action.triggered.connect(self.main_window.toggle_gesture_control)
        
        menu.addSeparator()
        
        # Show/Hide chat
        chat_action = menu.addAction("Toggle Chat Panel")
        chat_action.triggered.connect(self.toggle_chat_panel)
        
        # Show menu at button position
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
        y = parent_geometry.bottom() - overlay_height - 30  # 30px from bottom
        
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
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
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
        frame = self.client.get_video()
        if frame is None:
            frame = NOCAM_FRAME.copy()
        elif ENABLE_ENCODE:
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        
        if self.client.audio_data is None:
            # replace bottom center part of the frame with nomic frame
            nomic_h, nomic_w, _ = NOMIC_FRAME.shape
            x, y = FRAME_WIDTH//2 - nomic_w//2, FRAME_HEIGHT - 50
            frame[y:y+nomic_h, x:x+nomic_w] = NOMIC_FRAME.copy()

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_viewer.setPixmap(QPixmap.fromImage(q_img))


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
        else:  # secondary
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

        # Chat messages area
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

        # Client selection menu
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

        self.select_all_checkbox, _ = self.add_client("")  # Select All Checkbox
        self.clients_menu.addSeparator()

        self.clients_button = ModernButton("Select Recipients", "secondary", self)
        self.clients_button.setMenu(self.clients_menu)
        self.layout.addWidget(self.clients_button)

        # Control buttons section
        buttons_frame = QFrame()
        buttons_layout = QHBoxLayout(buttons_frame)
        buttons_layout.setSpacing(12)
        
        self.file_button = ModernButton("Send File", "secondary", self)
        self.gesture_button = ModernButton("Gesture Control", "warning", self)
        
        buttons_layout.addWidget(self.file_button)
        buttons_layout.addWidget(self.gesture_button)
        
        self.layout.addWidget(buttons_frame)

        # Message input section
        message_frame = QFrame()
        message_layout = QVBoxLayout(message_frame)
        message_layout.setSpacing(8)
        
        self.line_edit = QLineEdit(self)
        self.line_edit.setPlaceholderText("Type your message here...")
        self.line_edit.setStyleSheet(f"""
            QLineEdit {{
                padding: 6px 10px;   /* smaller padding */
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                font-size: 14px;
                background-color: {COLORS['surface']};
                min-height: 32px;   /* ensures full text visibility */
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

        # Spacing
        self.layout.addSpacing(20)

        # End call button
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

        if name == "":  # Select All Checkbox
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
        formatted_msg = f"<div style='margin: 8px 0; padding: 8px; background-color: {COLORS['surface_hover']}; border-radius: 6px;'>"
        formatted_msg += f"<span style='font-weight: 600; color: {COLORS['primary']};'>{from_name}</span>"
        formatted_msg += f" <span style='color: {COLORS['text_muted']}; font-size: 12px;'>→ {to_name}</span><br>"
        formatted_msg += f"<span style='color: {COLORS['text']};'>{msg}</span></div>"
        self.central_widget.insertHtml(formatted_msg)
        
        #
        # Add this to the end of your qt_gui.py file to complete it

        # Auto-scroll to bottom
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

        # Title
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

        # Username input
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

        # Join button
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

        # Video area
        self.video_list_widget = VideoListWidget()
        self.setCentralWidget(self.video_list_widget)

        # Create chat widget BEFORE using it
        self.chat_widget = ChatWidget()

        # Chat sidebar
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
        
        # Connect chat widget signals
        self.chat_widget.send_button.clicked.connect(lambda: self.send_msg(TEXT))
        self.chat_widget.line_edit.returnPressed.connect(lambda: self.send_msg(TEXT))
        self.chat_widget.file_button.clicked.connect(lambda: self.send_msg(FILE))
        self.chat_widget.gesture_button.clicked.connect(self.toggle_gesture_control)
        self.chat_widget.end_button.clicked.connect(self.close)

        # Modern menu bar
        self.setup_menu_bar()
        
        # Initialize video controls overlay
        self.controls_overlay = VideoControlsOverlay(self)
        self.controls_overlay.show()
        # Initialize UI states to match client states
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

        # Camera menu
        self.camera_menu = menubar.addMenu("📹 Camera")
        self.camera_menu.addAction("Disable", self.toggle_camera)
        
        # Microphone menu
        self.microphone_menu = menubar.addMenu("🎤 Microphone")
        self.microphone_menu.addAction("Disable", self.toggle_microphone)
        
        # Layout menu
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
    
    # def toggle_camera(self):
    #     if self.client.camera_enabled:
    #         self.camera_menu.actions()[0].setText("Enable")
    #     else:
    #         self.camera_menu.actions()[0].setText("Disable")
    #     self.client.camera_enabled = not self.client.camera_enabled

    # def toggle_microphone(self):
    #     if self.client.microphone_enabled:
    #         self.microphone_menu.actions()[0].setText("Enable")
    #     else:
    #         self.microphone_menu.actions()[0].setText("Disable")
    #     self.client.microphone_enabled = not self.client.microphone_enabled
    def update_camera_ui_state(self):
        """Update all camera-related UI elements to match current state"""
        if self.client.camera_enabled:
            # Menu
            self.camera_menu.actions()[0].setText("Disable")
            # Video controls overlay
            if hasattr(self, 'controls_overlay'):
                self.controls_overlay.camera_btn.is_active = True
                self.controls_overlay.camera_btn.init_style()
                self.controls_overlay.camera_btn.setText("📹")
                self.controls_overlay.camera_btn.setToolTip("Camera On - Click to turn off")
        else:
            # Menu
            self.camera_menu.actions()[0].setText("Enable")
            # Video controls overlay
            if hasattr(self, 'controls_overlay'):
                self.controls_overlay.camera_btn.is_active = False
                self.controls_overlay.camera_btn.init_style()
                self.controls_overlay.camera_btn.setText("📹")
                self.controls_overlay.camera_btn.setToolTip("Camera Off - Click to turn on")

    def update_microphone_ui_state(self):
        """Update all microphone-related UI elements to match current state"""
        if self.client.microphone_enabled:
            # Menu
            self.microphone_menu.actions()[0].setText("Disable")
            # Video controls overlay
            if hasattr(self, 'controls_overlay'):
                self.controls_overlay.mic_btn.is_active = True
                self.controls_overlay.mic_btn.init_style()
                self.controls_overlay.mic_btn.setText("🎤")
                self.controls_overlay.mic_btn.setToolTip("Microphone On - Click to turn off")
        else:
            # Menu
            self.microphone_menu.actions()[0].setText("Enable")
            # Video controls overlay
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
        """Handle gesture control button click"""
        if not hasattr(self, 'gesture_controller'):
            # Import and initialize gesture controller
            try:
                from gesture_control import integrate_gesture_control
                self.gesture_controller = integrate_gesture_control(self)
                if self.gesture_controller is None:
                    return  # MediaPipe not available
                    
                # Connect gesture controller to camera for overlay drawing
                if self.client.camera:
                    self.client.camera.set_gesture_controller(self.gesture_controller)
                    
                self.gesture_controller.start()
                self.chat_widget.gesture_button.setText("Stop Gesture Control")
                self.chat_widget.gesture_button.button_type = "danger"
                self.chat_widget.gesture_button.init_style()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start gesture control: {str(e)}")
        else:
            # Stop gesture control
            self.gesture_controller.stop_gesture_control()
            
            # Disconnect from camera
            if self.client.camera:
                self.client.camera.set_gesture_controller(None)
                
            self.gesture_controller = None
            self.chat_widget.gesture_button.setText("Gesture Control")
            self.chat_widget.gesture_button.button_type = "warning"
            self.chat_widget.gesture_button.init_style()

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