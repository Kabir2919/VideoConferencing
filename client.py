# Add this at the top of client.py, right after the other imports and before IP definition

import errno, socket
import time
import sys
import pickle
from collections import defaultdict
import os
from PyQt6.QtCore import QThreadPool, QRunnable, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication, QMessageBox
from qt_gui import MainWindow, Camera, Microphone, Worker
import traceback
from datetime import datetime

from constants import *

IP = socket.gethostbyname(socket.gethostname())
IP = "192.168.220.55"  # Uncomment and set manually if needed
VIDEO_ADDR = (IP, VIDEO_PORT)
AUDIO_ADDR = (IP, AUDIO_PORT)

# Special marker for camera off state - MUST be defined before Client class
CAMERA_OFF_MARKER = b'CAMERA_OFF_MARKER_V1'

# CRITICAL: Make sure this import is at the module level
# This ensures the marker is available to all functions
import errno

class Client:
    def __init__(self, name: str, current_device = False):
        self.name = name
        self.current_device = current_device

        self.video_frame = None
        self.audio_data = None

        if self.current_device:
            self.camera = Camera()
            self.microphone = Microphone()
        else:
            self.camera = None
            self.microphone = None
        
        self.camera_enabled = True
        self.microphone_enabled = True

    # Replace the get_video method in the Client class (client.py)

    def get_video(self):
        """
        Get video frame for transmission or display.
        FIXED: Thread-safe with proper state handling to prevent disconnections.
        """
        # Check camera enabled state - this is thread-safe read
        if not self.camera_enabled:
            # Camera is disabled - send marker
            self.video_frame = CAMERA_OFF_MARKER
            return CAMERA_OFF_MARKER

        if self.camera is not None:
            try:
                # Get frame from camera hardware
                frame = self.camera.get_frame(apply_overlays=False)
                
                # If camera returns None (hardware issue), send camera off marker
                if frame is None:
                    self.video_frame = CAMERA_OFF_MARKER
                    return CAMERA_OFF_MARKER
                
                # Valid frame received
                self.video_frame = frame
                return frame
                
            except Exception as e:
                # Error getting frame - return camera off marker to maintain connection
                print(f"[CLIENT] Error in get_video: {e}")
                self.video_frame = CAMERA_OFF_MARKER
                return CAMERA_OFF_MARKER
        else:
            # No camera available
            self.video_frame = CAMERA_OFF_MARKER
            return CAMERA_OFF_MARKER
    
    def get_audio(self):
        if not self.microphone_enabled:
            self.audio_data = None
            return None

        if self.microphone is not None:
            self.audio_data = self.microphone.get_data()

        return self.audio_data


class ServerConnection(QThread):
    add_client_signal = pyqtSignal(Client)
    remove_client_signal = pyqtSignal(str)
    add_msg_signal = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.threadpool = QThreadPool()

        self.main_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.audio_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.connected = False
        self.recieving_filename = None

    def run(self):
        if not self.init_conn(): 
            return
            
        self.start_conn_threads()
        self.start_broadcast_threads()

        self.add_client_signal.emit(client)

        while self.connected:
            time.sleep(0.1)
            
        self.disconnect_server()

    def init_conn(self):
        try:
            print(f"[DEBUG] Attempting to connect to {IP}:{MAIN_PORT}")
            
            self.main_socket.settimeout(10.0)
            self.video_socket.settimeout(5.0)
            self.audio_socket.settimeout(5.0)
            
            self.main_socket.connect((IP, MAIN_PORT))
            print(f"[DEBUG] Connected to main server successfully")

            client.name = self.name
            self.main_socket.send_bytes(self.name.encode())
            conn_status = self.main_socket.recv_bytes().decode()
            print(f"[DEBUG] Server response: {conn_status}")
            
            if conn_status != OK:
                print(f"[ERROR] Server rejected connection: {conn_status}")
                QMessageBox.critical(None, "Error", conn_status)
                self.main_socket.close()
                if hasattr(self, 'window'):
                    window.close()
                return False
            
            try:
                self.video_socket.bind(('', 0))
                self.audio_socket.bind(('', 0))
                print(f"[DEBUG] UDP sockets bound successfully")
            except Exception as e:
                print(f"[ERROR] Failed to bind UDP sockets: {e}")
                return False
            
            self.video_socket.settimeout(5.0)
            self.audio_socket.settimeout(5.0)
            
            print(f"[DEBUG] Sending video/audio registration messages")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.send_msg(self.video_socket, Message(self.name, ADD, VIDEO))
                    self.send_msg(self.audio_socket, Message(self.name, ADD, AUDIO))
                    print(f"[DEBUG] Media registration attempt {attempt + 1} completed")
                    break
                except Exception as e:
                    print(f"[DEBUG] Media registration attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(0.5)
            
            self.main_socket.settimeout(None)
            self.video_socket.settimeout(None)
            self.audio_socket.settimeout(None)
            
            self.connected = True
            print(f"[DEBUG] Connection initialization complete")
            
            time.sleep(0.5)
            return True
            
        except socket.timeout:
            print(f"[ERROR] Connection timeout")
            self.connected = False
            return False
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            import traceback
            traceback.print_exc()
            self.connected = False
            return False
    
    def start_conn_threads(self):
        self.main_conn_thread = Worker(self.handle_conn, self.main_socket, TEXT)
        self.threadpool.start(self.main_conn_thread)

        self.video_conn_thread = Worker(self.handle_conn, self.video_socket, VIDEO)
        self.threadpool.start(self.video_conn_thread)

        self.audio_conn_thread = Worker(self.handle_conn, self.audio_socket, AUDIO)
        self.threadpool.start(self.audio_conn_thread)

    def start_broadcast_threads(self):
        self.video_broadcast_thread = Worker(self.media_broadcast_loop, self.video_socket, VIDEO)
        self.threadpool.start(self.video_broadcast_thread)

        self.audio_broadcast_thread = Worker(self.media_broadcast_loop, self.audio_socket, AUDIO)
        self.threadpool.start(self.audio_broadcast_thread)
    
    def disconnect_server(self):
        try:
            self.connected = False
        except:
            pass

        time.sleep(0.2)

        try:
            if getattr(self, 'main_socket', None):
                try:
                    self.send_msg(self.main_socket, Message(self.name, DISCONNECT))
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if getattr(self, 'main_socket', None):
                self.main_socket.close()
        except Exception:
            pass

        try:
            if getattr(self, 'video_socket', None):
                self.video_socket.close()
        except Exception:
            pass

        try:
            if getattr(self, 'audio_socket', None):
                self.audio_socket.close()
        except Exception:
            pass

    def send_msg(self, conn: socket.socket, msg: Message):
        if not self.connected and msg.request != ADD:
            return
        try:
            msg_bytes = pickle.dumps(msg)
            if msg.data_type in [VIDEO, AUDIO]:
                max_size = MEDIA_SIZE[msg.data_type]
                if len(msg_bytes) > max_size:
                    print(f"[WARNING] {msg.data_type} packet too large: {len(msg_bytes)} bytes - SKIPPING")
                    return
                conn.sendto(msg_bytes, (IP, VIDEO_PORT if msg.data_type == VIDEO else AUDIO_PORT))
            else:
                conn.send_bytes(msg_bytes)
        except OSError as e:
            winerr = getattr(e, 'winerror', None)
            errnum = getattr(e, 'errno', None)
            is_udp = getattr(conn, 'type', None) == socket.SOCK_DGRAM

            if is_udp and (winerr == 10038 or errnum in (errno.EBADF, errno.ENOTCONN)):
                return

            if not is_udp and (winerr == 10038 or errnum in (errno.EBADF, errno.ENOTCONN, errno.ECONNRESET)):
                self.connected = False
                return

            print(f"[ERROR] Send failed: {e}")
        except Exception as e:
            print(f"[ERROR] Send failed: {e}")
    
    def send_file(self, filepath: str, to_names: tuple[str]):
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'rb') as f:
                while True:
                    data = f.read(SIZE)
                    if not data:
                        break
                    msg = Message(self.name, POST, FILE, data, to_names)
                    self.send_msg(self.main_socket, msg)
                msg = Message(self.name, POST, FILE, None, to_names)
                self.send_msg(self.main_socket, msg)
            self.add_msg_signal.emit(self.name, f"File {filename} sent.")
        except Exception as e:
            self.add_msg_signal.emit("System", f"Error sending file: {e}")
    
    # Replace the media_broadcast_loop method in ServerConnection class (client.py)

    def media_broadcast_loop(self, conn: socket.socket, media: str):
        """
        FIXED: Ultra-stable broadcast loop that never causes disconnections.
        Key improvements:
        1. Never stops sending packets (always sends CAMERA_OFF_MARKER if needed)
        2. Thread-safe state reading
        3. Graceful error handling
        4. No blocking operations
        """
        consecutive_errors = 0
        max_errors = 10  # Increased tolerance
        last_camera_state = None
        
        # Wait for connection to stabilize
        time.sleep(0.3)
        
        print(f"[{media}_BROADCAST] Starting broadcast loop")
        
        while self.connected:
            try:
                if media == VIDEO:
                    # Read camera state (thread-safe)
                    current_camera_state = client.camera_enabled
                    
                    # Log state changes for debugging
                    if last_camera_state is not None and last_camera_state != current_camera_state:
                        print(f"[VIDEO_BROADCAST] Camera state changed: {last_camera_state} -> {current_camera_state}")
                        # Brief pause during state transition for stability
                        time.sleep(0.05)
                    
                    last_camera_state = current_camera_state
                    
                    # CRITICAL: Always get data (either frame or marker)
                    try:
                        data = client.get_video()
                    except Exception as e:
                        print(f"[VIDEO_BROADCAST] Error getting video: {e}")
                        data = CAMERA_OFF_MARKER
                    
                    # CRITICAL: Never send None - always send marker as fallback
                    if data is None:
                        data = CAMERA_OFF_MARKER
                    
                    # Build message
                    msg = Message(self.name, POST, media, data)
                    
                    # Serialize message
                    try:
                        msg_bytes = pickle.dumps(msg)
                    except Exception as e:
                        print(f"[VIDEO_BROADCAST] Serialization error: {e}")
                        consecutive_errors += 1
                        time.sleep(0.033)
                        continue
                    
                    # Check size
                    if len(msg_bytes) > MEDIA_SIZE[media]:
                        print(f"[WARNING] {media} packet too large: {len(msg_bytes)} > {MEDIA_SIZE[media]} - sending marker instead")
                        # Send camera off marker instead of dropping packet
                        msg = Message(self.name, POST, media, CAMERA_OFF_MARKER)
                        msg_bytes = pickle.dumps(msg)
                        
                elif media == AUDIO:
                    data = client.get_audio()
                    
                    # Audio can be None (mic off) - just skip this cycle
                    if data is None:
                        time.sleep(0.023)
                        continue
                        
                    msg = Message(self.name, POST, media, data)
                    
                    try:
                        msg_bytes = pickle.dumps(msg)
                    except Exception as e:
                        print(f"[AUDIO_BROADCAST] Serialization error: {e}")
                        consecutive_errors += 1
                        time.sleep(0.023)
                        continue
                    
                    if len(msg_bytes) > MEDIA_SIZE[media]:
                        print(f"[WARNING] {media} packet too large: {len(msg_bytes)} > {MEDIA_SIZE[media]} - SKIPPING")
                        time.sleep(0.023)
                        continue
                else:
                    print(f"[ERROR] Invalid media type: {media}")
                    break
                
                # Send the message
                try:
                    self.send_msg(conn, msg)
                    consecutive_errors = 0  # Reset on success
                except OSError as e:
                    # Check if it's a benign socket error
                    winerr = getattr(e, 'winerror', None)
                    errnum = getattr(e, 'errno', None)
                    
                    if winerr == 10038 or errnum in (errno.EBADF, errno.ENOTCONN):
                        # Socket closed - this is normal during shutdown
                        print(f"[{media}_BROADCAST] Socket closed gracefully")
                        break
                    else:
                        # Real error - count it
                        consecutive_errors += 1
                        print(f"[WARNING] Send error in {media} broadcast ({consecutive_errors}/{max_errors}): {e}")
                        
                        if consecutive_errors >= max_errors:
                            print(f"[ERROR] Too many errors in {media} broadcast")
                            break
                        
                        time.sleep(0.1)
                        continue
                except Exception as e:
                    consecutive_errors += 1
                    print(f"[WARNING] Unexpected error in {media} broadcast ({consecutive_errors}/{max_errors}): {e}")
                    
                    if consecutive_errors >= max_errors:
                        print(f"[ERROR] Too many consecutive errors in {media} broadcast")
                        break
                        
                    time.sleep(0.1)
                    continue
                
                # Timing for frame rate
                if media == VIDEO:
                    time.sleep(0.033)  # ~30 FPS
                else:  # AUDIO
                    time.sleep(0.023)  # ~43 FPS
                    
            except Exception as e:
                consecutive_errors += 1
                print(f"[ERROR] Critical error in {media} broadcast loop ({consecutive_errors}/{max_errors}): {e}")
                import traceback
                traceback.print_exc()
                
                if consecutive_errors >= max_errors:
                    print(f"[FATAL] Too many consecutive errors in {media} broadcast, stopping")
                    break
                    
                time.sleep(0.2)
        
        print(f"[{media}_BROADCAST] Broadcast loop exited cleanly")

    def handle_conn(self, conn: socket.socket, media: str):
        """Handle connection for TEXT (TCP) or VIDEO/AUDIO (UDP)"""
        consecutive_errors = 0
        max_errors = 10
        
        print(f"[{self.name}] [{media}] Handler started")
        
        while self.connected:
            try:
                if media in [VIDEO, AUDIO]:
                    msg_bytes, addr = conn.recvfrom(MEDIA_SIZE[media])
                else:
                    msg_bytes = conn.recv_bytes()
                    
                if not msg_bytes:
                    time.sleep(0.01)
                    continue
                    
                try:
                    msg = pickle.loads(msg_bytes)
                except pickle.UnpicklingError:
                    print(f"[{self.name}] [{media}] [ERROR] UnpicklingError")
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        break
                    continue

                if msg.request == DISCONNECT:
                    print(f"[{media}] Received disconnect message")
                    break
                    
                self.handle_msg(msg)
                consecutive_errors = 0
                
            except socket.timeout:
                continue
            except (ConnectionResetError, OSError, socket.error) as e:
                winerr = getattr(e, 'winerror', None)
                errnum = getattr(e, 'errno', None)
                is_udp = media in [VIDEO, AUDIO]
                
                if is_udp and (winerr == 10038 or errnum in (errno.EBADF, errno.ENOTCONN)):
                    print(f"[{media}] Socket closed (benign); exiting handler.")
                    break
                    
                print(f"[{self.name}] [{media}] [ERROR] Connection error: {e}")
                break
            except Exception as e:
                consecutive_errors += 1
                print(f"[{self.name}] [{media}] [ERROR] {e}")
                if consecutive_errors >= max_errors:
                    print(f"[WARN] too many errors → pausing + retrying")
                    consecutive_errors = 0
                    time.sleep(0.25)
                time.sleep(0.1)
        
        print(f"[{media}] Handler thread exiting")
        if media == TEXT:
            self.connected = False

    def handle_msg(self, msg: Message):
        """Handle received message with camera off marker support"""
        global all_clients
        client_name = msg.from_name
        
        if msg.request == POST:
            if client_name not in all_clients:
                print(f"[{self.name}] [ERROR] Invalid client name {client_name}: {msg}")
                return
                
            if msg.data_type == VIDEO:
                # Check if this is a camera off marker
                if isinstance(msg.data, bytes) and msg.data == CAMERA_OFF_MARKER:
                    all_clients[client_name].video_frame = None
                    print(f"[{self.name}] [{client_name}] Camera OFF marker received")
                elif msg.data is None:
                    # Explicit None means camera off
                    all_clients[client_name].video_frame = None
                    print(f"[{self.name}] [{client_name}] Video frame is None")
                else:
                    # Valid video data
                    all_clients[client_name].video_frame = msg.data
                    
            elif msg.data_type == AUDIO:
                all_clients[client_name].audio_data = msg.data
            elif msg.data_type == TEXT:
                self.add_msg_signal.emit(client_name, msg.data)
            elif msg.data_type == FILE:
                self.handle_file_message(msg, client_name)
            else:
                print(f"[{self.name}] [ERROR] Invalid data type {msg.data_type}")
                
        elif msg.request == ADD:
            if client_name in all_clients:
                print(f"[{self.name}] [ERROR] Client already exists with name {client_name}")
                return
            all_clients[client_name] = Client(client_name)
            self.add_client_signal.emit(all_clients[client_name])
            
        elif msg.request == RM:
            if client_name not in all_clients:
                print(f"[{self.name}] [ERROR] Invalid client name {client_name}")
                return
            self.remove_client_signal.emit(client_name)
            all_clients.pop(client_name)

    def handle_file_message(self, msg, client_name):
        """Handle file transfer messages"""
        if isinstance(msg.data, str):
            if os.path.exists(msg.data):
                filename, ext = os.path.splitext(msg.data)
                i = 1
                while os.path.exists(f"{filename}({i}){ext}"):
                    i += 1
                msg.data = f"{filename}({i}){ext}"
            self.recieving_filename = msg.data
            with open(self.recieving_filename, 'wb') as f:
                pass
        elif msg.data is None:
            if self.recieving_filename:
                self.add_msg_signal.emit(client_name, f"File {self.recieving_filename} received.")
                self.recieving_filename = None
        else:
            if self.recieving_filename:
                with open(self.recieving_filename, 'ab') as f:
                    f.write(msg.data)


client = Client("You", current_device=True)
all_clients = defaultdict(lambda: Client(""))


if __name__ == "__main__":
    app = QApplication(sys.argv)

    server_conn = ServerConnection()
    window = MainWindow(client, server_conn)
    window.show()

    status_code = app.exec()
    server_conn.disconnect_server()
    os._exit(status_code)