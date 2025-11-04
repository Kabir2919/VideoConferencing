
import time
import sys
import socket
import pickle
from collections import defaultdict
import os
from PyQt6.QtCore import QThreadPool, QRunnable, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication, QMessageBox
from qt_gui import MainWindow, Camera, Microphone, Worker

from constants import *

IP = socket.gethostbyname(socket.gethostname())
# IP = "172.20.10.4"  # Uncomment and set manually if needed
VIDEO_ADDR = (IP, VIDEO_PORT)
AUDIO_ADDR = (IP, AUDIO_PORT)


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

    def get_video(self):
        """Get video frame for transmission or display"""
        if not self.camera_enabled:
            self.video_frame = None
            return None

        if self.camera is not None:
            self.video_frame = self.camera.get_frame(apply_overlays=False)

        return self.video_frame
    
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
            
        self.start_conn_threads() # Start receiving threads for all servers
        self.start_broadcast_threads() # Start sending threads for audio and video

        self.add_client_signal.emit(client)

        while self.connected:
            time.sleep(0.1)  # Prevent busy waiting
            
        self.disconnect_server()

    def init_conn(self):
        try:
            print(f"[DEBUG] Attempting to connect to {IP}:{MAIN_PORT}")
            
            # Set socket timeouts
            self.main_socket.settimeout(10.0)
            self.video_socket.settimeout(5.0)
            self.audio_socket.settimeout(5.0)
            
            # Connect to main server
            self.main_socket.connect((IP, MAIN_PORT))
            print(f"[DEBUG] Connected to main server successfully")

            # Send name and get response
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
            
            # Bind UDP sockets to receive data
            try:
                # Bind to any available port for receiving
                self.video_socket.bind(('', 0))  # Let OS choose port
                self.audio_socket.bind(('', 0))  # Let OS choose port
                print(f"[DEBUG] UDP sockets bound successfully")
                print(f"[DEBUG] Video socket bound to: {self.video_socket.getsockname()}")
                print(f"[DEBUG] Audio socket bound to: {self.audio_socket.getsockname()}")
            except Exception as e:
                print(f"[ERROR] Failed to bind UDP sockets: {e}")
                return False
            
            # Keep timeout for registration phase
            self.video_socket.settimeout(5.0)
            self.audio_socket.settimeout(5.0)
            
            # Register with media servers with retry
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
            
            # Remove socket timeouts for normal operation
            self.main_socket.settimeout(None)
            self.video_socket.settimeout(None)
            self.audio_socket.settimeout(None)
            
            self.connected = True
            print(f"[DEBUG] Connection initialization complete")
            
            # Small delay to ensure everything is set up
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
            if self.connected:
                self.send_msg(self.main_socket, Message(self.name, DISCONNECT))
        except:
            pass
        finally:
            self.main_socket.close()
            self.video_socket.close()
            self.audio_socket.close()
    
    def send_msg(self, conn: socket.socket, msg: Message):
        if not self.connected and msg.request != ADD:
            return
            
        try:
            msg_bytes = pickle.dumps(msg)
            
            # FIXED: Check packet size before sending
            if msg.data_type in [VIDEO, AUDIO]:
                max_size = MEDIA_SIZE[msg.data_type]
                if len(msg_bytes) > max_size:
                    # print(f"[ERROR] {msg.data_type} packet too large: {len(msg_bytes)} > {max_size}")
                    return
                conn.sendto(msg_bytes, (IP, VIDEO_PORT if msg.data_type == VIDEO else AUDIO_PORT))
            else:
                conn.send_bytes(msg_bytes)
                
        except (BrokenPipeError, ConnectionResetError, OSError, socket.error) as e:
            print(f"[ERROR] Send failed: {e}")
            if self.connected:  # Only set to False if we were previously connected
                self.connected = False
    
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
                # Send end-of-file marker
                msg = Message(self.name, POST, FILE, None, to_names)
                self.send_msg(self.main_socket, msg)
            self.add_msg_signal.emit(self.name, f"File {filename} sent.")
        except Exception as e:
            self.add_msg_signal.emit("System", f"Error sending file: {e}")
    
    def media_broadcast_loop(self, conn: socket.socket, media: str):
        consecutive_errors = 0
        max_errors = 5
        
        while self.connected:
            try:
                if media == VIDEO:
                    data = client.get_video()
                    # FIXED: Skip if no data or if data is too large
                    if data is None:
                        time.sleep(0.033)  # ~30 FPS
                        continue
                        
                    # Check data size before sending
                    msg = Message(self.name, POST, media, data)
                    msg_bytes = pickle.dumps(msg)
                    if len(msg_bytes) > MEDIA_SIZE[media]:
                        print(f"[WARNING] {media} packet too large: {len(msg_bytes)} > {MEDIA_SIZE[media]}")
                        time.sleep(0.033)
                        continue
                        
                elif media == AUDIO:
                    data = client.get_audio()
                    if data is None:
                        time.sleep(0.023)  # ~43 FPS for audio
                        continue
                        
                    # Check data size before sending
                    msg = Message(self.name, POST, media, data)
                    msg_bytes = pickle.dumps(msg)
                    if len(msg_bytes) > MEDIA_SIZE[media]:
                        print(f"[WARNING] {media} packet too large: {len(msg_bytes)} > {MEDIA_SIZE[media]}")
                        time.sleep(0.023)
                        continue
                else:
                    print(f"[ERROR] Invalid media type: {media}")
                    break
                    
                self.send_msg(conn, msg)
                consecutive_errors = 0  # Reset error counter on success
                
                # Add appropriate delays
                if media == VIDEO:
                    time.sleep(0.033)  # ~30 FPS
                else:  # AUDIO
                    time.sleep(0.023)  # ~43 FPS
                    
            except Exception as e:
                consecutive_errors += 1
                print(f"[ERROR] Media broadcast error ({media}): {e}")
                
                if consecutive_errors >= max_errors:
                    print(f"[ERROR] Too many consecutive errors in {media} broadcast, stopping")
                    break
                    
                time.sleep(0.1)  # Brief pause before retrying

    def handle_conn(self, conn: socket.socket, media: str):
        consecutive_errors = 0
        max_errors = 10
        
        while self.connected:
            try:
                if media in [VIDEO, AUDIO]:
                    msg_bytes, addr = conn.recvfrom(MEDIA_SIZE[media])
                else:
                    msg_bytes = conn.recv_bytes()
                    
                if not msg_bytes:
                    print(f"[{media}] Empty message received, connection may be closed")
                    break
                    
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
                consecutive_errors = 0  # Reset on successful message handling
                
            except socket.timeout:
                continue  # Timeout is normal, just continue
            except (ConnectionResetError, OSError, socket.error) as e:
                print(f"[{self.name}] [{media}] [ERROR] Connection error: {e}")
                break
            except Exception as e:
                consecutive_errors += 1
                print(f"[{self.name}] [{media}] [ERROR] {e}")
                if consecutive_errors >= max_errors:
                    print(f"[{media}] Too many consecutive errors, stopping handler")
                    break
                time.sleep(0.1)
        
        print(f"[{media}] Handler thread exiting")
        if media == TEXT:  # Main connection handler
            self.connected = False

    def handle_msg(self, msg: Message):
        global all_clients
        client_name = msg.from_name
        
        if msg.request == POST:
            if client_name not in all_clients:
                print(f"[{self.name}] [ERROR] Invalid client name {client_name}: {msg}")
                return
                
            if msg.data_type == VIDEO:
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
            # Start of file transfer - filename
            if os.path.exists(msg.data):
                filename, ext = os.path.splitext(msg.data)
                i = 1
                while os.path.exists(f"{filename}({i}){ext}"):
                    i += 1
                msg.data = f"{filename}({i}){ext}"
            self.recieving_filename = msg.data
            with open(self.recieving_filename, 'wb') as f:
                pass  # Create empty file
        elif msg.data is None:
            # End of file transfer
            if self.recieving_filename:
                self.add_msg_signal.emit(client_name, f"File {self.recieving_filename} received.")
                self.recieving_filename = None
        else:
            # File data chunk
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