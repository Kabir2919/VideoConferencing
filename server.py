import socket
import threading
import time
import os
import traceback
import pickle
from dataclasses import dataclass, field

from constants import *

IP = ''  # Bind to all available interfaces

clients = {} # list of clients connected to the server
video_conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
audio_conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
media_conns = {VIDEO: video_conn, AUDIO: audio_conn}

@dataclass
class Client:
    name: str
    main_conn: socket.socket
    connected: bool
    media_addrs: dict = field(default_factory=lambda: {VIDEO: None, AUDIO: None})

    def send_msg(self, from_name: str, request: str, data_type: str = None, data: any = None):
        if not self.connected:
            return
            
        msg = Message(from_name, request, data_type, data)
        try:
            if data_type in [VIDEO, AUDIO]:
                addr = self.media_addrs.get(data_type, None)
                if addr is None:
                    # Skip if no media address registered yet
                    return
                media_conns[data_type].sendto(pickle.dumps(msg), addr)
            else:
                if self.main_conn and not self.main_conn._closed:
                    self.main_conn.send_bytes(pickle.dumps(msg))
        except (BrokenPipeError, ConnectionResetError, OSError, socket.error) as e:
            print(f"[{self.name}] [ERROR] Send failed: {e}")
            self.connected = False
        except Exception as e:
            print(f"[{self.name}] [ERROR] Unexpected error: {e}")
            self.connected = False


def broadcast_msg(from_name: str, request: str, data_type: str = None, data: any = None):
    """Broadcast message to all connected clients except sender"""
    clients_to_remove = []
    
    for client_name, client in clients.items():
        if client.name == from_name:
            continue
        if not client.connected:
            clients_to_remove.append(client_name)
            continue
        try:
            client.send_msg(from_name, request, data_type, data)
        except Exception as e:
            print(f"[BROADCAST] Error sending to {client_name}: {e}")
            clients_to_remove.append(client_name)
    
    # Clean up disconnected clients
    for client_name in clients_to_remove:
        if client_name in clients:
            print(f"[CLEANUP] Removing disconnected client: {client_name}")
            cleanup_client(clients[client_name])


def multicast_msg(from_name: str, request: str, to_names: tuple[str], data_type: str = None, data: any = None):
    """Send message to specific clients"""
    if not to_names:
        broadcast_msg(from_name, request, data_type, data)
        return
        
    for name in to_names:
        if name not in clients:
            continue
        if not clients[name].connected:
            continue
        try:
            clients[name].send_msg(from_name, request, data_type, data)
        except Exception as e:
            print(f"[MULTICAST] Error sending to {name}: {e}")


def media_server(media: str, port: int):
    """Handle UDP media server (video/audio)"""
    conn = media_conns[media]
    try:
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        conn.bind((IP, port))
        print(f"[LISTENING] {media} Server is listening on {IP if IP else 'all interfaces'}:{port}")
    except Exception as e:
        print(f"[ERROR] Failed to bind {media} server on port {port}: {e}")
        return

    consecutive_errors = 0
    max_errors = 10
    
    while True:
        try:
            conn.settimeout(1.0)  # 1 second timeout for graceful shutdown
            msg_bytes, addr = conn.recvfrom(MEDIA_SIZE[media])
            consecutive_errors = 0  # Reset error counter on success
            
            if not msg_bytes:
                continue
                
            try:
                msg: Message = pickle.loads(msg_bytes)
            except pickle.UnpicklingError:
                print(f"[{addr}] [{media}] [ERROR] UnpicklingError")
                continue

            if msg.request == ADD:
                if msg.from_name not in clients:
                    print(f"[{addr}] [{media}] [ERROR] Client {msg.from_name} not found for ADD request")
                    continue
                    
                client = clients[msg.from_name]
                client.media_addrs[media] = addr
                print(f"[{addr}] [{media}] {msg.from_name} registered")
                
            elif msg.request == POST:
                # Broadcast media data to all other clients
                broadcast_msg(msg.from_name, msg.request, msg.data_type, msg.data)
            else:
                print(f"[{addr}] [{media}] Unexpected request: {msg.request}")
                
        except socket.timeout:
            continue  # Normal timeout, continue loop
        except (ConnectionResetError, OSError) as e:
            consecutive_errors += 1
            if consecutive_errors >= max_errors:
                print(f"[{media}] Too many consecutive errors, restarting media server")
                break
            time.sleep(0.1)
        except Exception as e:
            consecutive_errors += 1
            print(f"[{media}] Unexpected error: {e}")
            if consecutive_errors >= max_errors:
                print(f"[{media}] Too many errors, stopping media server")
                break
            time.sleep(0.1)


def cleanup_client(client: Client):
    """Clean up client resources"""
    try:
        client.media_addrs.update({VIDEO: None, AUDIO: None})
        client.connected = False
        
        if client.main_conn and not client.main_conn._closed:
            try:
                client.main_conn.disconnect()
            except:
                client.main_conn.close()
                
        if client.name in clients:
            clients.pop(client.name)
            
    except Exception as e:
        print(f"[CLEANUP] Error cleaning up {client.name}: {e}")


def disconnect_client(client: Client):
    """Properly disconnect a client"""
    if not client.connected:
        return
        
    print(f"[DISCONNECT] {client.name} disconnecting from Main Server")
    
    # Notify other clients
    broadcast_msg(client.name, RM)
    
    # Clean up resources
    cleanup_client(client)


def handle_main_conn(name: str):
    """Handle main TCP connection for a client"""
    if name not in clients:
        print(f"[ERROR] Client {name} not found in handle_main_conn")
        return
        
    client: Client = clients[name]
    conn = client.main_conn

    try:
        # Send list of existing clients to new client
        for client_name in list(clients.keys()):  # Use list() to avoid dict changing during iteration
            if client_name == name:
                continue
            if client_name in clients and clients[client_name].connected:
                client.send_msg(client_name, ADD)
        
        # Notify other clients about new client
        broadcast_msg(name, ADD)

        # Main message handling loop
        while client.connected:
            try:
                conn.settimeout(30.0)  # 30 second timeout
                msg_bytes = conn.recv_bytes()
                
                if not msg_bytes:
                    print(f"[{name}] Empty message received, client disconnecting")
                    break
                    
                msg = pickle.loads(msg_bytes)
                print(f"[{name}] Received: {msg}")
                
                if msg.request == DISCONNECT:
                    print(f"[{name}] Explicit disconnect request")
                    break
                elif msg.request == POST:
                    # Handle different types of POST messages
                    if msg.data_type == TEXT:
                        multicast_msg(name, msg.request, msg.to_names, msg.data_type, msg.data)
                    elif msg.data_type == FILE:
                        multicast_msg(name, msg.request, msg.to_names, msg.data_type, msg.data)
                    else:
                        print(f"[{name}] Unhandled POST data_type: {msg.data_type}")
                else:
                    print(f"[{name}] Unhandled request: {msg.request}")
                    
            except socket.timeout:
                # Check if client is still connected
                if not client.connected:
                    break
                continue
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                print(f"[{name}] Connection error: {e}")
                break
            except pickle.UnpicklingError:
                print(f"[{name}] [ERROR] UnpicklingError")
                continue
            except Exception as e:
                print(f"[{name}] Unexpected error: {e}")
                break
                
    except Exception as e:
        print(f"[{name}] Handler error: {e}")
        traceback.print_exc()
    finally:
        disconnect_client(client)


def main_server():
    """Main server function"""
    main_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        main_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        main_socket.bind((IP, MAIN_PORT))
        main_socket.listen(5)  # Allow up to 5 pending connections
        print(f"[LISTENING] Main Server is listening on {IP if IP else 'all interfaces'}:{MAIN_PORT}")

        # Start media servers in daemon threads
        video_server_thread = threading.Thread(target=media_server, args=(VIDEO, VIDEO_PORT), daemon=True)
        audio_server_thread = threading.Thread(target=media_server, args=(AUDIO, AUDIO_PORT), daemon=True)
        
        video_server_thread.start()
        audio_server_thread.start()
        
        print("[INFO] Media servers started")

        while True:
            try:
                main_socket.settimeout(1.0)  # 1 second timeout for clean shutdown
                conn, addr = main_socket.accept()
                
                print(f"[NEW CONNECTION] Connection attempt from {addr}")
                
                # Set socket timeout for initial handshake
                conn.settimeout(10.0)
                
                # Receive client name
                name_bytes = conn.recv_bytes()
                if not name_bytes:
                    conn.close()
                    continue
                    
                name = name_bytes.decode()
                
                # Check if name is already taken
                if name in clients and clients[name].connected:
                    error_msg = "Username already taken"
                    conn.send_bytes(error_msg.encode())
                    conn.close()
                    print(f"[REJECTED] {name} - Username taken")
                    continue
                
                # Accept the client
                conn.send_bytes(OK.encode())
                conn.settimeout(None)  # Remove timeout for normal operation
                
                # Clean up any existing disconnected client with same name
                if name in clients:
                    cleanup_client(clients[name])
                
                # Create new client
                clients[name] = Client(name, conn, True)
                print(f"[ACCEPTED] {name} connected from {addr}")

                # Start handler thread for this client
                main_conn_thread = threading.Thread(
                    target=handle_main_conn, 
                    args=(name,), 
                    daemon=True
                )
                main_conn_thread.start()
                
            except socket.timeout:
                continue  # Normal timeout, continue listening
            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Keyboard interrupt received")
                break
            except Exception as e:
                print(f"[ERROR] Accept error: {e}")
                time.sleep(1)  # Brief pause before continuing
                
    except Exception as e:
        print(f"[ERROR] Main server error: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        print("[SHUTDOWN] Cleaning up...")
        for client in list(clients.values()):
            disconnect_client(client)
        main_socket.close()
        print("[SHUTDOWN] Server stopped")


if __name__ == "__main__":
    try:
        main_server()
    except KeyboardInterrupt:
        print(f"\n[EXITING] Keyboard Interrupt")
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        # Force exit
        os._exit(0)