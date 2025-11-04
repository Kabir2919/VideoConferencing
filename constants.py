import socket
import struct
import pickle
from dataclasses import astuple, dataclass

PORT = 53535
MAIN_PORT = 53530
VIDEO_PORT = 53531
AUDIO_PORT = 53532
DISCONNECT = 'QUIT!'
OK = 'OK'
SIZE = 1024

SERVER = 'SERVER'

# requests
GET = 'GET'
POST = 'POST'
ADD = 'ADD'
RM = 'RM'

# data types
VIDEO = 'Video'
AUDIO = 'Audio'
TEXT = 'Text'
FILE = 'File'

# FIXED: Reduced UDP packet sizes to avoid network limits
# Most networks have MTU of 1500 bytes, so UDP payload should be well under that
MEDIA_SIZE = {
    VIDEO: 30000,   # ~29KB for encoded frame + overhead
    AUDIO: 5000     # ~1.5KB for audio data + overhead
} # Much smaller, safer sizes


def send_bytes(self, msg):
    """Send bytes with length prefix"""
    if not msg:
        return
    try:
        # Prefix each message with a 4-byte length (network byte order)
        msg_with_length = struct.pack('>I', len(msg)) + msg
        self.sendall(msg_with_length)
    except (BrokenPipeError, ConnectionResetError, OSError, socket.error) as e:
        raise e

def recv_bytes(self):
    """Receive bytes with length prefix"""
    try:
        # Read message length and unpack it into an integer
        raw_msglen = self.recvall(4)
        if not raw_msglen:
            return b''
        msglen = struct.unpack('>I', raw_msglen)[0]
        
        # Validate message length
        if msglen <= 0 or msglen > 10 * 1024 * 1024:  # Max 10MB message
            print(f"[ERROR] Invalid message length: {msglen}")
            return b''
            
        # Read the message data
        return self.recvall(msglen)
    except (BrokenPipeError, ConnectionResetError, OSError, socket.error, struct.error):
        return b''

def recvall(self, n):
    """Helper function to recv n bytes or return None if EOF is hit"""
    if n <= 0:
        return b''
        
    data = bytearray()
    while len(data) < n:
        try:
            remaining = n - len(data)
            packet = self.recv(remaining)
            if not packet:
                # Connection closed
                return b''
            data.extend(packet)
        except (BrokenPipeError, ConnectionResetError, OSError, socket.error):
            return b''
    return bytes(data)

def disconnect(self):
    """Gracefully disconnect the socket"""
    try:
        if hasattr(self, '_closed') and not self._closed:
            # Send disconnect message if it's a main connection
            msg = Message(SERVER, DISCONNECT)
            try:
                self.send_bytes(pickle.dumps(msg))
            except:
                pass  # Ignore errors when sending disconnect message
        self.close()
    except:
        pass  # Ignore any errors during disconnect

# Monkey patch the socket class
socket.socket.send_bytes = send_bytes
socket.socket.recv_bytes = recv_bytes
socket.socket.recvall = recvall
socket.socket.disconnect = disconnect

@dataclass
class Message:
    from_name: str
    request: str
    data_type: str = None
    data: any = None
    to_names: tuple[str] = None

    def __str__(self):
        if self.data_type in [VIDEO, AUDIO]:
            data = f"<{self.data_type}_DATA>"
        else:
            data = str(self.data)[:100] + "..." if str(self.data) and len(str(self.data)) > 100 else self.data
        return f"[{self.from_name}] {self.request}:{self.data_type} -> {self.to_names} {data}"

    def __iter__(self):
        return iter(astuple(self))
    
    def __getitem__(self, keys):
        return iter(getattr(self, k) for k in keys)