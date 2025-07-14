#utils/tcp_client.py

import socket
import os

def send_bill(host: str, port: int, file_path: str) -> None:
    # đọc CSV bytes
    data = open(file_path, "rb").read()
    # filename + newline + data
    payload = os.path.basename(file_path).encode("utf-8") + b"\n" + data
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        # gửi độ dài payload, rồi payload
        s.sendall(len(payload).to_bytes(4, "big") + payload)