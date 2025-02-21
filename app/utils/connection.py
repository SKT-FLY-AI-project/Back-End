from fastapi import WebSocket
from typing import Dict
import json

# 클라이언트 연결 관리
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception:
                self.disconnect(client_id)
    
    def is_connected(self, client_id: str) -> bool:
        return client_id in self.active_connections

# 연결 관리자 인스턴스 생성
manager = ConnectionManager()