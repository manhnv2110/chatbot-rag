from pydantic import BaseModel
from collections import defaultdict

class ConversationManager:
    # Quản lý lịch sử chat 
    def __init__(self):
        # Format: 
        # {
        #    session_id: [
        #       { "role": "user", "content": "" },
        #       { "role": "bot", "content": "" }
        #    ]
        # }
        self.histories = defaultdict(list)

    def add_message(self, session_id: str, role: str, content: str, max_length=20):
        # Thêm tin nhắn vào lịch sử chat 
        self.histories[session_id].append({ 
            "role": role, 
            "content": content 
        })

        if len(self.histories[session_id]) > max_length:
            self.histories[session_id] = self.histories[session_id][-max_length:]

    def get_context(self, session_id: str, limit: int=5):
        # Lấy ngữ cảnh hội thoại gần nhất (mặc định là 5 tin nhắn gần nhất)
        history = self.histories.get(session_id, [])
        return history[-limit:] 

    def clear_history(self, session_id):
        # Xóa toàn bộ lịch sử chat 
        if session_id in self.histories:
            del self.histories[session_id]
