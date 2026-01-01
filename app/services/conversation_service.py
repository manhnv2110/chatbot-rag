from typing import List, Optional, Dict
from app.core.database import connectDB
from app.models.conversation import (
    Conversation, Message, ConversationWithMessages
)
from datetime import datetime
import json 

class ConversationService:
    """Service để quản lý conversations và messages (MySQL compatible)"""

    def create_conversation(self, user_id: int, title: str = "Hội thoại mới") -> Conversation:
        """Tạo conversation mới cho user"""
        try:
            with connectDB() as conn:
                with conn.cursor() as cur:
                    # 1. Insert
                    cur.execute(
                        """
                        INSERT INTO conversations (user_id, title)
                        VALUES (%s, %s)
                        """,
                        (user_id, title)
                    )

                    conversation_id = cur.lastrowid

                    # 2. Select lại bản ghi vừa insert
                    cur.execute(
                        """
                        SELECT id, user_id, title, created_at, updated_at, is_active
                        FROM conversations
                        WHERE id = %s
                        """,
                        (conversation_id,)
                    )

                    result = cur.fetchone()
                    conn.commit()

                    return Conversation(
                        id=result["id"],
                        user_id=result["user_id"],
                        title=result["title"],
                        created_at=result["created_at"],
                        updated_at=result["updated_at"],
                        is_active=result["is_active"],
                    )
        except Exception as e:
            print(f"❌ Error creating conversation: {e}")
            raise

    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """Lấy thông tin conversation"""
        try:
            with connectDB() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, user_id, title, created_at, updated_at, is_active
                        FROM conversations
                        WHERE id = %s
                        """,
                        (conversation_id,)
                    )
                    result = cur.fetchone()

                    if not result:
                        return None

                    return Conversation(
                        id=result["id"],
                        user_id=result["user_id"],
                        title=result["title"],
                        created_at=result["created_at"],
                        updated_at=result["updated_at"],
                        is_active=result["is_active"],
                    )
        except Exception as e:
            print(f"❌ Error getting conversation: {e}")
            return None

    def get_user_conversations(self, user_id: int, limit: int = 20) -> List[Conversation]:
        """Lấy danh sách conversations của user"""
        try:
            with connectDB() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, user_id, title, created_at, updated_at, is_active
                        FROM conversations
                        WHERE user_id = %s AND is_active = 1
                        ORDER BY updated_at DESC
                        LIMIT %s
                        """,
                        (user_id, limit)
                    )
                    results = cur.fetchall()

                    return [
                        Conversation(
                            id=r["id"],
                            user_id=r["user_id"],
                            title=r["title"],
                            created_at=r["created_at"],
                            updated_at=r["updated_at"],
                            is_active=r["is_active"],
                        )
                        for r in results
                    ]
        except Exception as e:
            print(f"❌ Error getting user conversations: {e}")
            return []

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> Message:
        """Thêm message vào conversation"""
        try:
            with connectDB() as conn:
                with conn.cursor() as cur:
                    metadata_json = json.dumps(metadata) if metadata else None 
                    
                    # 1. Insert message
                    cur.execute(
                        """
                        INSERT INTO bot_messages (conversation_id, role, content, metadata)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (conversation_id, role, content, metadata_json),
                    )

                    message_id = cur.lastrowid

                    # 2. Update conversation.updated_at
                    cur.execute(
                        """
                        UPDATE conversations
                        SET updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """,
                        (conversation_id,),
                    )

                    # 3. Select message vừa insert
                    cur.execute(
                        """
                        SELECT id, conversation_id, role, content, metadata, created_at
                        FROM bot_messages
                        WHERE id = %s
                        """,
                        (message_id,),
                    )

                    result = cur.fetchone()
                    conn.commit()

                    # ✅ FIX: Parse metadata từ JSON string sang dict
                    parsed_metadata = None
                    if result["metadata"]:
                        try:
                            if isinstance(result["metadata"], str):
                                parsed_metadata = json.loads(result["metadata"])
                            else:
                                parsed_metadata = result["metadata"]
                        except json.JSONDecodeError:
                            print(f"⚠️ Warning: Cannot parse metadata: {result['metadata']}")
                            parsed_metadata = None

                    return Message(
                        id=result["id"],
                        conversation_id=result["conversation_id"],
                        role=result["role"],
                        content=result["content"],
                        metadata=parsed_metadata,
                        created_at=result["created_at"],
                    )
        except Exception as e:
            print(f"❌ Error adding message: {e}")
            raise

    def get_conversation_messages(
        self,
        conversation_id: int,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Lấy messages của conversation"""
        try:
            with connectDB() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT id, conversation_id, role, content, metadata, created_at
                        FROM bot_messages
                        WHERE conversation_id = %s
                        ORDER BY created_at ASC
                    """
                    params = [conversation_id]

                    if limit:
                        query += " LIMIT %s"
                        params.append(limit)

                    cur.execute(query, params)
                    results = cur.fetchall()

                    messages = []
                    for r in results:
                        # ✅ FIX: Parse metadata từ JSON string sang dict
                        parsed_metadata = None
                        if r["metadata"]:
                            try:
                                if isinstance(r["metadata"], str):
                                    parsed_metadata = json.loads(r["metadata"])
                                else:
                                    parsed_metadata = r["metadata"]
                            except json.JSONDecodeError:
                                print(f"⚠️ Warning: Cannot parse metadata for message {r['id']}")
                                parsed_metadata = None

                        messages.append(
                            Message(
                                id=r["id"],
                                conversation_id=r["conversation_id"],
                                role=r["role"],
                                content=r["content"],
                                metadata=parsed_metadata,
                                created_at=r["created_at"],
                            )
                        )
                    
                    return messages
                    
        except Exception as e:
            print(f"❌ Error getting messages: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_conversation_with_messages(
        self, conversation_id: int
    ) -> Optional[ConversationWithMessages]:
        """Lấy conversation kèm messages"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None

        messages = self.get_conversation_messages(conversation_id)

        return ConversationWithMessages(
            id=conversation.id,
            user_id=conversation.user_id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            is_active=conversation.is_active,
            messages=messages,
        )

    def update_conversation_title(self, conversation_id: int, title: str) -> bool:
        """Update title conversation"""
        try:
            with connectDB() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE conversations
                        SET title = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """,
                        (title, conversation_id),
                    )
                    conn.commit()
                    return True
        except Exception as e:
            print(f"❌ Error updating conversation title: {e}")
            return False

    def delete_conversation(self, conversation_id: int) -> bool:
        """Soft delete conversation"""
        try:
            with connectDB() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE conversations
                        SET is_active = 0, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """,
                        (conversation_id,),
                    )
                    conn.commit()
                    return True
        except Exception as e:
            print(f"❌ Error deleting conversation: {e}")
            return False

    def get_conversation_history_for_llm(
        self, conversation_id: int, limit: int = 10
    ) -> List[Dict]:
        """Lấy history cho LLM"""
        messages = self.get_conversation_messages(conversation_id)

        recent_messages = messages[-limit:] if len(messages) > limit else messages

        return [
            {"role": msg.role, "content": msg.content}
            for msg in recent_messages
        ]