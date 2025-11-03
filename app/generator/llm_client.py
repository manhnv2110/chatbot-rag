import os 
from dotenv import load_dotenv 
from groq import Groq 

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_text(prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                { "role": "system", "content": "Bạn là một trợ lý AI hữu ích." },
                { "role": "user", "content": prompt }
            ],
            temperature=temperature,
            max_tokens=max_tokens 
        ) 

        generated_text = response.choices[0].message.content.strip()
        return generated_text
    except Exception as e:
        print(f"Error: {e}")
        return "Xin lỗi, hiện tại tôi đang gặp sự cố khi tạo câu trả lời"
        