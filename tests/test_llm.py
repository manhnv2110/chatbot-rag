from llm_client import generate_text

res = generate_text(
    prompt="Cho tôi biết cách học lập trình hiệu quả",
    temperature=0.8,
    max_tokens=512
)

print(res)