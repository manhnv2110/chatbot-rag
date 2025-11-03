from rag_pipeline import run_rag_pipeline

#query = "Cho tôi biết có bao nhiêu sản phẩm ở trong cửa hàng"
query = "Tôi muốn tìm áo thun màu đen. Trong cửa hàng có bao nhiêu loại như thế nhỉ?"
answer = run_rag_pipeline(query)
print("💬 Câu trả lời:")
print(answer)