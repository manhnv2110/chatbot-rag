from app.generator.llm_client import generate_text 
from app.retrieval.retrieve import retrieve

def run_rag_pipeline(user_query):
    relevant_docs = retrieve(user_query)["results"]

    if not relevant_docs: 
        return {
            "query": user_query,
            "answer": "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong cơ sở dữ liệu."
        }

    context_text = "\n\n".join([doc["document_text"] for doc in relevant_docs])

    prompt = f"""
        Dưới đây là các thông tin thực tế từ databse:
        {context_text}
        Hãy dùng các thông tin này để trả lời câu hỏi của người dùng:
        {user_query}
        Nếu không tìm thấy thông tin liên quan, hãy nói rõ ràng bạn không chắc chắn.
    """

    answer = generate_text(prompt)
    return {
        "query": user_query,
        "answer": answer
    }