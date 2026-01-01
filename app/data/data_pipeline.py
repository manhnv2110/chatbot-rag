from app.data.product_data_builder import build_products_embeddings
from app.data.category_data_builder import build_categories_embeddings
from app.data.faq_data_builder import build_faq_embeddings
from app.data.policy_data_builder import build_policy_embeddings
from app.data.order_data_builder import build_order_guide_embeddings

if __name__ == "__main__":
    build_faq_embeddings()
    build_policy_embeddings()
    build_order_guide_embeddings()
    build_categories_embeddings()
    build_products_embeddings()