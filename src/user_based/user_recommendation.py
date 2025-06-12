from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def user_recommendation(user_id, data, user_embeddings, product_encoder, top_n=10, min_rating=4.0):
    """
    Memberikan rekomendasi produk untuk user tertentu berdasarkan kemiripan pengguna lain (user-based CF).

    Parameter:
    - user_id: ID user hasil dari LabelEncoder (bukan buyer_id asli)
    - data: DataFrame yang telah memiliki kolom 'user', 'product', 'product_rating', 'product_id', 'product_name'
    - user_embeddings: matriks embedding pengguna (dari model)
    - product_encoder: encoder untuk produk
    - top_n: jumlah rekomendasi produk
    - min_rating: threshold rating yang dianggap sebagai "tinggi"

    Output:
    - Print daftar rekomendasi
    """

    if user_id >= len(user_embeddings):
        print(f"User ID {user_id} tidak valid.")
        return

    # Cosine Similarity antar pengguna
    target_user_embedding = user_embeddings[user_id]
    similarities = cosine_similarity([target_user_embedding], user_embeddings)

    # Top-N pengguna mirip (kecuali dirinya sendiri)
    similar_users = similarities.argsort()[0][::-1][1:top_n + 1]

    # Produk dengan rating tinggi dari pengguna mirip
    recommended_products = set()
    for similar_user in similar_users:
        user_data = data[data['user'] == similar_user]
        high_rated_products = user_data[user_data['product_rating'] >= min_rating]['product']
        recommended_products.update(high_rated_products)

    # Hilangkan produk yang sudah pernah dibeli/nilai oleh user target
    target_user_products = set(data[data['user'] == user_id]['product'])
    recommended_products -= target_user_products

    if not recommended_products:
        print(f"Tidak ada rekomendasi yang tersedia untuk User {user_id}.")
        return

    # Ambil top_n produk
    recommended_products = list(recommended_products)[:top_n]

    # Decode ke ID asli dan cari nama produk
    recommended_product_ids = product_encoder.inverse_transform(recommended_products)
    recommended_names = []
    for pid in recommended_product_ids:
        product_row = data[data['product_id'] == pid]
        if not product_row.empty:
            recommended_names.append(product_row['product_name'].iloc[0])
        else:
            recommended_names.append(f"(nama tidak ditemukan untuk ID {pid})")

    # Hasil
    print(f"User-Based Recommendations for User {user_id}:")
    for name in recommended_names:
        print(f"- {name}")