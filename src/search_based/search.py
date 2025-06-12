import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# Load model dan data
model = tf.keras.models.load_model("model_final.h5")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = load_npz("tfidf_matrix.npz")
df_produk = pd.read_csv("df_produk.csv")

# Tentukan jumlah kategori (misalnya dari one-hot encoder)
num_categories = df_produk['category_label'].nunique()

# Fungsi rekomendasi
def recommend_with_nn(keyword, df_produk, vectorizer, tfidf_matrix_produk, model, top_n=30):
    # Vektorisasi keyword
    query_vec = vectorizer.transform([keyword.lower()])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix_produk).flatten()

    # Ambil kandidat terdekat
    candidate_indices = cosine_sim.argsort()[-50:][::-1]  # ambil 50 untuk variasi
    hasil = []

    # Proses semua kandidat
    for idx in candidate_indices:
        produk_tfidf_vec = tfidf_matrix_produk[idx].toarray().flatten()
        category_label = df_produk.iloc[idx]['category_label']
        category_one_hot = np.zeros(num_categories)
        category_one_hot[category_label] = 1

        # Gabung input untuk model
        x_input = np.concatenate([category_one_hot, produk_tfidf_vec]).reshape(1, -1)
        prob = model.predict(x_input, verbose=0)[0][0]

        hasil.append({
            'product_name': df_produk.iloc[idx]['product_name'],
            'seller_id': df_produk.iloc[idx].get('seller_id', 'N/A'),
            'score': prob
        })

    # Urutkan hasil tanpa menampilkan skor
    df_hasil = pd.DataFrame(hasil)
    df_hasil_sorted = df_hasil.sort_values(by='score', ascending=False).drop(columns=['score'])
    return df_hasil_sorted.head(top_n)
