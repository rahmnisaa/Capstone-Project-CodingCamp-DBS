# Capstone-Project-CodingCamp-DBS
# 🤖 Pinjemin - Machine Learning Models

Selamat datang di repository **Machine Learning Models** untuk aplikasi **Pinjemin**! Repository ini berisi model-model rekomendasi dan pencarian berbasis machine learning yang mendukung sistem pencarian dan penyewaan barang berdasarkan hobi pengguna.

### 📁 model/
Berisi file hasil pelatihan dari setiap model:

- `item_based/`: Model rekomendasi berdasarkan kemiripan antar barang.
- `user_based/`: Model rekomendasi berdasarkan kesamaan antar pengguna.
- `keyword_search/`: Model berbasis pencarian kata kunci dengan pendekatan vektorisasi dan pembobotan (misal TF-IDF atau BM25).

### 📁 src/
Berisi kode sumber untuk setiap model, mulai dari tahap preprocessing, pelatihan model, hingga inferensi.

- `item_based/`, `user_based/`, `keyword_search/`: Script Python modular untuk masing-masing pendekatan.
- `notebooks/`: Jupyter Notebook dari tahap awal hingga inferensi, cocok untuk eksplorasi dan dokumentasi proses pengembangan.

### 📁 all_dataset/
Berisi semua dataset yang digunakan dalam proses pelatihan dan pengujian:

- `user_data.csv`: Informasi pengguna (misalnya hobi, lokasi, dan histori interaksi).
- `item_data.csv`: Informasi barang (misalnya nama, kategori, deskripsi).
- `interaction_data.csv`: Data interaksi antara pengguna dan barang (klik, sewa, simpan, dll).

## 🧠 Model yang Digunakan

- **Item-Based Collaborative Filtering**  
  Rekomendasi barang berdasarkan kemiripan deskripsi atau interaksi pengguna lain terhadap barang yang sama.

- **User-Based Collaborative Filtering**  
  Menyajikan rekomendasi dari barang yang disukai oleh pengguna dengan profil serupa.

- **Keyword-Based Search (Content-Based)**  
  Sistem pencarian yang menyesuaikan deskripsi barang dengan preferensi atau input pencarian pengguna menggunakan representasi teks.

## 📌 Catatan

- Model dilatih dan dievaluasi menggunakan metrik relevansi dan kemiripan (misalnya cosine similarity, precision@k).
- File dataset tidak dibagikan secara publik untuk menjaga privasi dan kepatuhan terhadap kebijakan data.

---

📬 Untuk pertanyaan atau kontribusi, silakan buka *issue* atau *pull request*.  
Terima kasih telah mengunjungi proyek kami!

