from sklearn.metrics.pairwise import cosine_similarity

def recommend(product_id, top_n, model, combined_features, product_id_to_index, df):
    if product_id not in product_id_to_index:
        print("Product ID not found.")
        return []
    
    query_index = product_id_to_index[product_id]
    
    # Embed all items
    item_embeddings = model.predict(combined_features, verbose=0)
    query_embedding = item_embeddings[query_index].reshape(1, -1)
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, item_embeddings)[0]
    
    # Ambil top N rekomendasi selain dirinya sendiri
    similar_indices = similarities.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != query_index][:top_n]
    
    recommended_df = df.iloc[similar_indices].copy()
    recommended_df['similarity'] = similarities[similar_indices]
    
    return recommended_df
