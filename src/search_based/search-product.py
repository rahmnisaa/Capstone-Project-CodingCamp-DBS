def search_product(user_input, model, tokenizer, product_embeddings, data, max_seq_len):
    user_seq = tokenizer.texts_to_sequences([user_input.lower()])
    user_padded = pad_sequences(user_seq, maxlen=max_seq_len-1)
    user_embed = model.predict(user_padded)

    similarities = cosine_similarity(user_embed, product_embeddings)
    indices = similarities.argsort()[0][::-1]
    results = data.iloc[indices][['product_name', 'store_id']]
    return results.drop_duplicates(subset=['store_id']).head(10)