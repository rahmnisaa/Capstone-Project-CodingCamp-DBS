def get_embedding(model, tokenizer, texts, max_seq_len):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_seq_len-1)
    return model.predict(padded)