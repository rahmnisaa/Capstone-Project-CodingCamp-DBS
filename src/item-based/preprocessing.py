import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def preprocess_data(data):
    # Drop baris dengan product_rating == 'No Rating'
    df = data[data['product_rating'] != 'No Rating'].copy()

    # Konversi ke float
    df['product_rating'] = df['product_rating'].astype(float)

    # Normalize price and rating
    scaler = MinMaxScaler()
    df[['norm_price', 'norm_rating']] = scaler.fit_transform(df[['product_price', 'product_rating']])

    # TF-IDF untuk nama produk
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['product_name'])

    # Gabungkan semua fitur ke dalam item vector
    price_rating_features = df[['norm_price', 'norm_rating']].values
    combined_features = np.hstack([tfidf_matrix.toarray(), price_rating_features])

    return df, combined_features
