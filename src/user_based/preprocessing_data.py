import pandas as pd

def preprocess_data(data):
    # Drop baris dengan product_rating == 'No Rating'
    df = data[data['product_rating'] != 'No Rating']

    # Konversi ke float
    df['product_rating'] = df['product_rating'].astype(float)

    return df