from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

# Load Data dan Model
data = pd.read_csv('datasetfinal.csv')
n_users = data['user'].nunique()
model = load_model('model.h5')
user_embeddings = model.predict(np.arange(n_users))

app = FastAPI()

# Pydantic Model untuk input user_id
class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 10

class ProductRecommendation(BaseModel):
    product_name: str
    product_id: int
    store_id: int
    product_rating: float
    product_price: float

class RecommendationResponse(BaseModel):
    recommendations: list[ProductRecommendation]

@app.post("/recommendations/", response_model=RecommendationResponse)
async def recommend_by_similar_users(request: RecommendationRequest):
    user_id = request.user_id
    top_n = request.top_n
    
    # Pastikan user_id valid
    if user_id < 1 or user_id > n_users:
        raise HTTPException(status_code=400, detail="User ID is out of range")
    
    # Cosine Similarity antar Pengguna
    target_user_embedding = user_embeddings[user_id - 1]  # Adjust if user_id starts from 1
    similarities = cosine_similarity([target_user_embedding], user_embeddings)
    
    # Sort berdasarkan Kemiripan
    similar_users = similarities.argsort()[0][::-1][1:top_n + 1]
    
    # Ambil produk yang diberi rating tinggi oleh pengguna mirip
    recommended_products = set()
    for similar_user in similar_users:
        user_data = data[data['customer_id'] == similar_user]
        high_rated_products = user_data[user_data['product_rating'] >= 4.0][['product_id', 'product_name', 'store_id', 'product_rating', 'product_price']]
        
        for _, row in high_rated_products.iterrows():
            recommended_products.add((row['product_id'], row['product_name'], row['store_id'], row['product_rating'], row['product_price']))
    
    # Produk yang belum pernah dinilai oleh pengguna target
    target_user_products = data[data['customer_id'] == user_id]['product_id']
    recommended_products = {prod for prod in recommended_products if prod[0] not in target_user_products}
    
    # Ambil top_n produk
    recommended_products = list(recommended_products)[:top_n]
    
    # Kembalikan hasil rekomendasi
    recommendations = []
    for prod in recommended_products:
        product_id, product_name, store_id, product_rating, product_price = prod
        recommendations.append(ProductRecommendation(
            product_name=product_name,
            product_id=product_id,
            store_id=store_id,
            product_rating=product_rating,
            product_price=product_price
        ))
    
    return {"recommendations": recommendations}
