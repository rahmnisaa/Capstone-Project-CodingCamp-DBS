import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_user_embedding_model(data, n_latent_factors=50):
    # Label encoding untuk user dan product
    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    data['user'] = user_encoder.fit_transform(data['buyer_id'])
    data['product'] = product_encoder.fit_transform(data['product_id'])

    n_users = data['user'].nunique()
    n_products = data['product'].nunique()

    # Input layers
    user_input = tf.keras.Input(shape=(1,), name='user_input')
    product_input = tf.keras.Input(shape=(1,), name='product_input')

    # Embedding layers
    user_embedding = tf.keras.layers.Embedding(input_dim=n_users, output_dim=n_latent_factors, name='user_embedding')(user_input)
    product_embedding = tf.keras.layers.Embedding(input_dim=n_products, output_dim=n_latent_factors, name='product_embedding')(product_input)

    # Flatten embeddings
    user_vec = tf.keras.layers.Flatten()(user_embedding)
    product_vec = tf.keras.layers.Flatten()(product_embedding)

    # Model (output hanya user vector, sesuai kode awal)
    model = tf.keras.Model(inputs=user_input, outputs=user_vec)

    # Return model
    return model
