import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_autoencoder(input_dim, output_dim):
    model = Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(output_dim, activation='linear'),
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(input_dim, activation='linear')  # Output size sama dengan input
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, combined_features, epochs=30, batch_size=64):
    model.fit(combined_features, combined_features, epochs=epochs, batch_size=batch_size, verbose=1)
    return model
