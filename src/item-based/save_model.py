import tensorflow as tf
import os

def save_model(model, save_path):
    tf.saved_model.save(model, save_path)  # Menyimpan model dalam format SavedModel
    print(f"✅ Saved TensorFlow model in SavedModel format at {save_path}")

def convert_to_tfjs(save_path, tfjs_path):
    os.system(f"tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model {save_path} {tfjs_path}")
    print(f"✅ Converted to TensorFlow.js format at {tfjs_path}")
