import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Constants
LATENT_DIM = 100
NUM_IMAGES = 5

# Load the trained generator model (make sure this file is present in the same directory)
@st.cache_resource
def load_generator():
    return tf.keras.models.load_model("generator_model.h5")

generator = load_generator()

# Streamlit UI
st.title("MNIST Digit Generator using cGAN (TensorFlow) METI- PK002592")
st.markdown("Select a digit (0–9) and generate **5 different images** using your trained model.")

# Digit input
selected_digit = st.selectbox("Choose a digit:", list(range(10)), index=0)

if st.button("Generate Images"):
    # Generate random noise and labels
    noise = tf.random.normal([NUM_IMAGES, LATENT_DIM])
    labels = tf.constant([[selected_digit]] * NUM_IMAGES, dtype=tf.int32)

    # Generate images using the model
    generated_images = generator([noise, labels], training=False)

    # Plot images
    fig, axes = plt.subplots(1, NUM_IMAGES, figsize=(10, 2))
    for i in range(NUM_IMAGES):
        axes[i].imshow((generated_images[i, :, :, 0] + 1) / 2.0, cmap="gray")
        axes[i].axis("off")

    st.pyplot(fig)
