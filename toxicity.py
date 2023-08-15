import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd

df = pd.read_csv('train.csv')
X = df['comment_text']


model = load_model('toxicity.h5')


vectorizer = tf.keras.layers.TextVectorization(max_tokens=200000, output_sequence_length=1850, output_mode='int')
vectorizer.adapt(X.values) 

st.title("Toxicity Checker")

comment = st.text_area("Enter your comment:")

if st.button("Post Comment"):
    vectorized_comment = vectorizer(np.array([comment]))  # Wrap in a NumPy array
    predictions = model.predict(vectorized_comment)
    if np.any(predictions > 0.5):
        st.error("Cannot post this comment as it is against our policy.")
    else:
        st.success("Comment posted successfully!")
