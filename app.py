import streamlit as st
import pickle
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

final_intents = 'data/final_intents.json'
with open(final_intents, 'r', encoding='utf-8') as f:
  data = json.load(f)

# Placeholder function to load model and components
def load_model_components(model_path, tokenizer_path, lab_enc_path):
    # Load your model here
    # model = tf.keras.models.load_model(model_path)
    model = keras.models.load_model(model_path)
    # model = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    # Load tokenizer and label encoder
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(lab_enc_path, 'rb') as enc:
        label_encoder = pickle.load(enc)
    return model, tokenizer, label_encoder

def generate_response(user_input, model, tokenizer, label_encoder):
    # processed_input = tokenizer.texts_to_sequences([user_input])
    # response_code = model.predict(processed_input)
    # response = label_encoder.inverse_transform([response_code])
    max_len = 96
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]),truncating='post', maxlen=max_len))
    tag = label_encoder.inverse_transform([np.argmax(result)])
    for i in data['intents']:
            if i['tag'] == tag:
                response = np.random.choice(i['responses'])
    return response

# Load components

model_path = "utils/model/finetuned_model.h5"
tokenizer_path = "utils/tokenizer/tokenizer.pkl"
lab_enc_path = "utils/encoder/label_encoder.pkl"

model, tokenizer, label_encoder = load_model_components(model_path, tokenizer_path, lab_enc_path)


#=============Streamlit Deployment===================
gradient_text_html = """
<style>
.gradient-text {
    font-weight: bold;
    background: -webkit-linear-gradient(left, red, orange);
    background: linear-gradient(to right, red, orange);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 3em;
}
</style>
<div class="gradient-text">Mental Health Therapist</div>
"""

def display_message(message, is_user=True):
    """Displays a message in the chat."""
    # Define message bubble properties
    color = "#FF5733" if is_user else "#3333FF"
    align = "right" if is_user else "left"
    max_width = "150%"  # Adjust the width of the message bubble as needed
    
    # Create containers for the messages with proper alignment
    if is_user:
        col1, col2, col3 = st.columns([1, 5, 5])  # Adjust the ratios for alignment
        with col2:
            st.write("")  # Empty block for spacing
        with col3:
            # Message bubble for the user aligned to the right
            st.markdown(
                f"<div style='text-align: {align}; background-color: {color}; "
                f"border-radius: 10px; padding: 10px; color: white; "
                f"display: inline-block; max-width: {max_width}; "
                f"float: {align}; margin-top:4px;'>"
                f"{message}</div>", unsafe_allow_html=True
            )
    else:
        col1, col2, col3 = st.columns([5, 5, 1])  # Adjust the ratios for alignment
        with col1:
            # Message bubble for the bot aligned to the left
            st.markdown(
                f"<div style='text-align: {align}; background-color: {color}; "
                f"border-radius: 10px; padding: 10px; color: white; "
                f"display: inline-block; max-width: {max_width}; "
                f"float: {align}; margin-top:4px;'>"
                f"{message}</div>", unsafe_allow_html=True
            )
        with col2:
            st.write("")  # Empty block for spacing


st.markdown(gradient_text_html, unsafe_allow_html=True)
st.caption("Tell me anything! I'm a virtual person!")


# Initialize session state for conversation and user input if not already present
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'input_value' not in st.session_state:
    st.session_state.input_value = ""

# Display chat history
chat_placeholder = st.empty()
with chat_placeholder.container():
    for message, is_user in st.session_state.conversation:
        display_message(message, is_user)

# Create a form for the user input and send button
with st.form(key='chat_form'):
    # Use a separate key for the input value in session state
    user_input = st.text_input("Type your message:", value=st.session_state.input_value, key="user_input_field")
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    st.session_state.conversation.append(("You: " + user_input, True))
    # Your logic to generate a response goes here
    response = generate_response(user_input,model, tokenizer, label_encoder)  # Replace with your response generation logic
    st.session_state.conversation.append(("Bot: " + response, False))
    # Update the input value in session state to clear the input field
    st.session_state.input_value = ""
    # Update chat display
    # chat_placeholder.empty()
    st.experimental_rerun()






