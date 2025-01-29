import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.title("Chatbot with DeepSeek-R1")
st.write("Enter a message below and get a response from the AI.")

# User input
user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    response = generate_response(user_input)
    st.text_area("Chatbot:", response, height=200)

st.write("Powered by DeepSeek-R1")