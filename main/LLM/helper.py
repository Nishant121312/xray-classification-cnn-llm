import os
import streamlit as st
import google.generativeai as genai

# Retrieve API key from Streamlit Secrets or .env (for local development)
api_key = st.secrets.get("GOOGLE_GEMINI_API_KEY")

# Fallback to .env file when running locally
if api_key is None:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")

# Raise an error if API key is still missing
if not api_key:
    st.error("❌ API key not found! Add it to Streamlit Secrets or a .env file.")
    st.stop()

# Configure the Gemini API
genai.configure(api_key=api_key)

# Use a valid model
MODEL_NAME = "gemini-1.5-pro-latest"

# Initialize the Gemini model
try:
    gemini_model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    st.error(f"❌ Error initializing model '{MODEL_NAME}': {e}")
    st.stop()

# Define functions
def generate_disease_summary(disease):
    """Generate a concise and medically accurate summary of a given disease."""
    if not disease:
        return "❌ Error: Disease name cannot be empty."

    prompt = f"""
    You are a medical assistant. Provide a concise and medically accurate summary of the disease: {disease}.
    Keep the explanation clear, factual, and easy to understand.
    """

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response and hasattr(response, "text") else "❌ Error: No response from API."
    except Exception as e:
        return f"⚠️ Error: {e}"

def generate_detailed_overview(disease, question):
    """Provide a comprehensive overview of a disease and answer a specific question."""
    if not disease or not question:
        return "❌ Error: Both disease name and question must be provided."

    prompt = f"""
    You are a medical assistant. Provide a comprehensive explanation about the disease: {disease}.
    Specifically address the following question: {question}.
    Ensure the response is accurate, detailed, and medically sound.
    """

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response and hasattr(response, "text") else "❌ Error: No response from API."
    except Exception as e:
        return f"⚠️ Error: {e}"

# Streamlit UI
st.title("🩺 Medical Assistant powered by Gemini API 🚀")

disease = st.text_input("Enter the disease name:")
if disease:
    st.subheader("🔍 Disease Summary")
    st.write(generate_disease_summary(disease))

question = st.text_input("❓ Ask a specific question about the disease:")
if disease and question:
    st.subheader("📝 Detailed Overview")
    st.write(generate_detailed_overview(disease, question))
