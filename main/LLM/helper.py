import os
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai

# Load environment variables dynamically
env_path = find_dotenv()
if not env_path:
    raise FileNotFoundError("❌ .env file not found! Ensure it exists in 'C:\\xray-classification-cnn-llm'.")

load_dotenv(env_path)

# Retrieve API key
api_key = os.getenv("GOOGLE_GEMINI_API_KEY")  # Matches your .env file
if not api_key:
    raise ValueError("❌ API key not found! Ensure GOOGLE_GEMINI_API_KEY is set in the .env file.")

# Configure the Gemini API
genai.configure(api_key=api_key)

# Use a valid model
MODEL_NAME = "gemini-1.5-pro-latest"

# Initialize the Gemini model
try:
    gemini_model = genai.GenerativeModel(MODEL_NAME)
    print(f"✅ Using model: {MODEL_NAME}")
except Exception as e:
    raise ValueError(f"❌ Error initializing model '{MODEL_NAME}': {e}")

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
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Error during disease summary generation: {e}")
        return "❌ Error: Unable to generate disease summary."

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
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Error during detailed overview generation: {e}")
        return "❌ Error: Unable to generate detailed overview."

if __name__ == "__main__":
    print("🩺 Welcome to the Medical Assistant powered by Gemini API! 🚀")

    disease = input("Enter the disease name: ").strip()
    print("\n🔍 Generating disease summary...")
    print(generate_disease_summary(disease))

    question = input("\n❓ Enter a specific question about the disease: ").strip()
    print("\n📝 Generating detailed overview...")
    print(generate_detailed_overview(disease, question))
