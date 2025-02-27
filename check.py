import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv("C:/xray-classification-cnn-llm/.env")

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Ensure GEMINI_API_KEY is set in the .env file.")

genai.configure(api_key=api_key)

# List available models
print("Fetching available models...")
try:
    models = genai.list_models()
    for model in models:
        print(model.name)
except Exception as e:
    print(f"Error while listing models: {e}")
