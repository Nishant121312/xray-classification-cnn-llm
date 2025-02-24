import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the Gemini API (from Google AI Studio)
genai.configure(api_key=os.getenv("AIzaSyBZ-n2ACm8c_EvyV5xKqylONT235g7dX-Y"))

# Initialize the Gemini model
gemini_model = genai.GenerativeModel("gemini-pro")

# Function to generate a disease summary
def generate_disease_summary(disease):
    """
    Generate a disease summary based on the input disease using Gemini API.
    """
    prompt = f"""
    As a knowledgeable medical assistant, provide a clear and concise summary about the disease {disease}.
    Ensure the information is easy to understand and medically accurate.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

# Function to generate a detailed disease overview
def generate_detailed_overview(disease, question):
    """
    Generate a detailed overview for a disease and answer a specific question using Gemini API.
    """
    prompt = f"""
    As a medical assistant, provide a comprehensive and accurate explanation about the disease {disease}.
    Here is the specific question: {question}
    Ensure the information is medically sound and stays focused only on {disease}.
    """
    response = gemini_model.generate_content(prompt)
    return response.text
