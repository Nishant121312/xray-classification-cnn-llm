import os
import streamlit as st
from dotenv import load_dotenv
from LLM.helper import generate_disease_summary, generate_detailed_overview
from preprocessing import predict_image_label

# Load environment variables
load_dotenv()

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .stApp {
        padding: 10px;
    }
    .custom-column {
        padding: 0 40px;
    }
    </style>
    """, unsafe_allow_html=True
)

def main():
    st.title("X-ray Image Classification and Disease Overview")

    # Create two columns for image upload and prediction
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="custom-column">', unsafe_allow_html=True)

        # Upload the X-ray image
        uploaded_image = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded X-ray Image", use_column_width=True, width=300)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if uploaded_image is not None:
            st.markdown('<div class="custom-column">', unsafe_allow_html=True)

            # Predict the label using the uploaded image
            predicted_label = predict_image_label(uploaded_image)

            st.subheader("Predicted Label:")
            st.write(predicted_label)

            if predicted_label == "Normal":
                st.write("✅ You have a normal condition.")
            else:
                # Generate a disease summary using the Gemini API
                summary = generate_disease_summary(predicted_label)
                st.subheader("Disease Information:")
                st.write(summary)

            st.markdown('</div>', unsafe_allow_html=True)

    # Follow-up question section
    st.markdown("<div>", unsafe_allow_html=True)
    st.markdown("<h1>Ask a follow-up question:</h1>", unsafe_allow_html=True)

    follow_up_question = st.text_input("Enter your question about the disease:")

    if follow_up_question and uploaded_image is not None and predicted_label != "Normal":
        st.subheader("Detailed Overview:")

        # Get a more detailed response to the follow-up question
        detailed_response = generate_detailed_overview(predicted_label, follow_up_question)

        st.write(detailed_response)

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
