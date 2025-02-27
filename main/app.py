import os
import streamlit as st
from dotenv import load_dotenv

# ✅ Set Page Config FIRST (Before Any Streamlit Commands)
st.set_page_config(layout="wide")

# ✅ Load environment variables
load_dotenv()

# ✅ Import modules with error handling
try:
    from LLM.helper import generate_disease_summary, generate_detailed_overview
    from preprocessing import predict_image_label
except ImportError as e:
    st.error(f"❌ Module Import Error: {e}")
    st.stop()

# ✅ Custom Styling
st.markdown("""
    <style>
    .stApp { padding: 10px; }
    .custom-column { padding: 0 40px; }
    </style>
    """, unsafe_allow_html=True
)

def main():
    st.title("🩺 X-ray Image Classification & Disease Overview")

    # ✅ Create two columns for image upload & prediction
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="custom-column">', unsafe_allow_html=True)

        # ✅ Upload the X-ray image
        uploaded_image = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "jpeg", "png"])

        if uploaded_image:
            st.image(uploaded_image, caption="🖼 Uploaded X-ray Image", use_column_width=True, width=300)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if uploaded_image:
            st.markdown('<div class="custom-column">', unsafe_allow_html=True)

            # ✅ Predict the label using the uploaded image
            try:
                predicted_label = predict_image_label(uploaded_image)
                st.subheader("🔍 Predicted Label:")
                st.write(predicted_label)
            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")
                st.stop()

            if predicted_label == "Normal":
                st.success("✅ You have a normal condition.")
            else:
                # ✅ Generate disease summary using the Gemini API
                try:
                    summary = generate_disease_summary(predicted_label)
                    st.subheader("🩺 Disease Information:")
                    st.write(summary)
                except Exception as e:
                    st.error(f"❌ Disease Summary Error: {e}")

            st.markdown('</div>', unsafe_allow_html=True)

    # ✅ Follow-up question section
    st.markdown("<div>", unsafe_allow_html=True)
    st.subheader("💬 Ask a Follow-up Question:")

    follow_up_question = st.text_input("Enter your question about the disease:")

    if follow_up_question and uploaded_image and predicted_label != "Normal":
        st.subheader("📖 Detailed Overview:")

        # ✅ Get detailed response to the follow-up question
        try:
            detailed_response = generate_detailed_overview(predicted_label, follow_up_question)
            st.write(detailed_response)
        except Exception as e:
            st.error(f"❌ Detailed Overview Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
