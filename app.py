
import streamlit as st
import openai
import cv2
import numpy as np
from PIL import Image

# Load OpenAI API key securely from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

def analyze_fracture_with_gpt(description):
    prompt = f"""
You are a metallurgical failure analyst with expertise in fracture mechanics and the ASM Handbook.

Analyze the following fracture surface image description:

{description}

Provide a detailed explanation including the likely failure mode (brittle, ductile, fatigue, torsion), possible causes, and visual indicators.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a professional fracture analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.5
    )
    return response['choices'][0]['message']['content']

def extract_image_features(image):
    image_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    # Simple heuristics to generate descriptive input for GPT
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1.2, 50,
                               param1=50, param2=30, minRadius=10, maxRadius=100)

    description = "The image shows a metal fracture surface. "
    if circles is not None:
        description += "There are concentric circular patterns that may indicate fatigue origins. "
    if np.mean(edges) > 20:
        description += "The fracture surface appears rough, which may suggest brittle failure. "
    else:
        description += "The surface appears smooth, which may suggest ductile failure. "

    return edges, image_array, description

# Streamlit app layout
st.set_page_config(page_title="Fracture Analyzer AI", layout="centered")
st.title("🔍 Fracture Surface Analyzer (GPT-powered)")
st.write("Upload a photo of a fractured metal component and get expert-level analysis from GPT-4.")

uploaded_file = st.file_uploader("Upload Fracture Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    edges, original_image, description = extract_image_features(image)

    st.subheader("Original Image")
    st.image(original_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Edge Detection Result")
    st.image(edges, caption="Edge Detected Image", use_column_width=True, clamp=True, channels="GRAY")

    with st.spinner("Analyzing with GPT-4..."):
        feedback = analyze_fracture_with_gpt(description)

    st.subheader("AI Analysis Result")
    st.markdown(f"> {feedback}")
