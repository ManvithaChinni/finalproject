import streamlit as st
from google.cloud import vision
from gtts import gTTS
from PIL import Image
import os


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_key.json"  


def scene_understanding(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as img_file:
        content = img_file.read()
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    descriptions = [label.description for label in response.label_annotations]
    return "Scene description: " + ", ".join(descriptions)


def extract_text_google(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as img_file:
        content = img_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.text_annotations:
        return response.text_annotations[0].description
    else:
        return "No text detected."


def text_to_speech(text, output_file="output.mp3", lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save(output_file)
    return output_file


st.title("👁️‍🗨️ Vision Assist for the Visually Impaired 👩‍🦯")
st.write("Upload an image to understand its content and convert text to speech:")


uploaded_file = st.file_uploader("📂 Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file:
    
    image_path = "uploaded_image.png"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    
    st.image(image_path, caption="Uploaded Image", use_column_width=True)

    
    st.header("1. Scene Understanding")
    description = scene_understanding(image_path)
    st.write(description)

    
    st.header("2. Text Recognition and TTS")
    text = extract_text_google(image_path)
    st.write("Extracted Text:")
    st.write(text)

    
    st.subheader("Convert Text to Speech")
    lang = st.selectbox("Select Language", ["English", "Hindi", "French", "Spanish"], index=0)
    lang_codes = {"English": "en", "Hindi": "hi", "French": "fr", "Spanish": "es"}
    selected_lang = lang_codes[lang]

    if st.button("Convert to Speech"):
        audio_file = text_to_speech(text, lang=selected_lang)
        st.audio(audio_file, format="audio/mp3")
