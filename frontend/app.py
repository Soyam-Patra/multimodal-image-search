import requests
from PIL import Image
import streamlit as st

# ------------------------
# UI
# ------------------------
st.title("🔍 Multimodal Image Search")

query = st.text_input("Enter your query")

if query:
    response = requests.post(
        "http://127.0.0.1:8000/search-text",
        params={"query": query}
    )

    results = response.json()["results"]

    st.subheader("Results")

    for item in results:
        path = item["path"]
        score = item["score"]

        img = Image.open(path)
        st.image(img, caption=f"Score: {score:.3f}")

st.subheader("Image Search")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Query Image")

    files = {"file": uploaded_file.getvalue()}

    response = requests.post(
        "http://127.0.0.1:8000/search-image",
        files=files
    )

    results = response.json()["results"]

    for item in results:
        path = item["path"]
        score = item["score"]

        img = Image.open(path)
        st.image(img, caption=f"Score: {score:.3f}")

