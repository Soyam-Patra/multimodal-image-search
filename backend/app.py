import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, UploadFile, File
import numpy as np
import pickle
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import io
import streamlit as st
import requests

app = FastAPI()

# ------------------------
# Load model
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model = model.to(device)
model.eval()

# ------------------------
# Load index
# ------------------------
embeddings = np.load("data/embeddings.npy")

with open("data/paths.pkl", "rb") as f:
    paths = pickle.load(f)

index = faiss.read_index("data/faiss.index")

# ------------------------
# Encode text
# ------------------------
def encode_text(query):
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.text_model(**inputs)
        text_embeds = outputs.pooler_output
        text_features = model.text_projection(text_embeds)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy().astype("float32")

# ------------------------
# Encode image
# ------------------------
def encode_image(image):
    inputs = processor(images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.vision_model(**inputs)
        image_embeds = outputs.pooler_output
        image_features = model.visual_projection(image_embeds)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy().astype("float32")

# ------------------------
# Search
# ------------------------
def search(query_vec, top_k=5):
    scores, indices = index.search(query_vec, top_k)

    return [
        {"path": paths[i], "score": float(scores[0][j])}
        for j, i in enumerate(indices[0])
    ]

# ------------------------
# API endpoints
# ------------------------

@app.post("/search-text")
def search_text(query: str):
    query_vec = encode_text(query)
    results = search(query_vec)
    return {"results": results}


@app.post("/search-image")
async def search_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    query_vec = encode_image(image)
    results = search(query_vec)

    return {"results": results}


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