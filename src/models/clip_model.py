from langchain_astradb import AstraDBVectorStore
from langchain.schema import Document
import os
import torch
import clip
from PIL import Image
from langchain.embeddings.base import Embeddings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="clip")


device = "cuda" if torch.cuda.is_available() else "cpu"
# Load and save weights locally
model, preprocess = clip.load(
    "ViT-B/32",
    download_root="C:/models/clip_weights"  # <-- this folder will store weights
)

class CLIPEmbeddings(Embeddings):
    def __init__(self, model, preprocess, device):
        self.model = model
        self.preprocess = preprocess
        self.device = device

    def embed_query(self, text: str = None, **kwargs):
        if text is None:
            raise ValueError("No text provided to embed_query")

        if os.path.exists(text) and text.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(text).convert("RGB")
            return self.embed_image(image)
        else:
            return self.embed_text(text)

    def embed_documents(self, texts_or_paths):
        """
        Handle list of text or image paths
        """
        embeddings = []
        for item in texts_or_paths:
            embeddings.append(self.embed_query(item))
        return embeddings

    def embed_text(self, text: str):
        """Embed text only"""
        text_features = clip.tokenize([text], truncate=True).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(text_features).cpu().numpy()[0]
        return emb.tolist()

    def embed_image(self, image: Image.Image):
        """Embed a PIL image"""
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(image_tensor).cpu().numpy()[0]
        return emb.tolist()


# Initialize embeddings
clip_embeddings = CLIPEmbeddings(model, preprocess, device)