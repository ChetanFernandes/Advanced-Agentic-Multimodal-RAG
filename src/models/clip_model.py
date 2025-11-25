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
model = None
preprocess = None


def load_clip_model():
    """
    Lazy-load the CLIP model only once per worker.
    Prevents heavy model loading at import time.
    """
    global model, preprocess

    if model is None or preprocess is None:
        print("Loading CLIP model...")
        '''
        model_loaded, preprocess_loaded = clip.load(
            "ViT-B/32",
            download_root="C:/models/clip_weights"
        )
        '''
        
        model_loaded, preprocess_loaded = clip.load("ViT-B/32",download_root="/app/clip_weights")

        model_loaded = model_loaded.to(device)
        model, preprocess = model_loaded, preprocess_loaded

    return model, preprocess

class CLIPEmbeddings(Embeddings):
    def __init__(self, device):
        self.device = device

    def _ensure_model(self):
        """Make sure the model is loaded before use"""
        load_clip_model()  # loads only once
        return model, preprocess

    def embed_query(self, text: str = None, **kwargs):
        model_loaded, preprocess_loaded = self._ensure_model()

        if text is None:
            raise ValueError("No text provided to embed_query")

        if os.path.exists(text) and text.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(text).convert("RGB")
            return self.embed_image(image, model_loaded, preprocess_loaded)
        else:
            return self.embed_text(text, model_loaded)

    def embed_documents(self, texts_or_paths):
        embeddings = []
        for item in texts_or_paths:
            embeddings.append(self.embed_query(item))
        return embeddings

    def embed_text(self, text: str, model_loaded):
        text_features = clip.tokenize([text], truncate=True).to(self.device)
        with torch.no_grad():
            emb = model_loaded.encode_text(text_features).cpu().numpy()[0]
        return emb.tolist()

    def embed_image(self, image: Image.Image, model_loaded, preprocess_loaded):
        image_tensor = preprocess_loaded(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = model_loaded.encode_image(image_tensor).cpu().numpy()[0]
        return emb.tolist()


# Initialize embeddings object (WITHOUT loading model)
clip_embeddings = CLIPEmbeddings(device)