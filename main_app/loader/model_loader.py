from tqdm import tqdm

# ============================================
# LOAD OPENCLIP
# ============================================
# loader/model_loader.py

import open_clip
import easyocr

def load_clip():

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k"
        )
    )

    model.eval()

    return model, preprocess

def load_ocr():

    reader = easyocr.Reader(['en'])

    return reader