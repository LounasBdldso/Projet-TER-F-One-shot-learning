"""
utils/preprocessor.py — Traitement des images

Transforme les images reçues par l'API (bytes) en tenseurs
prêts à être utilisés par les modèles.
"""

import io
from PIL import Image
import torch
from torchvision import transforms


# ============================================================
# TRANSFORMS
# ============================================================
# Transform standard pour l'inférence ProtoNet
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Transform pour GraFIQs (normalisation différente)
GRAFIQS_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])


# ============================================================
# FONCTIONS PRINCIPALES
# ============================================================
def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """
    Convertit des bytes en image PIL RGB.
    Compatible avec tous les formats : JPEG, PNG, WEBP...
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def pil_to_tensor(image: Image.Image, transform=None) -> torch.Tensor:
    """
    Convertit une image PIL en tenseur normalisé.
    Utilise EVAL_TRANSFORM par défaut.
    """
    if transform is None:
        transform = EVAL_TRANSFORM
    return transform(image)


def preprocess_image(image_bytes: bytes) -> tuple[Image.Image, torch.Tensor]:
    """
    Pipeline complet : bytes → PIL + tenseur normalisé.

    Retourne :
        pil_image : Image PIL originale (pour YOLO et l'annotation)
        tensor    : tenseur normalisé prêt pour l'encodeur
    """
    pil_image = bytes_to_pil(image_bytes)
    tensor    = pil_to_tensor(pil_image)
    return pil_image, tensor


def preprocess_for_grafiqs(image: Image.Image) -> torch.Tensor:
    """
    Prépare une image PIL pour GraFIQs.
    Normalisation différente de ProtoNet.
    """
    return GRAFIQS_TRANSFORM(image)


def pil_to_bytes(image: Image.Image, format: str = "JPEG", quality: int = 95) -> bytes:
    """
    Convertit une image PIL en bytes.
    Utilisé pour renvoyer l'image annotée au frontend.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality)
    buffer.seek(0)
    return buffer.getvalue()