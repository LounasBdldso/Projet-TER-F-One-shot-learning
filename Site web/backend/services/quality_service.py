"""
services/quality_service.py — Évaluation de la qualité des images

Combine :
  1. Netteté     (gradient Sobel — robuste à la compression JPEG)
  2. Luminosité  (moyenne des pixels en niveaux de gris)
  3. Taille      (% de l'image occupé par le visage)
  4. GraFIQs     (score calibré sur dataset de référence)

Seuils GraFIQs calibrés empiriquement sur dataset CelebA :
  GRAFIQS_MAX_THEORIQUE = 0.00085  → score brut parfait  = 100/100
  GRAFIQS_SEUIL_REJET   = 0.00115  → score brut limite   =   0/100
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.autograd as autograd
from PIL import Image
from torchvision import transforms

# ============================================================
# CONFIG
# ============================================================
GRAFIQS_DIR           = Path("models/grafiqs")
WEIGHTS_PATH          = Path("models/weights/resnet50_webface_arcface.pth")

# Seuils calibrés sur dataset CelebA (100 images annotées manuellement)
GRAFIQS_MAX_THEORIQUE = 0.00085   # score brut parfait  → 100/100
GRAFIQS_SEUIL_REJET   = 0.00115   # score brut limite   →   0/100

# Poids du score final
WEIGHTS_FINAL = {
    "sharpness":   0.20,
    "brightness":  0.15,
    "face_size":   0.15,
    "grafiqs":     0.50,
}

# Seuils de recommandation
SCORE_GOOD       = 70
SCORE_ACCEPTABLE = 40

# Transform GraFIQs
GRAFIQS_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ============================================================
# CHARGEMENT GraFIQs
# ============================================================
def load_quality_model():
    """
    Charge le backbone iResNet50 GraFIQs.
    Appelé une seule fois au démarrage via lifespan dans main.py.
    """
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Poids GraFIQs introuvables : {WEIGHTS_PATH}\n"
            f"Place resnet50_webface_arcface.pth dans models/weights/"
        )

    sys.path.insert(0, str(GRAFIQS_DIR))
    from backbones.iresnet import iresnet50
    from backbones.bn import BN_Model

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = iresnet50(num_features=512, dropout=0.4, use_se=False).to(device)
    backbone.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    backbone.return_intermediate = True
    backbone.eval()

    model = BN_Model(backbone, device)
    return model, device


# ============================================================
# 1. MÉTRIQUES CLASSIQUES
# ============================================================
def compute_sharpness(face_pil: Image.Image) -> tuple[float, float]:
    """
    Gradient de Sobel — plus robuste à la compression JPEG que le Laplacien.
    Score normalisé 0-100.
    """
    face_gray = np.array(face_pil.convert("L"))
    sobelx    = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely    = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    raw       = float(magnitude.mean())
    score     = min(100.0, (raw / 50.0) * 100.0)
    return round(score, 2), round(raw, 4)


def compute_brightness(face_pil: Image.Image) -> tuple[float, float]:
    """
    Luminosité moyenne — courbe en cloche centrée sur 130/255.
    Score normalisé 0-100.
    """
    face_gray = np.array(face_pil.convert("L"))
    raw       = float(face_gray.mean())
    deviation = abs(raw - 130.0)
    score     = max(0.0, 100.0 - (deviation / 80.0) * 100.0)
    return round(score, 2), round(raw, 2)


def compute_face_size(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int]
) -> tuple[float, float]:
    """
    Pourcentage de l'image occupé par le visage.
    Idéal : >= 10% de l'image. Score normalisé 0-100.
    """
    x1, y1, x2, y2   = box
    img_w, img_h      = image_size
    face_area         = (x2 - x1) * (y2 - y1)
    image_area        = img_w * img_h
    ratio             = face_area / image_area
    score             = min(100.0, (ratio / 0.10) * 100.0)
    return round(score, 2), round(ratio * 100, 2)


# ============================================================
# 2. SCORE GraFIQs — NORMALISATION ABSOLUE CALIBRÉE
# ============================================================
def compute_grafiqs_raw(
    face_pil: Image.Image,
    grafiqs_model,
    device: torch.device,
) -> float:
    """
    Calcule le score brut GraFIQs (somme des gradient magnitudes).
    Score brut faible = bonne qualité.
    Score brut élevé  = mauvaise qualité.
    """
    img_tensor = GRAFIQS_TRANSFORM(face_pil).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)

    bn_score, (emb, block1, block2, block3, block4, bn) = grafiqs_model.get_BN(img_tensor)

    grads = autograd.grad(
        outputs=bn_score,
        inputs=[img_tensor, block1, block2, block3, block4],
        allow_unused=True,
    )

    grad_block4 = grads[4]
    if grad_block4 is not None:
        raw = float(torch.abs(grad_block4[0].cpu()).sum())
    else:
        raw = float(torch.abs(grads[0][0].cpu()).sum())

    return raw


def normalize_grafiqs(raw_score: float) -> float:
    """
    Normalisation absolue basée sur les seuils calibrés.

    Principe :
      raw <= GRAFIQS_MAX_THEORIQUE → score = 100  (meilleure qualité possible)
      raw >= GRAFIQS_SEUIL_REJET  → score =   0  (image rejetée)
      entre les deux              → interpolation linéaire inversée

    Seuils calibrés sur 100 images CelebA annotées manuellement.
    """
    if raw_score <= GRAFIQS_MAX_THEORIQUE:
        return 100.0

    if raw_score >= GRAFIQS_SEUIL_REJET:
        return 0.0

    # Interpolation linéaire inversée entre les deux seuils
    ratio = (raw_score - GRAFIQS_MAX_THEORIQUE) / \
            (GRAFIQS_SEUIL_REJET - GRAFIQS_MAX_THEORIQUE)
    score = (1.0 - ratio) * 100.0
    return round(score, 2)


# ============================================================
# 3. SCORE FINAL
# ============================================================
def compute_final_score(
    sharpness: float,
    brightness: float,
    face_size: float,
    grafiqs: float,
) -> float:
    score = (
        WEIGHTS_FINAL["sharpness"]  * sharpness  +
        WEIGHTS_FINAL["brightness"] * brightness +
        WEIGHTS_FINAL["face_size"]  * face_size  +
        WEIGHTS_FINAL["grafiqs"]    * grafiqs
    )
    return round(score, 2)


def get_recommendation(score: float) -> str:
    if score >= SCORE_GOOD:
        return "bonne"
    elif score >= SCORE_ACCEPTABLE:
        return "acceptable"
    else:
        return "a_remplacer"


# ============================================================
# 4. PIPELINE PRINCIPAL
# ============================================================
def assess_image_quality(
    face_pil: Image.Image,
    box: tuple[int, int, int, int] | None,
    image_size: tuple[int, int],
    grafiqs_model,
    device: torch.device,
) -> dict:
    """
    Pipeline complet de qualité pour une image de référence.

    Args:
        face_pil     : image PIL du visage cropé
        box          : (x1, y1, x2, y2) de la détection YOLO
                       None si aucun visage détecté (fallback image entière)
        image_size   : (width, height) de l'image originale
        grafiqs_model: modèle GraFIQs chargé
        device       : cuda ou cpu

    Returns:
        dict avec tous les scores et la recommandation finale :
        {
            "final_score"     : float,
            "recommendation"  : "bonne" | "acceptable" | "a_remplacer",
            "face_detected"   : bool,
            "sharpness_score" : float,
            "sharpness_raw"   : float,
            "brightness_score": float,
            "brightness_raw"  : float,
            "face_size_score" : float,
            "face_size_pct"   : float,
            "grafiqs_score"   : float,
            "grafiqs_raw"     : float,
        }
    """
    face_detected = box is not None

    # Si pas de visage détecté → box couvre l'image entière
    if box is None:
        box = (0, 0, image_size[0], image_size[1])

    # Métriques classiques
    sharpness_score,  sharpness_raw  = compute_sharpness(face_pil)
    brightness_score, brightness_raw = compute_brightness(face_pil)
    face_size_score,  face_size_pct  = compute_face_size(box, image_size)

    # Score GraFIQs
    grafiqs_raw   = compute_grafiqs_raw(face_pil, grafiqs_model, device)
    grafiqs_score = normalize_grafiqs(grafiqs_raw)

    # Score final
    final_score    = compute_final_score(
        sharpness_score, brightness_score,
        face_size_score, grafiqs_score,
    )
    recommendation = get_recommendation(final_score)

    return {
        "final_score":      final_score,
        "recommendation":   recommendation,
        "face_detected":    face_detected,
        "sharpness_score":  sharpness_score,
        "sharpness_raw":    sharpness_raw,
        "brightness_score": brightness_score,
        "brightness_raw":   brightness_raw,
        "face_size_score":  face_size_score,
        "face_size_pct":    face_size_pct,
        "grafiqs_score":    grafiqs_score,
        "grafiqs_raw":      grafiqs_raw,
    }