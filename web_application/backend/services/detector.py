"""
services/detector.py — Détection des visages avec YOLO

Expose :
    load_detector()  → charge le modèle YOLO
    detect_faces()   → détecte et croppe les visages depuis une image PIL
"""

from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
YOLO_MODEL_PATH  = Path("models/weights/yolov8n-face.pt")
YOLO_CONFIDENCE  = 0.15   # seuil bas pour ne rater aucun visage
PADDING_RATIO    = 0.15   # padding autour du visage cropé


# ============================================================
# CHARGEMENT
# ============================================================
def load_detector() -> YOLO:
    """
    Charge le modèle YOLO depuis le dossier weights.
    Appelé une seule fois au démarrage via lifespan dans main.py.
    """
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle YOLO introuvable : {YOLO_MODEL_PATH}\n"
            f"Place yolov8n-face.pt dans models/weights/"
        )
    model = YOLO(str(YOLO_MODEL_PATH))
    return model


# ============================================================
# DÉTECTION
# ============================================================
def detect_faces(
    image: Image.Image,
    yolo_model: YOLO,
    confidence: float = YOLO_CONFIDENCE,
    padding_ratio: float = PADDING_RATIO,
) -> tuple[list[tuple], list[Image.Image]]:
    """
    Détecte tous les visages dans une image PIL.

    Args:
        image         : image PIL RGB (photo de groupe ou de référence)
        yolo_model    : modèle YOLO chargé
        confidence    : seuil de confiance YOLO
        padding_ratio : % de padding autour de chaque visage

    Returns:
        boxes      : liste de tuples (x1, y1, x2, y2) — boîtes avec padding
        face_crops : liste d'images PIL croppées (une par visage)
    """
    results = yolo_model(image, conf=confidence, verbose=False)

    w, h   = image.size
    boxes  = []
    crops  = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Padding autour du visage
            bw, bh = x2 - x1, y2 - y1
            pad_x  = int(bw * padding_ratio)
            pad_y  = int(bh * padding_ratio)

            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(w, x2 + pad_x)
            py2 = min(h, y2 + pad_y)

            boxes.append((px1, py1, px2, py2))
            crops.append(image.crop((px1, py1, px2, py2)))

    return boxes, crops


def detect_best_face(
    image: Image.Image,
    yolo_model: YOLO,
    confidence: float = YOLO_CONFIDENCE,
    padding_ratio: float = PADDING_RATIO,
) -> tuple[Image.Image, bool]:
    """
    Détecte le visage le plus grand dans une image.
    Utilisé pour les photos de référence (1 personne par photo).

    Returns:
        face_crop     : image PIL croppée sur le visage
        face_detected : True si un visage a été détecté, False si fallback
    """
    results   = yolo_model(image, conf=confidence, verbose=False)
    best_box  = None
    best_area = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box  = (int(x1), int(y1), int(x2), int(y2))

    if best_box is None:
        # Fallback : image entière
        return image, False

    w, h = image.size
    x1, y1, x2, y2 = best_box
    bw, bh = x2 - x1, y2 - y1
    pad_x  = int(bw * padding_ratio)
    pad_y  = int(bh * padding_ratio)

    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(w, x2 + pad_x)
    py2 = min(h, y2 + pad_y)

    return image.crop((px1, py1, px2, py2)), True