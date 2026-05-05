"""
services/recognition_service.py — Pipeline complet de reconnaissance

Expose :
    recognize_group()     → pipeline complet photo de groupe
    draw_predictions()    → annotation de l'image avec les résultats
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch

from services.detector  import detect_faces, detect_best_face
from services.encoder   import encode_support_set, identify_faces
from utils.preprocessor import bytes_to_pil, pil_to_bytes

# ============================================================
# CONFIG
# ============================================================
FONT_SIZE        = 36
BOX_LINE_WIDTH   = 3
PADDING_RATIO    = 0.15

COLORS = [
    "#FF4444", "#44FF44", "#4444FF", "#FFFF44", "#FF44FF",
    "#44FFFF", "#FF8844", "#88FF44", "#4488FF", "#FF4488",
    "#44FF88", "#8844FF", "#FFAA44", "#44FFAA", "#AA44FF",
    "#FF6644", "#44FF66", "#6644FF", "#FFCC44", "#44FFCC",
    "#FF4466", "#66FF44", "#4466FF", "#FFCC88", "#88CCFF",
    "#FF88CC", "#88FFCC", "#CC88FF", "#FFFF88", "#88FFFF",
]


# ============================================================
# FONT HELPER
# ============================================================
def _load_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont:
    """Charge une police lisible avec fallback automatique."""
    font_paths = [
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def recognize_group(
    group_image_bytes: bytes,
    support_files: list[dict],
    detector,
    encoder,
    ckpt_config: dict,
    device: torch.device,
) -> dict:
    """
    Pipeline complet de reconnaissance faciale sur une photo de groupe.

    Args:
        group_image_bytes : bytes de la photo de groupe
        support_files     : liste de dicts :
                            {"name": str, "image_bytes": bytes}
                            une ou plusieurs photos par personne
        detector          : modèle YOLO chargé
        encoder           : ProtoNetEncoder chargé
        ckpt_config       : config du checkpoint (embedding_dim, etc.)
        device            : cuda ou cpu

    Returns:
        {
            "n_faces"     : int,
            "predictions" : [{"name", "confidence", "distance"}, ...],
            "boxes"       : [(x1, y1, x2, y2), ...],
            "success"     : bool,
            "message"     : str,
        }
    """
    # --- 1. Charger la photo de groupe ---
    group_image = bytes_to_pil(group_image_bytes)

    # --- 2. Détecter tous les visages dans la photo de groupe ---
    boxes, group_crops = detect_faces(group_image, detector)

    if not boxes:
        return {
            "n_faces":     0,
            "predictions": [],
            "boxes":       [],
            "success":     False,
            "message":     "Aucun visage détecté dans la photo de groupe.",
        }

    # --- 3. Construire le support set ---
    # Regrouper les photos par nom (plusieurs photos possible par personne)
    name_to_crops: dict[str, list[Image.Image]] = {}

    for file_info in support_files:
        name        = file_info["name"]
        image_bytes = file_info["image_bytes"]

        ref_image = bytes_to_pil(image_bytes)

        # Détecter le meilleur visage dans la photo de référence
        face_crop, detected = detect_best_face(ref_image, detector)

        if name not in name_to_crops:
            name_to_crops[name] = []
        name_to_crops[name].append(face_crop)

    if not name_to_crops:
        return {
            "n_faces":     len(boxes),
            "predictions": [],
            "boxes":       boxes,
            "success":     False,
            "message":     "Aucune photo de référence valide fournie.",
        }

    # Convertir en format attendu par encode_support_set
    support_data = [
        {"name": name, "crops": crops}
        for name, crops in name_to_crops.items()
    ]

    # --- 4. Calculer les prototypes ---
    names, prototypes = encode_support_set(support_data, encoder, device)

    # --- 5. Identifier chaque visage ---
    predictions = identify_faces(
        query_crops=group_crops,
        names=names,
        prototypes=prototypes,
        encoder=encoder,
        device=device,
    )

    return {
        "n_faces":     len(boxes),
        "predictions": predictions,
        "boxes":       boxes,
        "success":     True,
        "message":     f"{len(boxes)} visage(s) détecté(s) et identifié(s).",
    }


# ============================================================
# ANNOTATION DE L'IMAGE
# ============================================================
def draw_predictions(
    group_image_bytes: bytes,
    boxes: list[tuple],
    predictions: list[dict],
) -> bytes:
    """
    Dessine les boîtes et labels sur la photo de groupe.

    Args:
        group_image_bytes : bytes de la photo de groupe originale
        boxes             : liste de tuples (x1, y1, x2, y2)
        predictions       : liste de dicts {"name", "confidence", "distance"}

    Returns:
        bytes de l'image annotée (JPEG)
    """
    image = bytes_to_pil(group_image_bytes)
    draw  = ImageDraw.Draw(image)
    font  = _load_font(FONT_SIZE)

    # Mapping nom → couleur (une couleur par identité unique)
    unique_names  = sorted(set(p["name"] for p in predictions))
    name_to_color = {
        name: COLORS[i % len(COLORS)]
        for i, name in enumerate(unique_names)
    }

    for box, pred in zip(boxes, predictions):
        x1, y1, x2, y2 = box
        name            = pred["name"]
        confidence      = pred["confidence"]
        color           = name_to_color.get(name, "#FFFFFF")

        # Épaisseur de la boîte proportionnelle à la taille du visage
        box_w = max(BOX_LINE_WIDTH, (x2 - x1) // 80)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_w)

        # Label : nom + confiance
        label     = f"{name}  {confidence:.0%}"
        bbox_text = draw.textbbox((x1, y1), label, font=font)
        text_w    = bbox_text[2] - bbox_text[0]
        text_h    = bbox_text[3] - bbox_text[1]
        padding   = 6

        # Position au-dessus de la boîte (ou en dessous si trop haut)
        label_y = y1 - text_h - padding * 2
        if label_y < 0:
            label_y = y2 + padding

        # Fond coloré derrière le label
        draw.rectangle(
            [x1 - padding,
             label_y - padding,
             x1 + text_w + padding,
             label_y + text_h + padding],
            fill=color,
        )

        # Texte blanc
        draw.text((x1, label_y), label, fill="white", font=font)

    # Watermark discret
    try:
        font_small = _load_font(20)
    except Exception:
        font_small = ImageFont.load_default()

    w, h = image.size
    draw.text(
        (10, h - 30),
        f"ProtoNet ResNet-18  |  {len(predictions)} visage(s)",
        fill="#888888",
        font=font_small,
    )

    return pil_to_bytes(image, format="JPEG", quality=95)