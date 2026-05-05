import os
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ultralytics import YOLO

import config
# On suppose que tu as bien implémenté cosine_distance dans model.py (vu qu'on en a parlé)
# Si ce n'est pas le cas, garde euclidean_distance. Ici j'ai mis la version cosinus pour profiter du Scale.
from model import ProtoNetEncoder, euclidean_distance
# ============================================================
# PARAMÈTRES
# ============================================================
SUPPORT_SET_DIR = "./classe"

# --- NOUVEAUX PARAMÈTRES POUR LE DOSSIER DE TEST ---
TEST_IMAGES_DIR = "./Test"       # Dossier contenant toutes tes photos de groupe à tester
RESULTS_DIR = "./results"        # Dossier de destination pour les photos annotées

CONFIDENCE_THRESHOLD = 0.3       # Seuil YOLO pour les photos de groupe
SUPPORT_CONFIDENCE = 0.3         # Seuil YOLO pour les photos individuelles


# ============================================================
# 1. SUPPORT SET (inchangé)
# ============================================================
def extract_name_from_filename(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts)


def crop_face_from_photo(image_path, yolo_model, confidence=0.3):
    results = yolo_model(image_path, conf=confidence, verbose=False)
    image = Image.open(image_path).convert("RGB")
    best_box = None
    best_area = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (int(x1), int(y1), int(x2), int(y2))

    if best_box is None:
        return image, False

    w, h = image.size
    x1, y1, x2, y2 = best_box
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * 0.15)
    pad_y = int(bh * 0.15)

    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(w, x2 + pad_x)
    py2 = min(h, y2 + pad_y)

    face_crop = image.crop((px1, py1, px2, py2))
    return face_crop, True


def load_support_set(support_dir, yolo_model, transform):
    name_to_paths = {}
    for filename in sorted(os.listdir(support_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        filepath = os.path.join(support_dir, filename)
        name = extract_name_from_filename(filename)
        if name not in name_to_paths:
            name_to_paths[name] = []
        name_to_paths[name].append(filepath)

    names = []
    all_images = []
    detection_stats = {"detected": 0, "fallback": 0}

    for name in sorted(name_to_paths.keys()):
        paths = name_to_paths[name]
        person_images = []
        for path in paths:
            face_crop, detected = crop_face_from_photo(path, yolo_model, SUPPORT_CONFIDENCE)
            if detected:
                detection_stats["detected"] += 1
            else:
                detection_stats["fallback"] += 1
            img_tensor = transform(face_crop)
            person_images.append(img_tensor)

        names.append(name)
        all_images.append(torch.stack(person_images))

    print(f"\nSupport set : {len(names)} personnes")
    print(f"  Visages détectés par YOLO : {detection_stats['detected']}")
    return names, all_images


# ============================================================
# 2. DÉTECTION YOLO SUR LA PHOTO DE CLASSE (inchangé)
# ============================================================
def detect_faces(class_photo_path, yolo_model, confidence=0.3):
    results = yolo_model(class_photo_path, conf=confidence, verbose=False)
    image = Image.open(class_photo_path).convert("RGB")
    boxes = []
    confidences = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
            confidences.append(conf)

    return image, boxes, confidences


def crop_faces(image, boxes, transform, padding_ratio=0.15):
    w, h = image.size
    face_tensors = []
    padded_boxes = []

    for (x1, y1, x2, y2) in boxes:
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * padding_ratio), int(bh * padding_ratio)

        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w, x2 + pad_x)
        py2 = min(h, y2 + pad_y)

        face = image.crop((px1, py1, px2, py2))
        face_tensors.append(transform(face))
        padded_boxes.append((px1, py1, px2, py2))

    return torch.stack(face_tensors), padded_boxes


# ============================================================
# 3. IDENTIFICATION (Version Euclidienne classique)
# ============================================================
@torch.no_grad()
def identify_faces(encoder, support_names, support_images, query_faces, device):
    encoder.eval()

    prototypes = []
    for person_images in support_images:
        person_images = person_images.to(device)
        embeddings = encoder(person_images)
        prototypes.append(embeddings.mean(dim=0))
    prototypes = torch.stack(prototypes)

    query_faces = query_faces.to(device)
    query_embeddings = encoder(query_faces)

    # Utilisation de la distance Euclidienne (ton modèle actuel)
    distances = euclidean_distance(query_embeddings, prototypes)

    predictions = []
    for i in range(distances.shape[0]):
        dists = distances[i]
        min_dist, min_idx = dists.min(dim=0)
        
        # Softmax classique
        confidence_scores = torch.softmax(-dists, dim=0)

        predictions.append({
            "name": support_names[min_idx.item()],
            "distance": min_dist.item(),
            "confidence": confidence_scores[min_idx].item(),
        })

    return predictions


# ============================================================
# 4. ANNOTATION (inchangé)
# ============================================================
def annotate_image(image, boxes, predictions, output_path):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except (IOError, OSError):
        font = ImageFont.load_default()

    colors = [
        "#FF4444", "#44FF44", "#4444FF", "#FFFF44", "#FF44FF",
        "#44FFFF", "#FF8844", "#88FF44", "#4488FF", "#FF4488",
        "#44FF88", "#8844FF", "#FFAA44", "#44FFAA", "#AA44FF"
    ]
    unique_names = sorted(set(p["name"] for p in predictions))
    name_to_color = {name: colors[i % len(colors)] for i, name in enumerate(unique_names)}

    for box, pred in zip(boxes, predictions):
        x1, y1, x2, y2 = box
        name = pred["name"]
        confidence = pred["confidence"]
        color = name_to_color[name]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{name} ({confidence:.0%})"
        bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
        draw.text((x1, y1 - 25), label, fill="white", font=font)

    image.save(output_path, quality=95)


# ============================================================
# MAIN (Modifié pour la boucle sur le dossier)
# ============================================================
def main():
    # --- Création du dossier de résultats s'il n'existe pas ---
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"📁 Création du dossier : {RESULTS_DIR}")

    if not os.path.exists(SUPPORT_SET_DIR):
        print(f"ERREUR : Support set introuvable : {SUPPORT_SET_DIR}")
        sys.exit(1)

    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"ERREUR : Dossier de test introuvable : {TEST_IMAGES_DIR}")
        print("Veuillez créer le dossier './Test' et y placer vos photos.")
        sys.exit(1)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "protonet_resnet18_webface.pth")
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initialisation sur : {device}")

    eval_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print("\n[1/3] Chargement de YOLO...")
    yolo_model = YOLO(config.YOLO_MODEL_PATH)

    print("\n[2/3] Chargement du Modèle ProtoNet...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_config = checkpoint["config"]
    backbone_name = ckpt_config.get("backbone", "resnet50")

    encoder = ProtoNetEncoder(
        embedding_dim=ckpt_config["embedding_dim"],
        backbone_name=backbone_name
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()
    print(f"✓ Modèle chargé (Époque {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.4f})")

    print("\n[3/3] Chargement du Support Set (Les identités de référence)...")
    support_names, support_images = load_support_set(SUPPORT_SET_DIR, yolo_model, eval_transform)

    # --- BOUCLE SUR LE DOSSIER DE TEST ---
    print("\n" + "=" * 60)
    print(f"LANCEMENT DE L'INFÉRENCE BATCH SUR : {TEST_IMAGES_DIR}")
    print("=" * 60)

    # On récupère toutes les images dans le dossier Test
    test_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    if not test_files:
        print(f"Aucune image trouvée dans {TEST_IMAGES_DIR}.")
        sys.exit(0)

    for filename in sorted(test_files):
        img_path = os.path.join(TEST_IMAGES_DIR, filename)
        output_filename = f"result_{filename}"
        output_path = os.path.join(RESULTS_DIR, output_filename)

        print(f"\n📸 Traitement de : {filename}...")
        
        # 1. Détection YOLO
        class_image, boxes, confidences = detect_faces(img_path, yolo_model, CONFIDENCE_THRESHOLD)
        
        if not boxes:
            print(f"  ⚠ Aucun visage détecté sur {filename}. Image ignorée.")
            continue
            
        print(f"  ✓ {len(boxes)} visages trouvés.")

        # 2. Découpage
        query_faces, padded_boxes = crop_faces(class_image, boxes, eval_transform)

        # 3. Identification ProtoNet
        predictions = identify_faces(encoder, support_names, support_images, query_faces, device)

        # 4. Sauvegarde
        class_image_clean = Image.open(img_path).convert("RGB")
        annotate_image(class_image_clean, padded_boxes, predictions, output_path)
        print(f"  💾 Sauvegardé sous : {output_path}")

    print("\n" + "=" * 60)
    print(f"🎉 TRAITEMENT TERMINÉ ! Toutes les images sont dans {RESULTS_DIR}")

if __name__ == "__main__":
    main()