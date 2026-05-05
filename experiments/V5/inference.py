import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ultralytics import YOLO

import config
from model import ProtoNetEncoder, cosine_distance  # FIX : cosine_distance au lieu de euclidean_distance

# ============================================================
# PARAMÈTRES
# ============================================================
SUPPORT_SET_DIR      = "./classe"
TEST_IMAGES_DIR      = "./Test"
RESULTS_DIR          = "./results"
CONFIDENCE_THRESHOLD = 0.3
SUPPORT_CONFIDENCE   = 0.3
SCALE                = 20.0   # même scale que pendant l'entraînement


# ============================================================
# 1. SUPPORT SET
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
    """
    Charge le support set depuis le dossier ./classe.
    Chaque fichier = 1 personne (nom extrait du filename).
    Plusieurs photos par personne supportées.
    """
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
    print(f"  Fallback (image entière)  : {detection_stats['fallback']}")
    return names, all_images


# ============================================================
# 2. DÉTECTION YOLO SUR LA PHOTO DE GROUPE
# ============================================================
def detect_faces(image_path, yolo_model, confidence=0.3):
    results = yolo_model(image_path, conf=confidence, verbose=False)
    image = Image.open(image_path).convert("RGB")
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
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)

        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w, x2 + pad_x)
        py2 = min(h, y2 + pad_y)

        face = image.crop((px1, py1, px2, py2))
        face_tensors.append(transform(face))
        padded_boxes.append((px1, py1, px2, py2))

    return torch.stack(face_tensors), padded_boxes


# ============================================================
# 3. IDENTIFICATION — Distance cosinus + Assignation unique
# ============================================================
@torch.no_grad()
def identify_faces(encoder, support_names, support_images, query_faces, device):
    """
    Identifie chaque visage détecté via ProtoNet + distance cosinus.

    Pipeline :
    1. Calcul des prototypes (moyenne des embeddings normalisés L2)
    2. Calcul des embeddings query (normalisés L2 via forward())
    3. Distance cosinus entre chaque query et chaque prototype
    4. Softmax sur les distances → confiances
    5. Assignation unique gloutonne (greedy matching)
    """
    encoder.eval()

    # --- 1. Prototypes ---
    prototypes = []
    for person_images in support_images:
        person_images = person_images.to(device)
        embeddings = encoder(person_images)          # déjà normalisé L2 via forward()
        prototype = embeddings.mean(dim=0)
        prototypes.append(prototype)

    prototypes = torch.stack(prototypes)
    prototypes = F.normalize(prototypes, p=2, dim=1) # renormaliser après la moyenne

    # --- 2. Embeddings query ---
    query_faces = query_faces.to(device)
    query_embeddings = encoder(query_faces)          # déjà normalisé L2 via forward()

    # --- 3. Distance cosinus ---
    # cosine_distance retourne 1 - similarité → [0, 2]
    # 0 = identique, 2 = opposé
    distances = cosine_distance(query_embeddings, prototypes)  # [N_queries, N_support]

    # --- 4. Confiances via softmax sur distances scalées ---
    confidences = torch.softmax(-distances * SCALE, dim=1)     # [N_queries, N_support]

    n_queries = distances.shape[0]
    n_support = distances.shape[1]

    # --- 5. Assignation gloutonne ---
    # Lister tous les couples (query, support) triés par confiance décroissante
    candidates = []
    for face_idx in range(n_queries):
        for proto_idx in range(n_support):
            candidates.append({
                "face_idx":   face_idx,
                "proto_idx":  proto_idx,
                "confidence": confidences[face_idx, proto_idx].item(),
                "distance":   distances[face_idx, proto_idx].item(),
            })

    candidates.sort(key=lambda x: x["confidence"], reverse=True)

    assigned_faces = set()
    assigned_names = set()
    predictions = [None] * n_queries

    for cand in candidates:
        face_idx  = cand["face_idx"]
        proto_idx = cand["proto_idx"]

        if face_idx in assigned_faces:
            continue
        if support_names[proto_idx] in assigned_names:
            continue

        predictions[face_idx] = {
            "name":       support_names[proto_idx],
            "distance":   cand["distance"],
            "confidence": cand["confidence"],
        }
        assigned_faces.add(face_idx)
        assigned_names.add(support_names[proto_idx])

        if len(assigned_faces) == n_queries:
            break

    # Visages non assignés (plus d'identités disponibles dans le support set)
    for face_idx in range(n_queries):
        if predictions[face_idx] is None:
            dists   = distances[face_idx]
            min_idx = dists.argmin().item()
            predictions[face_idx] = {
                "name":       "Inconnu",
                "distance":   dists[min_idx].item(),
                "confidence": 0.0,
            }

    return predictions


# ============================================================
# 4. ANNOTATION
# ============================================================
def annotate_image(image, boxes, predictions, output_path):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 100)
    except (IOError, OSError):
        font = ImageFont.load_default()

    colors = [
        "#FF4444", "#44FF44", "#4444FF", "#FFFF44", "#FF44FF",
        "#44FFFF", "#FF8844", "#88FF44", "#4488FF", "#FF4488",
        "#44FF88", "#8844FF", "#FFAA44", "#44FFAA", "#AA44FF",
        "#FF6644", "#44FF66", "#6644FF", "#FFCC44", "#44FFCC",
    ]
    unique_names = sorted(set(p["name"] for p in predictions))
    name_to_color = {name: colors[i % len(colors)] for i, name in enumerate(unique_names)}

    for box, pred in zip(boxes, predictions):
        x1, y1, x2, y2 = box
        name       = pred["name"]
        confidence = pred["confidence"]
        color      = name_to_color.get(name, "#FFFFFF")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{name} ({confidence:.0%})"
        bbox  = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
        draw.text((x1, y1 - 25), label, fill="white", font=font)

    image.save(output_path, quality=95)


# ============================================================
# 5. RÉSUMÉ CONSOLE
# ============================================================
def print_summary(predictions, filename):
    print(f"\n  Résultats pour {filename} :")
    for i, pred in enumerate(predictions):
        icon = "✓" if pred["confidence"] > 0.3 else "?"
        print(f"    {icon} Visage {i+1:2d} → {pred['name']:25s} "
              f"(confiance: {pred['confidence']:.0%}, distance: {pred['distance']:.3f})")

    from collections import Counter
    counts     = Counter(p["name"] for p in predictions)
    duplicates = {n: c for n, c in counts.items() if c > 1}

    if duplicates:
        print(f"\n    Doublons restants :")
        for name, count in duplicates.items():
            print(f"      - {name} : {count} fois")
    else:
        print(f"\n    Aucun doublon — chaque identité est unique !")

    print(f"    Identités uniques : {len(counts)}/{len(predictions)}")


# ============================================================
# MAIN
# ============================================================
def main():
    # --- Vérifications ---
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(SUPPORT_SET_DIR):
        print(f"ERREUR : Support set introuvable : {SUPPORT_SET_DIR}")
        sys.exit(1)

    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"ERREUR : Dossier de test introuvable : {TEST_IMAGES_DIR}")
        sys.exit(1)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "protonet_resnet_webface.pth")
    if not os.path.exists(checkpoint_path):
        print(f"ERREUR : Checkpoint introuvable : {checkpoint_path}")
        sys.exit(1)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # --- Transform ---
    eval_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- YOLO ---
    print("\n[1/3] Chargement de YOLO...")
    yolo_model = YOLO(config.YOLO_MODEL_PATH)

    # --- Modèle ProtoNet ---
    print("\n[2/3] Chargement du modèle ProtoNet...")
    checkpoint    = torch.load(checkpoint_path, map_location=device)
    ckpt_config   = checkpoint["config"]
    backbone_name = ckpt_config.get("backbone", "resnet18")

    encoder = ProtoNetEncoder(
        embedding_dim=ckpt_config["embedding_dim"],
        backbone_name=backbone_name,
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()
    print(f"Modèle chargé — Backbone: {backbone_name} | "
          f"Embedding: {ckpt_config['embedding_dim']} | "
          f"Époque: {checkpoint['epoch']} | "
          f"Val Acc: {checkpoint['val_acc']:.4f}")

    # --- Support set ---
    print("\n[3/3] Chargement du support set...")
    support_names, support_images = load_support_set(
        SUPPORT_SET_DIR, yolo_model, eval_transform
    )

    # --- Inférence ---
    print("\n" + "=" * 60)
    print(f"INFÉRENCE SUR : {TEST_IMAGES_DIR}")
    print("=" * 60)

    test_files = [
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not test_files:
        print(f"Aucune image trouvée dans {TEST_IMAGES_DIR}.")
        sys.exit(0)

    for filename in sorted(test_files):
        img_path    = os.path.join(TEST_IMAGES_DIR, filename)
        output_path = os.path.join(RESULTS_DIR, f"result_{filename}")

        print(f"\nTraitement : {filename}...")

        # Détection YOLO
        class_image, boxes, confidences = detect_faces(
            img_path, yolo_model, CONFIDENCE_THRESHOLD
        )

        if not boxes:
            print(f"  Aucun visage détecté — image ignorée.")
            continue

        print(f"  {len(boxes)} visage(s) détecté(s).")

        # Crop + identification
        query_faces, padded_boxes = crop_faces(class_image, boxes, eval_transform)
        predictions = identify_faces(
            encoder, support_names, support_images, query_faces, device
        )

        # Résumé console
        print_summary(predictions, filename)

        # Annotation + sauvegarde
        class_image_clean = Image.open(img_path).convert("RGB")
        annotate_image(class_image_clean, padded_boxes, predictions, output_path)
        print(f"  Sauvegardé : {output_path}")

    print("\n" + "=" * 60)
    print(f"Terminé ! Résultats dans : {RESULTS_DIR}")


if __name__ == "__main__":
    main()