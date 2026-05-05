import os
import sys
import math
import numpy as np
import csv
import torch
import torch.autograd as autograd
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import cv2

# --- PARAMÈTRES ---
DATASET_DIR = "./calibration_dataset" 
GRAFIQS_DIR = "./GraFIQs"
WEIGHTS_PATH = "./GraFIQs/weights/resnet50_webface_arcface.pth"
YOLO_MODEL_PATH = "yolov8n-face.pt"
CSV_OUTPUT = "./resultats_calibration.csv"

# ==========================================
# 1. INITIALISATION DES MODÈLES
# ==========================================
def load_grafiqs_model(weights_path, device):
    sys.path.insert(0, GRAFIQS_DIR)
    from backbones.iresnet import iresnet50
    from backbones.bn import BN_Model

    backbone = iresnet50(num_features=512, dropout=0.4, use_se=False).to(device)
    backbone.load_state_dict(torch.load(weights_path, map_location=device))
    backbone.return_intermediate = True
    backbone.eval()
    return BN_Model(backbone, device)

grafiqs_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def extract_and_align_face(image_path, yolo_model):
    """Extraction et alignement (identique à ton système actuel)."""
    image = Image.open(image_path).convert("RGB")
    results = yolo_model(image_path, conf=0.15, verbose=False)
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        return image 
        
    result = results[0]
    best_box, best_area, best_kpts = None, 0, None
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_box = (int(x1), int(y1), int(x2), int(y2))
            if hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints.xy) > i:
                kpts = result.keypoints.xy[i].cpu().numpy()
                if len(kpts) >= 2 and (kpts[0][0] != 0 or kpts[0][1] != 0):
                    best_kpts = kpts

    if best_kpts is not None:
        oeil_gauche, oeil_droit = best_kpts[0], best_kpts[1]
        dx, dy = oeil_droit[0] - oeil_gauche[0], oeil_droit[1] - oeil_gauche[1]
        angle = math.degrees(math.atan2(dy, dx))
        centre_yeux = (int((oeil_gauche[0] + oeil_droit[0]) // 2), int((oeil_gauche[1] + oeil_droit[1]) // 2))
        w_box, h_box = best_box[2] - best_box[0], best_box[3] - best_box[1]
        crop_size = int(max(w_box, h_box) * 1.20)
        M = cv2.getRotationMatrix2D(centre_yeux, angle, 1.0)
        M[0, 2] += (crop_size // 2 - centre_yeux[0])
        M[1, 2] += (crop_size // 2 - centre_yeux[1])
        cv_img = np.array(image)
        aligned_crop = cv2.warpAffine(cv_img, M, (crop_size, crop_size), flags=cv2.INTER_CUBIC)
        return Image.fromarray(aligned_crop)
    else:
        x1, y1, x2, y2 = best_box
        w, h = image.size
        pad_x, pad_y = int((x2-x1)*0.1), int((y2-y1)*0.1)
        return image.crop((max(0, x1-pad_x), max(0, y1-pad_y), min(w, x2+pad_x), min(h, y2+pad_y)))

def compute_grafiqs_raw(grafiqs_model, face_pil, device):
    """Calcul du score brut GraFIQs (plus c'est bas, mieux c'est)."""
    img_tensor = grafiqs_transform(face_pil).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)
    bn_score, (emb, b1, b2, b3, b4, bn) = grafiqs_model.get_BN(img_tensor)
    grads = autograd.grad(outputs=bn_score, inputs=[img_tensor, b1, b2, b3, b4], allow_unused=True)
    
    if grads[4] is not None:
        return float(torch.abs(grads[4][0].cpu()).sum())
    return float(torch.abs(grads[0][0].cpu()).sum())


# ==========================================
# 2. LOGIQUE DE PARCOURS IMBRIQUÉ
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Initialisation des modèles...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    grafiqs_model = load_grafiqs_model(WEIGHTS_PATH, device)

    # Structure pour stocker les résultats : dict de dicts d'images
    # stats_dict["mauvaise"]["blur_heavy"] = [0.0025, 0.0028, ...]
    stats_dict = {}
    toutes_les_images = [] # Pour l'export CSV

    print(f"\n📂 Exploration du dossier : {DATASET_DIR}")
    
    # Parcourir les dossiers principaux (bonne, variees, mauvaise)
    for main_cat in os.listdir(DATASET_DIR):
        main_path = os.path.join(DATASET_DIR, main_cat)
        if not os.path.isdir(main_path): continue
        
        stats_dict[main_cat] = {}
        
        # Cas 1 : "mauvaise" qui contient des sous-dossiers (blur_light, etc.)
        if main_cat == "mauvaise":
            for sub_cat in os.listdir(main_path):
                sub_path = os.path.join(main_path, sub_cat)
                if not os.path.isdir(sub_path): continue
                
                images = [f for f in os.listdir(sub_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not images: continue
                
                print(f"  ➜ Analyse de {len(images)} images dans [{main_cat} / {sub_cat}]...")
                stats_dict[main_cat][sub_cat] = []
                
                for img_name in images:
                    img_path = os.path.join(sub_path, img_name)
                    face_crop = extract_and_align_face(img_path, yolo_model)
                    score = compute_grafiqs_raw(grafiqs_model, face_crop, device)
                    stats_dict[main_cat][sub_cat].append(score)
                    toutes_les_images.append({"main_cat": main_cat, "sub_cat": sub_cat, "filename": img_name, "score": score})
                    
        # Cas 2 : "bonne" et "variees" qui contiennent directement les images
        else:
            images = [f for f in os.listdir(main_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                print(f"  ➜ Analyse de {len(images)} images dans [{main_cat}]...")
                stats_dict[main_cat]["original"] = [] # On utilise "original" comme sous-catégorie par défaut
                
                for img_name in images:
                    img_path = os.path.join(main_path, img_name)
                    face_crop = extract_and_align_face(img_path, yolo_model)
                    score = compute_grafiqs_raw(grafiqs_model, face_crop, device)
                    stats_dict[main_cat]["original"].append(score)
                    toutes_les_images.append({"main_cat": main_cat, "sub_cat": "original", "filename": img_name, "score": score})

    # ==========================================
    # 3. GÉNÉRATION DU RAPPORT EN CONSOLE
    # ==========================================
    print("\n" + "="*60)
    print("📊 RÉSULTATS DU BENCHMARK GRAFIQS (SCORES BRUTS)")
    print("Rappel : Plus le score est BAS, meilleure est la qualité.")
    print("="*60)

    for main_cat, sub_cats in stats_dict.items():
        print(f"\n📁 CATÉGORIE PRINCIPALE : {main_cat.upper()}")
        
        # Trier les sous-catégories par ordre alphabétique pour la lisibilité
        for sub_cat in sorted(sub_cats.keys()):
            scores = sub_cats[sub_cat]
            if not scores: continue
            
            mediane = np.median(scores)
            minimum = np.min(scores)
            maximum = np.max(scores)
            
            print(f"  ├─ {sub_cat.ljust(15)} : Médiane = {mediane:.5f} | Plage : [{minimum:.5f} -> {maximum:.5f}]")

    # ==========================================
    # 4. EXPORT CSV (Pour tes graphiques)
    # ==========================================
    with open(CSV_OUTPUT, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["main_cat", "sub_cat", "filename", "score"])
        writer.writeheader()
        for row in toutes_les_images:
            writer.writerow(row)
            
    print("\n" + "="*60)
    print(f"✅ Analyse terminée ! Données complètes exportées dans : {CSV_OUTPUT}")
    print("="*60)

if __name__ == "__main__":
    main()