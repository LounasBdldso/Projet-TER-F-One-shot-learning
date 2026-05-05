

import os
import sys
import csv
import math

import cv2
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ultralytics import YOLO

import config

# ============================================================
# PARAMÈTRES ET SEUILS CALIBRÉS
# ============================================================
SUPPORT_SET_DIR   = "./classe"
RESULTS_DIR       = "./results"
GRAFIQS_DIR       = "./GraFIQs"
WEIGHTS_PATH      = "./GraFIQs/weights/resnet50_webface_arcface.pth"
YOLO_CONFIDENCE   = 0.15
SCORE_GOOD        = 70
SCORE_ACCEPTABLE  = 50 # Légèrement remonté pour être plus strict

# --- LES SEUILS GRAFIQS ISSUS DE TON BENCHMARK ---
GRAFIQS_MAX_THEORIQUE = 0.00085  # Le score brut parfait (100/100)
GRAFIQS_SEUIL_REJET   = 0.00115  # Le point de rupture où l'image vaut 0/100

# --- NOUVELLE PONDÉRATION HYBRIDE ---
WEIGHTS_FINAL = {
    "sharpness":  0.35,   # +15% (Le bouclier anti-flou)
    "brightness": 0.25,   # +10% (Le bouclier anti-obscurité)
    "face_size":  0.10,   # -5%
    "grafiqs":    0.30,   # -20% (Maintenant utilisé comme expert géométrique)
}

# ============================================================
# CHARGEMENT DU BACKBONE GraFIQs (iResNet50)
# ============================================================
def load_grafiqs_model(weights_path, device):
    sys.path.insert(0, GRAFIQS_DIR)
    from backbones.iresnet import iresnet50
    from backbones.bn import BN_Model

    backbone = iresnet50(num_features=512, dropout=0.4, use_se=False).to(device)
    backbone.load_state_dict(torch.load(weights_path, map_location=device))
    backbone.return_intermediate = True
    backbone.eval()

    model = BN_Model(backbone, device)
    print(f"GraFIQs iResNet50 chargé depuis : {weights_path}")
    return model

grafiqs_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ============================================================
# 1. SCORE GraFIQs (NORMALISATION ABSOLUE)
# ============================================================
def compute_grafiqs_score(grafiqs_model, face_pil, device):
    img_tensor = grafiqs_transform(face_pil).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)

    bn_score, (emb, block1, block2, block3, block4, bn) = grafiqs_model.get_BN(img_tensor)
    grads = autograd.grad(outputs=bn_score, inputs=[img_tensor, block1, block2, block3, block4], allow_unused=True)

    grad_block4 = grads[4]
    if grad_block4 is not None:
        raw_score = float(torch.abs(grad_block4[0].cpu()).sum())
    else:
        raw_score = float(torch.abs(grads[0][0].cpu()).sum())
    return raw_score

def normalize_grafiqs_scores_absolute(raw_scores):
    """
    Convertit les scores bruts en note sur 100 en utilisant les seuils du Benchmark.
    Score brut plus bas = meilleure qualité.
    """
    normalized = []
    for score_brut in raw_scores:
        if score_brut >= GRAFIQS_SEUIL_REJET:
            normalized.append(0.0) # Carton rouge absolu (Trop de déformations)
        elif score_brut <= GRAFIQS_MAX_THEORIQUE:
            normalized.append(100.0) # Parfait
        else:
            # Interpolation linéaire inversée
            plage = GRAFIQS_SEUIL_REJET - GRAFIQS_MAX_THEORIQUE
            distance = GRAFIQS_SEUIL_REJET - score_brut
            norm = (distance / plage) * 100.0
            normalized.append(round(norm, 2))
    return normalized

# ============================================================
# 2. CRITÈRES CLASSIQUES
# ============================================================
def compute_sharpness(face_pil):
    face_gray = np.array(face_pil.convert("L"))
    laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
    score = min(100.0, (laplacian_var / 300.0) * 100.0) # Seuil ajusté pour webcam
    return round(score, 2), round(laplacian_var, 2)

def compute_brightness(face_pil):
    face_gray = np.array(face_pil.convert("L"))
    mean_brightness = face_gray.mean()
    optimal = 130.0
    tolerance = 80.0
    deviation = abs(mean_brightness - optimal)
    score = max(0.0, 100.0 - (deviation / tolerance) * 100.0)
    return round(score, 2), round(float(mean_brightness), 2)

def compute_face_size(box, image_size):
    x1, y1, x2, y2 = box
    img_w, img_h = image_size
    face_area = (x2 - x1) * (y2 - y1)
    image_area = img_w * img_h
    ratio = face_area / image_area
    score = min(100.0, (ratio / 0.10) * 100.0)
    return round(score, 2), round(ratio * 100, 2)

# ============================================================
# 3. SCORE FINAL
# ============================================================
def compute_final_score(sharpness, brightness, face_size, grafiqs):
    score = (
        WEIGHTS_FINAL["sharpness"]  * sharpness  +
        WEIGHTS_FINAL["brightness"] * brightness +
        WEIGHTS_FINAL["face_size"]  * face_size  +
        WEIGHTS_FINAL["grafiqs"]    * grafiqs
    )
    return round(score, 2)

def get_recommendation(score):
    if score >= SCORE_GOOD: return "OK Bonne"
    elif score >= SCORE_ACCEPTABLE: return "!! Acceptable"
    else: return "XX A remplacer"

# ============================================================
# 4. RAPPORT VISUEL & CSV
# ============================================================
def create_visual_report(results, output_path):
    if not results: return
    thumb_size, padding, text_height = 150, 15, 120
    cols = min(5, len(results))
    rows = (len(results) + cols - 1) // cols
    canvas = Image.new("RGB", (cols*(thumb_size+padding)+padding, rows*(thumb_size+text_height+padding)+padding), (30,30,30))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font_name = ImageFont.truetype("arial.ttf", 13)
        font_score = ImageFont.truetype("arial.ttf", 18)
    except:
        font_name = ImageFont.load_default()
        font_score = ImageFont.load_default()

    colors = {"OK Bonne": "#44FF88", "!! Acceptable": "#FFAA44", "XX A remplacer": "#FF4444"}

    for i, res in enumerate(results):
        x, y = padding + (i%cols)*(thumb_size+padding), padding + (i//cols)*(thumb_size+text_height+padding)
        face_img = res["face_crop"].resize((thumb_size, thumb_size)) if res["face_crop"] else Image.new("RGB", (thumb_size, thumb_size), (80,80,80))
        border_img = Image.new("RGB", (thumb_size+8, thumb_size+8), colors.get(res["recommendation"], "#FFFFFF"))
        border_img.paste(face_img, (4, 4))
        canvas.paste(border_img, (x-4, y-4))
        
        draw.text((x, y+thumb_size+4), res["name"][:20], fill="#FFFFFF", font=font_name)
        draw.text((x, y+thumb_size+16), f"Score: {res['final_score']:.0f}/100", fill=colors.get(res["recommendation"], "#FFF"), font=font_score)
        draw.text((x, y+thumb_size+38), f"Net: {res['sharpness_score']:.0f} | Lum: {res['brightness_score']:.0f}", fill="#AAA", font=font_name)
        draw.text((x, y+thumb_size+54), f"GraFIQs: {res['grafiqs_score']:.0f} | Brut: {res['grafiqs_raw']:.5f}", fill="#44CCFF", font=font_name)

    canvas.save(output_path, quality=95)

def export_csv(results, output_path):
    fieldnames = ["name", "image_path", "final_score", "recommendation", "sharpness_score", "sharpness_raw", "brightness_score", "brightness_raw", "face_size_score", "face_size_pct", "grafiqs_score", "grafiqs_raw", "face_detected"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results: writer.writerow({k: res.get(k, "") for k in fieldnames})

# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(SUPPORT_SET_DIR): 
        print(f"ERREUR : Support set introuvable : {SUPPORT_SET_DIR}")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[1/3] Chargement de YOLO...")
    yolo_model = YOLO(config.YOLO_MODEL_PATH)

    print("\n[2/3] Chargement du modele GraFIQs (iResNet50)...")
    grafiqs_model = load_grafiqs_model(WEIGHTS_PATH, device)

    print("\n[3/3] Analyse et Alignement des images...")
    image_files = [f for f in sorted(os.listdir(SUPPORT_SET_DIR)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    results, grafiqs_raws = [], []

    print("Extraction YOLO, Alignement et calcul GraFIQs...")
    for filename in image_files:
        filepath = os.path.join(SUPPORT_SET_DIR, filename)
        image = Image.open(filepath).convert("RGB")
        
        yolo_results = yolo_model(filepath, conf=YOLO_CONFIDENCE, verbose=False)
        best_box, best_area, best_kpts = None, 0, None

        if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
            result = yolo_results[0]
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

        face_detected = best_box is not None

        if face_detected:
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
                face_crop = Image.fromarray(aligned_crop)
                box_for_size = best_box
            else:
                x1, y1, x2, y2 = best_box
                w, h = image.size
                pad_x, pad_y = int((x2-x1)*0.1), int((y2-y1)*0.1)
                face_crop = image.crop((max(0, x1-pad_x), max(0, y1-pad_y), min(w, x2+pad_x), min(h, y2+pad_y)))
                box_for_size = best_box
        else:
            face_crop = image
            box_for_size = (0, 0, image.width, image.height)

        raw = compute_grafiqs_score(grafiqs_model, face_crop, device)
        grafiqs_raws.append(raw)

        results.append({
            "filename": filename, "filepath": filepath, "image": image,
            "face_crop": face_crop, "box_for_size": box_for_size,
            "face_detected": face_detected, "grafiqs_raw": raw,
        })

    # NOUVEAU : Application des seuils absolus calibrés
    grafiqs_normalized = normalize_grafiqs_scores_absolute(grafiqs_raws)
    
    print("\nCalcul des scores finaux...")
    final_results = []

    for i, res in enumerate(results):
        name = " ".join([p for p in os.path.splitext(res["filename"])[0].split("_") if not p.isdigit()])
        sharpness_score, sharpness_raw = compute_sharpness(res["face_crop"])
        brightness_score, brightness_raw = compute_brightness(res["face_crop"])
        face_size_score, face_size_pct = compute_face_size(res["box_for_size"], res["image"].size)
        grafiqs_score = grafiqs_normalized[i]

        final_score = compute_final_score(sharpness_score, brightness_score, face_size_score, grafiqs_score)
        recommendation = get_recommendation(final_score)

        print(f"[{recommendation}] {name[:15]:<15} | Global: {final_score:04.1f}/100 | Net: {sharpness_score:04.1f} | Lum: {brightness_score:04.1f} | GraFIQs: {grafiqs_score:04.1f}")
        
        final_results.append({
            "name": name, "image_path": res["filepath"], "final_score": final_score,
            "recommendation": recommendation, "sharpness_score": sharpness_score,
            "sharpness_raw": sharpness_raw, "brightness_score": brightness_score,
            "brightness_raw": brightness_raw, "face_size_score": face_size_score,
            "face_size_pct": face_size_pct, "grafiqs_score": grafiqs_score,
            "grafiqs_raw": res["grafiqs_raw"], "face_detected": res["face_detected"],
            "face_crop": res["face_crop"],
        })

    csv_path = os.path.join(RESULTS_DIR, "quality_report_grafiqs.csv")
    export_csv([{k:v for k,v in r.items() if k!="face_crop"} for r in final_results], csv_path)
    visual_path = os.path.join(RESULTS_DIR, "quality_report_grafiqs.jpg")
    create_visual_report(final_results, visual_path)
    
    print("\n" + "="*50)
    print("✅ Analyse terminee ! Rapport sauvegarde dans ./results/")
    print("="*50)

if __name__ == "__main__":
    main()