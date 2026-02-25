import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt



#-----------------------------------------
# seuil de confiance adaptatif
#---------------------------------------------

def adaptative_threshold(w,h):
    size=min(w,h)

    if size >= 100 :
        return 0.55    # one-shot normal
    elif size >= 70 :
        return 0.65
    elif size >= 50 :
        return 0.75
    else:
        return 1.0      # presque rejet








# -------------------------------------------------
# Quality Gate (inchangé conceptuellement)
# -------------------------------------------------
def quality_gate(face, min_score=0.5, min_size=45):
    x1, y1, x2, y2 = face.bbox.astype(int)
    w, h = x2 - x1, y2 - y1

    print("det_score =", face.det_score)
    print("bbox size =", w, "x", h)

    if w < min_size or h < min_size:
        return False
    if face.det_score < min_score:
        return False
    return True


# -------------------------------------------------
# Build template (ONE-SHOT)
# -------------------------------------------------
def build_template(face):
    """
    face : insightface.app.common.Face
    retourne : torch.Tensor (512,)
    """
    emb = torch.from_numpy(face.embedding)
    emb = torch.nn.functional.normalize(emb, p=2, dim=0)
    return emb


# -------------------------------------------------
# Identification open-set
# -------------------------------------------------
def identify(embedding, gallery, threshold=0.6, margin=0.05):
    scores = {k: torch.dot(embedding, v).item()
              for k, v in gallery.items()}

    best = max(scores, key=scores.get)
    best_score = scores[best]

    sorted_scores = sorted(scores.values(), reverse=True)

    if best_score < threshold:
        return "Unknown", best_score

    if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < margin:
        return "Unknown", best_score

    return best, best_score


# -------------------------------------------------
# Pipeline reconnaissance image
# -------------------------------------------------
def recognize_image(image, app, gallery,
                    threshold=0.6,
                    min_score=0.5):

    results = []

    faces = app.get(image)
    if len(faces) == 0:
        return results

    for face in faces:
        if not quality_gate(face, min_score=min_score):
            results.append(("Unknown", None))
            continue

        emb = torch.from_numpy(face.embedding)
        emb = torch.nn.functional.normalize(emb, p=2, dim=0)

        identity, score = identify(
            emb,
            gallery,
            threshold=threshold
        )

        results.append((identity, score))

    return results




#------------------------------------------------------
#Visualisation de la détection
#------------------------------------------------------



def visualize_decision(img_bgr, results, faces, threshold=0.5, figsize=(14, 10)):
    
    """
    img_bgr : image OpenCV (BGR)
    results : output du matching
    faces   : faces détectées InsightFace
    """
    vis = img_bgr.copy()

    for res, face in zip(results, faces):
        x1, y1, x2, y2 = face.bbox.astype(int)

        #  Supporte dict OU tuple ou Int 
        if isinstance(res, dict):
            identity = res.get("identity", "unknown")
            score = res.get("similarity", None)
        else:
            identity, score = res  # tuple (identity, score)

        # normalisation du label unknown
        if identity in ["Unknown", "UNKNOWN", None]:
            identity = "unknown"

        score_val = float(score) if score is not None else -1.0

        if identity == "unknown" or score is None or score_val < threshold:
            color = (0, 0, 255)
            label = f"UNKNOWN ({score_val:.2f})" if score is not None else "UNKNOWN"
        else:
            color = (0, 255, 0)
            label = f"{identity} ({score_val:.2f})"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)   ### Bounding box epaisse
        cv2.putText(vis, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)   ### texte lisible

    # pour affichage matplotlib
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(vis_rgb)
    plt.axis("off")
    plt.tight_layout()

    return vis_rgb

    #  Conversion couleur UNE FOIS ici
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #  Affichage contrôlé
    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.tight_layout()

    return img_rgb