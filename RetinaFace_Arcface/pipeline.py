
import torch
import numpy as np

print("torch:", torch.__version__)
print("numpy:", np.__version__)
print("MPS:", torch.backends.mps.is_available())

## A. Dépendances
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from insightface.app import FaceAnalysis



## B. Alignement par landmarks (CRUCIAL)
# Points cibles standards ArcFace (112x112)
ARCface_DST = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041],  # right mouth
], dtype=np.float32)




def align_face(img, landmarks, size=112):
    src = np.array(landmarks, dtype=np.float32)
    dst = ARCface_DST.copy()
    dst[:, 0] *= size / 112
    dst[:, 1] *= size / 112

    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    aligned = cv2.warpAffine(img, M, (size, size))
    return aligned




## Préprocessing & normalisation
transform = A.Compose([
    A.Resize(112, 112),
    A.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
    ToTensorV2()
])


## D. Extraction d’embedding
def extract_embedding(model, face_img, device):
    face = transform(image=face_img)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(face)
        emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze(0)


## E. Quality Gate (simple mais efficace)
def quality_gate(face_img, det_score, min_size=64, min_score=0.9):
    h, w, _ = face_img.shape
    if h < min_size or w < min_size:
        return False
    if det_score < min_score:
        return False
    return True

## F. Construction de la galerie (one-shot)
augment = A.Compose([
    A.Rotate(limit=5, p=0.5),
    A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
    A.Affine(
        translate_percent=0.02,
        scale=(0.95, 1.05),
        rotate=0,
        p=0.5
    )
])




def build_template(model, aligned_face, device, n_aug=25):
    embeddings = []

    embeddings.append(extract_embedding(model, aligned_face, device))

    for _ in range(n_aug):
        aug_face = augment(image=aligned_face)["image"]
        emb = extract_embedding(model, aug_face, device)
        embeddings.append(emb)

    template = torch.stack(embeddings).mean(dim=0)
    template = F.normalize(template, p=2, dim=0)
    return template



## G. Matching open-set(cosine)
def identify(embedding, gallery, threshold, margin=0.05):
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


## pre-H : appel insightFace pour la detection


app = FaceAnalysis(
    name="buffalo_l",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0)



## H. Pipeline complète sur une photo de foule
def recognize_image(image, model, gallery, threshold, device):
    results = []
    detections = RetinaFace.detect_faces(image)

    if detections is None:
        return results

    for det in detections.values():
        bbox = det["facial_area"]
        landmarks = det["landmarks"]
        score = det["score"]

        x1, y1, x2, y2 = bbox
        face = image[y1:y2, x1:x2]

        if not quality_gate(face, score):
            results.append(("Unknown", None))
            continue

        lm = [
            landmarks["left_eye"],
            landmarks["right_eye"],
            landmarks["nose"],
            landmarks["mouth_left"],
            landmarks["mouth_right"]
        ]

        aligned = align_face(image, lm)
        emb = extract_embedding(model, aligned, device)
        identity, sim = identify(emb, gallery, threshold)

        results.append((identity, sim))

    return results