"""
services/encoder.py — Embeddings ProtoNet

Expose :
    load_encoder()        → charge le modèle ProtoNet
    encode_image()        → embedding d'une image PIL
    encode_support_set()  → prototypes depuis les photos de référence
    identify_faces()      → identification avec assignation unique (greedy)
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from models.resnet18 import ProtoNetEncoder
from utils.preprocessor import pil_to_tensor, EVAL_TRANSFORM

# ============================================================
# CONFIG
# ============================================================
WEIGHTS_PATH = Path("models/weights/protonet_best.pth")
SCALE        = 20.0   # temperature scaling pour le softmax


# ============================================================
# CHARGEMENT
# ============================================================
def load_encoder() -> tuple[ProtoNetEncoder, dict, torch.device]:
    """
    Charge le modèle ProtoNet depuis le checkpoint.
    Retourne (encoder, ckpt_config, device).
    Appelé une seule fois au démarrage via lifespan dans main.py.
    """
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint ProtoNet introuvable : {WEIGHTS_PATH}\n"
            f"Place protonet_best.pth dans models/weights/"
        )

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    ckpt_config = checkpoint["config"]

    encoder = ProtoNetEncoder(
        embedding_dim=ckpt_config["embedding_dim"],
        backbone_name=ckpt_config.get("backbone", "resnet18"),
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    print(f"  Backbone      : {ckpt_config.get('backbone', 'resnet18')}")
    print(f"  Embedding dim : {ckpt_config['embedding_dim']}")
    print(f"  Epoque        : {checkpoint['epoch']}")
    print(f"  Val Acc       : {checkpoint['val_acc']:.4f}")
    print(f"  Device        : {device}")

    return encoder, ckpt_config, device


# ============================================================
# ENCODAGE
# ============================================================
@torch.no_grad()
def encode_image(
    face_crop: Image.Image,
    encoder: ProtoNetEncoder,
    device: torch.device,
) -> torch.Tensor:
    """
    Calcule l'embedding normalisé L2 d'un visage cropé.

    Args:
        face_crop : image PIL du visage (après crop YOLO)
        encoder   : ProtoNetEncoder chargé
        device    : cuda ou cpu

    Returns:
        embedding : tenseur [embedding_dim] normalisé L2
    """
    tensor = pil_to_tensor(face_crop, EVAL_TRANSFORM)
    tensor = tensor.unsqueeze(0).to(device)   # [1, 3, H, W]
    emb    = encoder(tensor)                   # [1, embedding_dim] — déjà normalisé L2
    return emb.squeeze(0)                      # [embedding_dim]


@torch.no_grad()
def encode_support_set(
    support_data: list[dict],
    encoder: ProtoNetEncoder,
    device: torch.device,
) -> tuple[list[str], torch.Tensor]:
    """
    Calcule les prototypes pour chaque personne du support set.

    Args:
        support_data : liste de dicts {"name": str, "crops": [PIL, ...]}
                       chaque personne peut avoir plusieurs photos
        encoder      : ProtoNetEncoder chargé
        device       : cuda ou cpu

    Returns:
        names      : liste des noms dans le même ordre que prototypes
        prototypes : tenseur [N_persons, embedding_dim] normalisé L2
    """
    names      = []
    prototypes = []

    for person in support_data:
        name  = person["name"]
        crops = person["crops"]   # liste d'images PIL

        # Encoder toutes les photos de cette personne
        embeddings = []
        for crop in crops:
            emb = encode_image(crop, encoder, device)
            embeddings.append(emb)

        # Prototype = moyenne des embeddings
        person_embs = torch.stack(embeddings)         # [N_photos, dim]
        prototype   = person_embs.mean(dim=0)         # [dim]
        prototype   = F.normalize(prototype, p=2, dim=0)  # renormaliser après moyenne

        names.append(name)
        prototypes.append(prototype)

    prototypes_tensor = torch.stack(prototypes)   # [N_persons, dim]
    return names, prototypes_tensor


# ============================================================
# IDENTIFICATION
# ============================================================
@torch.no_grad()
def identify_faces(
    query_crops: list[Image.Image],
    names: list[str],
    prototypes: torch.Tensor,
    encoder: ProtoNetEncoder,
    device: torch.device,
    scale: float = SCALE,
) -> list[dict]:
    """
    Identifie chaque visage via distance cosinus + assignation greedy.

    Args:
        query_crops : liste d'images PIL (visages détectés dans la photo de groupe)
        names       : liste des noms du support set
        prototypes  : tenseur [N_persons, dim] — prototypes du support set
        encoder     : ProtoNetEncoder chargé
        device      : cuda ou cpu
        scale       : temperature scaling pour le softmax

    Returns:
        predictions : liste de dicts par visage :
            {
                "name"      : str,
                "confidence": float (0-1),
                "distance"  : float (0-2),
            }
    """
    encoder.eval()

    # Encoder tous les visages query
    query_embeddings = []
    for crop in query_crops:
        emb = encode_image(crop, encoder, device)
        query_embeddings.append(emb)

    query_embeddings = torch.stack(query_embeddings)   # [N_queries, dim]

    # Distance cosinus : 1 - similarité
    # query_embeddings et prototypes sont déjà normalisés L2
    similarities = torch.mm(query_embeddings, prototypes.t())  # [N_queries, N_persons]
    distances    = 1 - similarities                             # [N_queries, N_persons]

    # Confiances via softmax scalé
    confidences  = torch.softmax(-distances * scale, dim=1)    # [N_queries, N_persons]

    n_queries = len(query_crops)
    n_persons = len(names)

    # Assignation gloutonne (greedy matching)
    # → chaque identité ne peut être attribuée qu'une seule fois
    candidates = []
    for q_idx in range(n_queries):
        for p_idx in range(n_persons):
            candidates.append({
                "query_idx":  q_idx,
                "person_idx": p_idx,
                "confidence": confidences[q_idx, p_idx].item(),
                "distance":   distances[q_idx, p_idx].item(),
            })

    candidates.sort(key=lambda x: x["confidence"], reverse=True)

    assigned_queries  = set()
    assigned_persons  = set()
    predictions       = [None] * n_queries

    for cand in candidates:
        q_idx = cand["query_idx"]
        p_idx = cand["person_idx"]

        if q_idx in assigned_queries:
            continue
        if p_idx in assigned_persons:
            continue

        predictions[q_idx] = {
            "name":       names[p_idx],
            "confidence": round(cand["confidence"], 4),
            "distance":   round(cand["distance"], 4),
        }
        assigned_queries.add(q_idx)
        assigned_persons.add(p_idx)

        if len(assigned_queries) == n_queries:
            break

    # Visages non assignés (plus de personnes dans le support set)
    for q_idx in range(n_queries):
        if predictions[q_idx] is None:
            predictions[q_idx] = {
                "name":       "Inconnu",
                "confidence": 0.0,
                "distance":   2.0,
            }

    return predictions