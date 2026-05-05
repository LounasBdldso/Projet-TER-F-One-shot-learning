"""
api/recognition.py — Routes FastAPI

Endpoints :
    POST /api/recognize  → reconnaissance faciale sur photo de groupe
    POST /api/quality    → évaluation qualité d'une photo de référence
"""

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from typing import List

from services.recognition_service import recognize_group, draw_predictions
from services.detector            import detect_best_face
from services.quality_service     import assess_image_quality
from utils.preprocessor           import bytes_to_pil

router = APIRouter()


# ============================================================
# POST /api/recognize
# ============================================================
@router.post("/recognize")
async def recognize(
    request: Request,
    group_photo: UploadFile = File(..., description="Photo de groupe"),
    reference_photos: List[UploadFile] = File(..., description="Photos de référence"),
    reference_names: List[str] = Form(..., description="Noms associés aux photos de référence"),
):
    """
    Reconnaissance faciale sur une photo de groupe.

    - **group_photo**      : la photo de groupe (JPEG/PNG)
    - **reference_photos** : liste de photos de référence (1 par envoi minimum)
    - **reference_names**  : liste des noms dans le même ordre que les photos

    Retourne l'image annotée directement (JPEG bytes).
    """
    # --- Validation ---
    if len(reference_photos) != len(reference_names):
        raise HTTPException(
            status_code=400,
            detail=f"Le nombre de photos ({len(reference_photos)}) "
                   f"ne correspond pas au nombre de noms ({len(reference_names)})."
        )

    if len(reference_photos) == 0:
        raise HTTPException(
            status_code=400,
            detail="Au moins une photo de référence est requise."
        )

    # --- Lire les bytes ---
    group_bytes = await group_photo.read()

    support_files = []
    for photo, name in zip(reference_photos, reference_names):
        image_bytes = await photo.read()
        support_files.append({
            "name":        name.strip(),
            "image_bytes": image_bytes,
        })

    # --- Récupérer les modèles depuis app.state ---
    detector    = request.app.state.detector
    encoder, ckpt_config, device = request.app.state.encoder

    # --- Pipeline de reconnaissance ---
    result = recognize_group(
        group_image_bytes=group_bytes,
        support_files=support_files,
        detector=detector,
        encoder=encoder,
        ckpt_config=ckpt_config,
        device=device,
    )

    if not result["success"]:
        raise HTTPException(status_code=422, detail=result["message"])

    # --- Annoter l'image ---
    annotated_bytes = draw_predictions(
        group_image_bytes=group_bytes,
        boxes=result["boxes"],
        predictions=result["predictions"],
    )

    # --- Renvoyer l'image annotée + métadonnées dans les headers ---
    return Response(
        content=annotated_bytes,
        media_type="image/jpeg",
        headers={
            "X-Faces-Detected": str(result["n_faces"]),
            "X-Message":        result["message"],
        }
    )


# ============================================================
# POST /api/recognize/json
# ============================================================
@router.post("/recognize/json")
async def recognize_json(
    request: Request,
    group_photo: UploadFile = File(...),
    reference_photos: List[UploadFile] = File(...),
    reference_names: List[str] = Form(...),
):
    """
    Même pipeline que /recognize mais retourne un JSON avec :
    - les prédictions (nom, confiance, distance)
    - les boîtes de détection
    - l'image annotée encodée en base64

    Utile si le frontend veut afficher les boîtes lui-même.
    """
    import base64

    if len(reference_photos) != len(reference_names):
        raise HTTPException(
            status_code=400,
            detail="Le nombre de photos ne correspond pas au nombre de noms."
        )

    group_bytes = await group_photo.read()

    support_files = []
    for photo, name in zip(reference_photos, reference_names):
        image_bytes = await photo.read()
        support_files.append({
            "name":        name.strip(),
            "image_bytes": image_bytes,
        })

    detector             = request.app.state.detector
    encoder, ckpt_config, device = request.app.state.encoder

    result = recognize_group(
        group_image_bytes=group_bytes,
        support_files=support_files,
        detector=detector,
        encoder=encoder,
        ckpt_config=ckpt_config,
        device=device,
    )

    if not result["success"]:
        raise HTTPException(status_code=422, detail=result["message"])

    annotated_bytes  = draw_predictions(
        group_image_bytes=group_bytes,
        boxes=result["boxes"],
        predictions=result["predictions"],
    )
    annotated_base64 = base64.b64encode(annotated_bytes).decode("utf-8")

    return JSONResponse({
        "success":        True,
        "message":        result["message"],
        "n_faces":        result["n_faces"],
        "predictions":    result["predictions"],
        "boxes":          result["boxes"],
        "annotated_image": annotated_base64,
    })


# ============================================================
# POST /api/quality
# ============================================================
@router.post("/quality")
async def quality(
    request: Request,
    photo: UploadFile = File(..., description="Photo de référence à évaluer"),
):
    """
    Évalue la qualité d'une photo de référence.

    Retourne un JSON avec :
    - **final_score**      : score global 0-100
    - **recommendation**   : "bonne" | "acceptable" | "a_remplacer"
    - **face_detected**    : bool
    - **sharpness_score**  : score netteté 0-100
    - **brightness_score** : score luminosité 0-100
    - **face_size_score**  : score taille visage 0-100
    - **grafiqs_score**    : score GraFIQs 0-100 (calibré)
    - **grafiqs_raw**      : score brut GraFIQs (pour debug)
    """
    # --- Lire l'image ---
    image_bytes = await photo.read()
    pil_image   = bytes_to_pil(image_bytes)

    # --- Récupérer les modèles ---
    detector              = request.app.state.detector
    grafiqs_model, device = request.app.state.quality_model

    # --- Détecter le visage ---
    face_crop, detected = detect_best_face(pil_image, detector)

    # Construire la box pour le calcul de taille
    if detected:
        from services.detector import YOLO_CONFIDENCE
        results = detector(pil_image, conf=YOLO_CONFIDENCE, verbose=False)
        best_box  = None
        best_area = 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_box  = (int(x1), int(y1), int(x2), int(y2))
        box = best_box
    else:
        box = None

    # --- Évaluer la qualité ---
    quality_result = assess_image_quality(
        face_pil=face_crop,
        box=box,
        image_size=pil_image.size,
        grafiqs_model=grafiqs_model,
        device=device,
    )

    return JSONResponse(quality_result)


# ============================================================
# GET /api/info
# ============================================================
@router.get("/info")
async def info(request: Request):
    """
    Retourne les informations sur le modèle chargé.
    """
    _, ckpt_config, device = request.app.state.encoder

    return JSONResponse({
        "backbone":      ckpt_config.get("backbone", "resnet18"),
        "embedding_dim": ckpt_config.get("embedding_dim", 256),
        "k_way":         ckpt_config.get("k_way", 25),
        "n_shot":        ckpt_config.get("n_shot", 1),
        "image_size":    ckpt_config.get("image_size", 112),
        "device":        str(device),
    })