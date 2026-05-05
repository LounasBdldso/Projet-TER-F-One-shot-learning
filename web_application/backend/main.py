"""
main.py — Point d'entrée FastAPI

LANCEMENT :
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

DOCS AUTO :
    http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.recognition import router as recognition_router


# ============================================================
# LIFESPAN — chargement des modèles au démarrage
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Charge tous les modèles une seule fois au démarrage du serveur.
    Evite de recharger les modèles à chaque requête.
    """
    print("Chargement des modèles...")

    from services.detector import load_detector
    from services.encoder  import load_encoder
    from services.quality_service import load_quality_model

    app.state.detector = load_detector()
    print("  YOLO detector chargé")

    app.state.encoder = load_encoder()
    print("  ProtoNet encoder chargé")

    app.state.quality_model = load_quality_model()
    print("  GraFIQs quality model chargé")

    print("Tous les modèles sont prêts !")
    yield

    # Nettoyage à l'arrêt
    print("Arrêt du serveur — nettoyage...")


# ============================================================
# APP
# ============================================================
app = FastAPI(
    title="Face Recognition API",
    description="API de reconnaissance faciale basée sur ProtoNet + ResNet-18",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS — autorise le frontend React (localhost:3000)
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React dev server
        "http://localhost:5173",   # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================
app.include_router(recognition_router, prefix="/api", tags=["Recognition"])


# ============================================================
# HEALTH CHECK
# ============================================================
@app.get("/", tags=["Health"])
async def root():
    return {
        "status":  "ok",
        "message": "Face Recognition API is running",
        "version": "1.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}