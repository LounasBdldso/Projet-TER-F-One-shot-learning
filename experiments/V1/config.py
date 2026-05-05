"""
Configuration pour l'entraînement ProtoNet avec DDP sur 2 GPU.
Dataset : WebFace112x112 (images déjà croppées et alignées).
"""

# ============================================================
# CHEMINS (à adapter selon la machine distante)
# ============================================================
CELEBA_DIR = "../data/webface_112x112"     # Dossier racine du dataset WebFace
YOLO_MODEL_PATH = "./yolov8n-face.pt"
CHECKPOINT_DIR = "./checkpoints"

# ============================================================
# FILTRAGE DU DATASET
# ============================================================
MIN_IMAGES_PER_IDENTITY = 20

# ============================================================
# SPLIT DES IDENTITÉS
# ============================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# HYPERPARAMÈTRES DE L'ÉPISODE
# ============================================================
N_SHOT = 1
K_WAY = 20
N_QUERY = 7

# ============================================================
# HYPERPARAMÈTRES DU MODÈLE
# ============================================================
BACKBONE = "resnet18"  # Options : "resnet18", "resnet50", "resnet101"
EMBEDDING_DIM = 128
IMAGE_SIZE = 112

# ============================================================
# HYPERPARAMÈTRES D'OPTIMISATION
# ============================================================
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5

# ============================================================
# HYPERPARAMÈTRES D'ENTRAÎNEMENT
# ============================================================
EPISODES_PER_EPOCH = 400    
NUM_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 25

# ============================================================
# DEVICE
# ============================================================
# En DDP, le device est automatiquement géré par le rang local
DEVICE = "cuda"

# ============================================================
# DDP
# ============================================================
NUM_WORKERS = 4             # Workers pour le DataLoader (par process)