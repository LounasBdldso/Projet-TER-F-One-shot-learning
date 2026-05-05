# 🧠 Qui est qui ? - Face Recognition & FIQA Pipeline

Une application Full-Stack de reconnaissance faciale basée sur l'apprentissage **Few-Shot (ProtoNet)**, intégrant un pipeline avancé d'évaluation de la qualité des images faciale (**FIQA**).

Ce projet a été développé dans le cadre d'un Master et propose une architecture modulaire séparant un moteur d'Intelligence Artificielle robuste (FastAPI/PyTorch) d'une interface utilisateur moderne (React/Vite).

---

## ✨ Fonctionnalités Principales

*   📸 **Détection de Visages :** Utilisation de `YOLOv8n-face` pour une détection rapide et précise, même sur des photos de groupe denses.
*   🧬 **Reconnaissance Few-Shot (ProtoNet) :** Encodage des visages via un `ResNet-18` adapté. Le système peut identifier des individus dans une photo de groupe à partir d'une seule image de référence (One-Shot Learning) grâce à la **Similarité Cosinus**.
*   🛡️ **Vigile de Qualité (FIQA Hybride) :** Avant l'enrôlement, chaque photo de référence est auditée :
    *   *Netteté* (OpenCV Laplacien / Filtre anti-flou)
    *   *Exposition* (OpenCV Luminosité moyenne)
    *   *Géométrie Faciale* (Modèle Deep Learning GraFIQs - iResNet50)
*   💻 **Interface Moderne :** Frontend réactif et animé développé avec React, TailwindCSS et AOS.

---

## 🏗️ Architecture et Technologies

### Backend (Moteur IA & API)
*   **Framework :** FastAPI (Python)
*   **Machine Learning :** PyTorch, Torchvision
*   **Computer Vision :** OpenCV, Ultralytics (YOLOv8)
*   **Modèles pré-entraînés :** ResNet-18 (ProtoNet), iResNet-50 (GraFIQs)

### Frontend (Interface Utilisateur)
*   **Framework :** React 18 (via Vite)
*   **Routage :** React Router DOM
*   **Stylisation :** Tailwind CSS
*   **Icônes & Animations :** Lucide React, AOS (Animate On Scroll)

---


## 🚀 Installation et Démarrage

### 1. Prérequis
*   Python 3.9+
*   Node.js 18+ & npm

### 2. Configuration du Backend

\`\`\`bash
cd backend

# Installation des dépendances Python
pip install fastapi uvicorn torch torchvision opencv-python pillow ultralytics python-multipart
\`\`\`

⚠️ **Important : Fichiers de poids**
Le dossier `backend/models/weights/` doit contenir les modèles suivants avant le lancement :
*   `yolov8n-face.pt` (Modèle YOLO de détection)
*   `protonet_best.pth` (Poids de ton modèle ResNet-18 entraîné)
*   `resnet50_webface_arcface.pth` (Poids du modèle GraFIQs)

**Lancement du serveur backend :**
\`\`\`bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
\`\`\`
L'API sera disponible sur : `http://localhost:8000` (Documentation Swagger sur `/docs`).

### 3. Configuration du Frontend

Ouvrez un nouveau terminal :

\`\`\`bash
cd frontend

# Installation des dépendances Node
npm install

# Lancement du serveur de développement
npm run dev
\`\`\`
L'interface web sera disponible sur : `http://localhost:5173` (ou le port indiqué par Vite).

---

## 🎮 Comment utiliser l'application ?

1.  **Audit de Qualité (Démo) :** Naviguez vers la page "Qualité d'image" pour tester la robustesse du filtre FIQA avec différentes photos (floues, sombres, bien cadrées).
2.  **Reconnaissance Faciale :**
    *   Allez sur la page "Reconnaissance".
    *   **Étape 1 :** Uploadez les photos de référence (Support Set) nommées avec le prénom des personnes (ex: `Lounas.jpg`).
    *   **Étape 2 :** Uploadez une photo de groupe contenant ces personnes.
    *   Cliquez sur **"Lancer la Reconnaissance"** pour voir le modèle YOLO détecter les visages et le ProtoNet les identifier.

---

## 👨‍🎓 Auteur
Développé dans le cadre d'un projet de Master.