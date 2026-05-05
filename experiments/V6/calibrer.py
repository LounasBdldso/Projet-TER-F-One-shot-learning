import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import json

class CalibrationDatasetGenerator:
    """
    Génère un dataset de calibration avec dégradations contrôlées.
    """
    
    def __init__(self, input_dir, output_dir):
        """
        Args:
            input_dir: Dossier contenant les images de référence de bonne qualité
            output_dir: Dossier de sortie pour le dataset de calibration
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.metadata = []  # Pour tracking des dégradations
    
    def generate_dataset(self):
        """
        Génère le dataset complet avec toutes les dégradations.
        """
        print("="*70)
        print("GÉNÉRATION DU DATASET DE CALIBRATION")
        print("="*70)
        
        # Créer la structure de dossiers
        categories = [
            "original",
            "blur_light", "blur_medium", "blur_heavy",
            "dark_light", "dark_heavy",
            "bright_light", "bright_heavy",
            "rotation_15", "rotation_30",
            "lowres_50", "lowres_25",
            "combined_blur_dark", "combined_blur_bright"
        ]
        
        for category in categories:
            (self.output_dir / category).mkdir(parents=True, exist_ok=True)
        
        # Charger toutes les images sources
        image_files = list(self.input_dir.glob("*.jpg")) + \
                     list(self.input_dir.glob("*.png")) + \
                     list(self.input_dir.glob("*.jpeg"))
        
        if not image_files:
            print(f"❌ Aucune image trouvée dans {self.input_dir}")
            return
        
        print(f"✅ {len(image_files)} images sources trouvées")
        print(f"📊 Génération de {len(image_files) * len(categories)} images...")
        print()
        
        total_generated = 0
        
        for img_path in image_files:
            basename = img_path.stem
            print(f"Traitement : {basename}")
            
            # Charger l'image
            img = Image.open(img_path).convert("RGB")
            
            # 1. Original (copie)
            self._save_image(img, "original", basename, {
                "degradation": "none",
                "quality_expected": "excellent"
            })
            
            # 2. Flou (3 niveaux)
            self._apply_blur(img, basename, "light", radius=2)
            self._apply_blur(img, basename, "medium", radius=4)
            self._apply_blur(img, basename, "heavy", radius=8)
            
            # 3. Luminosité basse (2 niveaux)
            self._apply_brightness(img, basename, "dark_light", factor=0.6)
            self._apply_brightness(img, basename, "dark_heavy", factor=0.3)
            
            # 4. Luminosité haute (2 niveaux)
            self._apply_brightness(img, basename, "bright_light", factor=1.4)
            self._apply_brightness(img, basename, "bright_heavy", factor=1.8)
            
            # 5. Rotation (2 niveaux)
            self._apply_rotation(img, basename, "rotation_15", angle=15)
            self._apply_rotation(img, basename, "rotation_30", angle=30)
            
            # 6. Basse résolution (2 niveaux)
            self._apply_lowres(img, basename, "lowres_50", scale=0.5)
            self._apply_lowres(img, basename, "lowres_25", scale=0.25)
            
            # 7. Combinaisons (défauts multiples)
            self._apply_combined(img, basename, "blur_dark", 
                               blur_radius=3, brightness=0.5)
            self._apply_combined(img, basename, "blur_bright",
                               blur_radius=3, brightness=1.5)
            
            total_generated += len(categories)
            print(f"  ✓ {len(categories)} variantes générées")
        
        # Sauvegarder métadonnées
        self._save_metadata()
        
        print()
        print("="*70)
        print(f"✅ DATASET GÉNÉRÉ : {total_generated} images")
        print(f"📁 Emplacement : {self.output_dir}")
        print(f"📄 Métadonnées : {self.output_dir / 'metadata.json'}")
        print("="*70)
    
    # ==================== DÉGRADATIONS ====================
    
    def _apply_blur(self, img, basename, level, radius):
        """Applique un flou gaussien."""
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        category = f"blur_{level}"
        self._save_image(blurred, category, basename, {
            "degradation": "blur",
            "level": level,
            "radius": radius,
            "quality_expected": "bad" if level == "heavy" else "acceptable"
        })
    
    def _apply_brightness(self, img, basename, category, factor):
        """Modifie la luminosité."""
        enhancer = ImageEnhance.Brightness(img)
        adjusted = enhancer.enhance(factor)
        
        quality = "bad" if factor < 0.4 or factor > 1.6 else "acceptable"
        
        self._save_image(adjusted, category, basename, {
            "degradation": "brightness",
            "factor": factor,
            "quality_expected": quality
        })
    
    def _apply_rotation(self, img, basename, category, angle):
        """Applique une rotation."""
        rotated = img.rotate(angle, expand=True, fillcolor=(128, 128, 128))
        self._save_image(rotated, category, basename, {
            "degradation": "rotation",
            "angle": angle,
            "quality_expected": "acceptable"
        })
    
    def _apply_lowres(self, img, basename, category, scale):
        """Réduit la résolution puis upscale."""
        w, h = img.size
        small = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        upscaled = small.resize((w, h), Image.LANCZOS)
        
        quality = "bad" if scale <= 0.3 else "acceptable"
        
        self._save_image(upscaled, category, basename, {
            "degradation": "lowres",
            "scale": scale,
            "quality_expected": quality
        })
    
    def _apply_combined(self, img, basename, category, blur_radius, brightness):
        """Combine plusieurs dégradations."""
        # Flou
        degraded = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        # Luminosité
        enhancer = ImageEnhance.Brightness(degraded)
        degraded = enhancer.enhance(brightness)
        
        self._save_image(degraded, f"combined_{category}", basename, {
            "degradation": "combined",
            "blur_radius": blur_radius,
            "brightness": brightness,
            "quality_expected": "bad"
        })
    
    # ==================== UTILITAIRES ====================
    
    def _save_image(self, img, category, basename, metadata):
        """Sauvegarde une image et enregistre ses métadonnées."""
        output_path = self.output_dir / category / f"{basename}.jpg"
        img.save(output_path, quality=95)
        
        self.metadata.append({
            "filename": f"{category}/{basename}.jpg",
            "category": category,
            "original_name": basename,
            **metadata
        })
    
    def _save_metadata(self):
        """Sauvegarde les métadonnées en JSON."""
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Métadonnées sauvegardées : {metadata_path}")


# ==================== SCRIPT D'UTILISATION ====================

def main():
    """
    Génère le dataset de calibration.
    """
    # Configuration
    INPUT_DIR = "./calibration/bonne"  # Vos images de bonne qualité
    OUTPUT_DIR = "./calibration/mauvaise"  # Dossier de sortie pour les images dégradées
    
    # Vérification
    if not os.path.exists(INPUT_DIR):
        print(f"❌ ERREUR : Le dossier {INPUT_DIR} n'existe pas")
        print(f"📝 Créez ce dossier et placez-y 10-15 images de bonne qualité")
        return
    
    # Génération
    generator = CalibrationDatasetGenerator(INPUT_DIR, OUTPUT_DIR)
    generator.generate_dataset()
    
    print("\n" + "="*70)
    print("PROCHAINES ÉTAPES")
    print("="*70)
    print("1. Vérifiez visuellement les images générées")
    print("2. Ajoutez des images 'naturelles' variées dans le dossier approprié")
    print("3. Lancez l'évaluation de qualité sur tout le dataset")
    print("4. Lancez la reconnaissance pour mesurer la corrélation")
    print("="*70)


if __name__ == "__main__":
    main()