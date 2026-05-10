"""
SYSTÈME DE DÉTECTION DE DÉFAUTS EN TEMPS RÉEL
Projet: Sougui.TN - Contrôle Qualité Artisanat
Détecte si un objet (verre, poterie, etc.) est cassé ou endommagé
"""

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import datetime

class DefectDetector:
    """Détecteur de défauts en temps réel"""
    
    def __init__(self, model_path=None):
        """Initialiser le détecteur"""
        self.model = None
        self.class_names = ['Intact', 'Cassé', 'Endommagé']
        self.colors = {
            'Intact': (0, 255, 0),      # Vert
            'Cassé': (0, 0, 255),       # Rouge
            'Endommagé': (0, 165, 255)  # Orange
        }
        
        if model_path:
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """Construire le modèle de détection"""
        print("🔨 Construction du modèle MobileNetV2...")
        
        # Utiliser MobileNetV2 pré-entraîné (léger et rapide)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Geler les couches de base
        base_model.trainable = False
        
        # Ajouter des couches personnalisées
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(3, activation='softmax')(x)  # 3 classes
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compiler
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Modèle construit avec succès!")
        print(f"📊 Classes: {self.class_names}")
    
    def preprocess_frame(self, frame):
        """Prétraiter une image pour le modèle"""
        # Redimensionner
        img = cv2.resize(frame, (224, 224))
        # Convertir BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normaliser
        img = preprocess_input(img)
        # Ajouter dimension batch
        img = np.expand_dims(img, axis=0)
        return img
    
    def detect_defect(self, frame):
        """Détecter les défauts dans une image"""
        # Prétraiter
        processed = self.preprocess_frame(frame)
        
        # Prédiction
        predictions = self.model.predict(processed, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx] * 100
        
        return {
            'class': self.class_names[class_idx],
            'confidence': confidence,
            'probabilities': {
                name: prob * 100 
                for name, prob in zip(self.class_names, predictions[0])
            }
        }
    
    def draw_results(self, frame, result):
        """Dessiner les résultats sur l'image"""
        h, w = frame.shape[:2]
        
        # Classe et confiance
        class_name = result['class']
        confidence = result['confidence']
        color = self.colors[class_name]
        
        # Rectangle de fond pour le texte
        cv2.rectangle(frame, (10, 10), (w-10, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w-10, 150), color, 3)
        
        # Texte principal
        text = f"État: {class_name}"
        cv2.putText(frame, text, (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Confiance
        conf_text = f"Confiance: {confidence:.1f}%"
        cv2.putText(frame, conf_text, (30, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Probabilités détaillées
        y_offset = 170
        for name, prob in result['probabilities'].items():
            prob_text = f"{name}: {prob:.1f}%"
            prob_color = self.colors[name]
            cv2.putText(frame, prob_text, (30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, prob_color, 2)
            y_offset += 30
        
        # Barre de progression
        bar_width = int((w - 40) * (confidence / 100))
        cv2.rectangle(frame, (20, 120), (20 + bar_width, 140), color, -1)
        
        # Timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (30, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run_camera(self, camera_id=0, save_screenshots=False):
        """Lancer la détection en temps réel"""
        print(f"📹 Ouverture de la caméra {camera_id}...")
        
        # Ouvrir la caméra
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la caméra!")
            print("💡 Vérifiez que votre webcam est connectée")
            return
        
        # Configuration de la caméra
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Caméra ouverte avec succès!")
        print("\n" + "="*60)
        print("🎯 INSTRUCTIONS:")
        print("   - Montrez un objet devant la caméra")
        print("   - Appuyez sur 'S' pour sauvegarder une capture")
        print("   - Appuyez sur 'Q' pour quitter")
        print("="*60 + "\n")
        
        screenshot_count = 0
        
        while True:
            # Lire une frame
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Erreur de lecture de la caméra")
                break
            
            # Détecter les défauts
            result = self.detect_defect(frame)
            
            # Dessiner les résultats
            frame_with_results = self.draw_results(frame.copy(), result)
            
            # Afficher
            cv2.imshow('Détection de Défauts - Sougui.TN', frame_with_results)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n👋 Fermeture de l'application...")
                break
            
            elif key == ord('s') or key == ord('S'):
                if save_screenshots:
                    filename = f"screenshot_{screenshot_count:04d}.jpg"
                    cv2.imwrite(filename, frame_with_results)
                    screenshot_count += 1
                    print(f"📸 Capture sauvegardée: {filename}")
        
        # Libérer les ressources
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Caméra fermée")
    
    def save_model(self, filepath='defect_detector_model.h5'):
        """Sauvegarder le modèle"""
        self.model.save(filepath)
        print(f"💾 Modèle sauvegardé: {filepath}")
    
    def load_model(self, filepath):
        """Charger un modèle sauvegardé"""
        self.model = keras.models.load_model(filepath)
        print(f"📂 Modèle chargé: {filepath}")


def main():
    """Fonction principale"""
    print("="*60)
    print("🏺 DÉTECTEUR DE DÉFAUTS - SOUGUI.TN")
    print("   Contrôle Qualité Artisanat Tunisien")
    print("="*60)
    
    # Créer le détecteur
    detector = DefectDetector()
    
    # Lancer la caméra
    detector.run_camera(
        camera_id=0,  # 0 = webcam par défaut
        save_screenshots=True
    )


if __name__ == "__main__":
    main()
