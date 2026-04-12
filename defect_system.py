import cv2
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================
# CONFIGURATION
# ==============================
IMG_SIZE = (224, 224)
CLASSES = ['Intact', 'Casse', 'Endommage']
DATA_DIR = "training_data"

# ==============================
# 1. COLLECTE DES DONNÉES
# ==============================
def collect_data(num_samples=100):
    for cls in CLASSES:
        os.makedirs(f"{DATA_DIR}/{cls}", exist_ok=True)

    cap = cv2.VideoCapture(0)
    counters = {cls: 0 for cls in CLASSES}

    print("\n📸 Appuie sur:")
    print("1 → Intact | 2 → Cassé | 3 → Endommagé | Q → Quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        y = 30
        for cls in CLASSES:
            txt = f"{cls}: {counters[cls]}/{num_samples}"
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            y += 30

        cv2.imshow("Collecte", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        for i, cls in enumerate(CLASSES):
            if key == ord(str(i+1)):
                if counters[cls] < num_samples:
                    path = f"{DATA_DIR}/{cls}/{cls}_{counters[cls]}.jpg"
                    cv2.imwrite(path, frame)
                    counters[cls] += 1
                    print("Saved:", path)

    cap.release()
    cv2.destroyAllWindows()


# ==============================
# 2. CONSTRUIRE MODELE
# ==============================
def build_model():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(CLASSES), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ==============================
# 3. ENTRAINEMENT
# ==============================
def train_model():
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        subset='validation'
    )

    model = build_model()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20
    )

    model.save("model.h5")
    print("✅ Modèle sauvegardé")

    return model


# ==============================
# 4. TEST CAMERA
# ==============================
def run_camera(model):
    cap = cv2.VideoCapture(0)

    colors = {
        'Intact': (0,255,0),
        'Casse': (0,0,255),
        'Endommage': (0,165,255)
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img, verbose=0)[0]
        idx = np.argmax(preds)
        conf = preds[idx] * 100
        label = CLASSES[idx]

        # seuil de confiance
        if conf < 60:
            label = "Incertain"

        color = colors.get(label, (255,255,255))

        text = f"{label} ({conf:.1f}%)"
        cv2.putText(frame, text, (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    print("\n1 = Collecte des données")
    print("2 = Entraîner le modèle")
    print("3 = Tester avec caméra")

    choice = input("Choix: ")

    if choice == "1":
        collect_data(100)

    elif choice == "2":
        train_model()

    elif choice == "3":
        model = keras.models.load_model("model.h5")
        run_camera(model)